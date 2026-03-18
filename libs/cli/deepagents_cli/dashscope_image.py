# ruff: noqa: E501
"""`generate_image`: DashScope multimodal image generation tool (CLI integration).

Calls the DashScope `qwen-image-2.0-pro` (or compatible) model to generate an image
from a text prompt, returns the result as a base64-encoded string, and optionally
saves the image to a specified path on the local filesystem.

The API key is read from the `DASHSCOPE_API_KEY` environment variable.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # required at runtime for Annotated resolution
)
from langchain_core.messages import ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import BaseTool, StructuredTool
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

DASHSCOPE_GENERATION_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_MODEL = "qwen-image-2.0-pro"
DEFAULT_SIZE = "1024*1024"
REQUEST_TIMEOUT = 120  # seconds

ImageSize = Literal[
    "256*256",
    "512*512",
    "1024*1024",
    "1280*720",
    "720*1280",
    "2048*2048",
]


class GenerateImageResult(TypedDict):
    """Result returned by the `generate_image` tool.

    Attributes:
        base64: Base64-encoded image bytes (PNG/JPEG/WebP depending on the model).
        saved_path: Absolute path where the image was saved, or `None` if no path
            was requested.
        model: The model that produced the image.
        revised_prompt: The prompt as (optionally) rewritten by the model. May be
            `None` when `prompt_extend=False` or the API does not return one.
    """

    base64: str
    saved_path: str | None
    model: str
    revised_prompt: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_api_key() -> str:
    """Read the DashScope API key from the environment.

    Returns:
        The value of `DASHSCOPE_API_KEY`.

    Raises:
        OSError: If `DASHSCOPE_API_KEY` is not set or is empty.
    """
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        msg = "DASHSCOPE_API_KEY environment variable is not set. Set it to your DashScope API key before using the generate_image tool."
        raise OSError(msg)
    return key


def _build_payload(
    prompt: str,
    *,
    model: str,
    size: str,
    reference_image_url: str | None,
    n: int,
    prompt_extend: bool,
    negative_prompt: str | None,
    watermark: bool,
) -> dict[str, Any]:
    """Build the JSON payload for the DashScope generation API.

    Args:
        prompt: Text description of the image to generate.
        model: DashScope model identifier.
        size: Output image dimensions as `"width*height"`.
        reference_image_url: Optional URL of a reference image to condition on.
        n: Number of images to generate (only the first is returned).
        prompt_extend: Whether to let the model extend/rewrite the prompt.
        negative_prompt: Optional description of content to avoid.
        watermark: Whether to add a watermark to the output.

    Returns:
        Dictionary ready to be serialised as the request body.
    """
    content: list[dict[str, str]] = []
    if reference_image_url:
        content.append({"image": reference_image_url})
    content.append({"text": prompt})

    parameters: dict[str, Any] = {
        "n": n,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
        "size": size,
    }
    if negative_prompt is not None:
        parameters["negative_prompt"] = negative_prompt

    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": parameters,
    }


def _extract_image_url(response_json: dict[str, Any]) -> tuple[str, str | None]:
    """Extract the first image URL and revised prompt from a DashScope response.

    Args:
        response_json: Parsed JSON from the DashScope API response.

    Returns:
        Tuple of `(image_url, revised_prompt)`.

    Raises:
        ValueError: If the response does not contain a usable image URL.
    """
    choices = response_json.get("output", {}).get("choices", [])
    if not choices:
        msg = f"DashScope API returned no choices. Full response: {response_json}"
        raise ValueError(msg)

    first_choice = choices[0]
    message_content = first_choice.get("message", {}).get("content", [])
    image_url: str | None = None
    revised_prompt: str | None = None

    for block in message_content:
        if isinstance(block, dict):
            if "image" in block and image_url is None:
                image_url = block["image"]
            if "text" in block and revised_prompt is None:
                revised_prompt = block["text"]

    if not image_url:
        msg = f"DashScope API response contained no image URL. Full response: {response_json}"
        raise ValueError(msg)

    return image_url, revised_prompt


def _download_as_base64(image_url: str, *, client: httpx.Client) -> str:
    """Download an image from a URL and return its base64-encoded content.

    Args:
        image_url: Publicly accessible image URL.
        client: An already-open `httpx.Client` to use for the request.

    Returns:
        Base64-encoded image bytes as a UTF-8 string.
    """
    response = client.get(image_url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


def _save_image(encoded: str, save_path: str) -> str:
    """Decode a base64 image string and write it to disk.

    Args:
        encoded: Base64-encoded image bytes.
        save_path: Destination file path (may be relative or absolute).

    Returns:
        Resolved absolute path where the file was written.
    """
    resolved = Path(save_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(base64.b64decode(encoded))
    logger.debug("Image saved to %s", resolved)
    return str(resolved)


# ---------------------------------------------------------------------------
# Core synchronous implementation
# ---------------------------------------------------------------------------


def _call_generate_image(
    prompt: str,
    *,
    save_path: str | None,
    model: str,
    size: str,
    reference_image_url: str | None,
    n: int,
    prompt_extend: bool,
    negative_prompt: str | None,
    watermark: bool,
) -> tuple[GenerateImageResult, str]:
    """Call the DashScope image generation API and return a `GenerateImageResult`.

    Args:
        prompt: Text description of the image to generate.
        save_path: Optional filesystem path to save the generated image. If `None`,
            the image is only returned as base64.
        model: DashScope model identifier.
        size: Output image dimensions as `"width*height"`.
        reference_image_url: Optional URL of a reference image to condition on.
        n: Number of images to generate (only the first is used).
        prompt_extend: Whether to let the model extend/rewrite the prompt.
        negative_prompt: Optional description of content to avoid.
        watermark: Whether to add a watermark to the output.

    Returns:
        Tuple of `(GenerateImageResult, image_url)` where `image_url` is the
        original URL returned by DashScope, used downstream to infer MIME type.
    """
    api_key = _read_api_key()
    payload = _build_payload(
        prompt,
        model=model,
        size=size,
        reference_image_url=reference_image_url,
        n=n,
        prompt_extend=prompt_extend,
        negative_prompt=negative_prompt,
        watermark=watermark,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(DASHSCOPE_GENERATION_URL, json=payload, headers=headers)
        resp.raise_for_status()
        response_json: dict[str, Any] = resp.json()
        image_url, revised_prompt = _extract_image_url(response_json)
        encoded = _download_as_base64(image_url, client=client)

    saved: str | None = None
    if save_path:
        saved = _save_image(encoded, save_path)

    return GenerateImageResult(
        base64=encoded,
        saved_path=saved,
        model=model,
        revised_prompt=revised_prompt,
    ), image_url


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


async def _acall_generate_image(
    prompt: str,
    *,
    save_path: str | None,
    model: str,
    size: str,
    reference_image_url: str | None,
    n: int,
    prompt_extend: bool,
    negative_prompt: str | None,
    watermark: bool,
) -> tuple[GenerateImageResult, str]:
    """Async version of `_call_generate_image`.

    Args:
        prompt: Text description of the image to generate.
        save_path: Optional filesystem path to save the generated image. If `None`,
            the image is only returned as base64.
        model: DashScope model identifier.
        size: Output image dimensions as `"width*height"`.
        reference_image_url: Optional URL of a reference image to condition on.
        n: Number of images to generate (only the first is used).
        prompt_extend: Whether to let the model extend/rewrite the prompt.
        negative_prompt: Optional description of content to avoid.
        watermark: Whether to add a watermark to the output.

    Returns:
        Tuple of `(GenerateImageResult, image_url)` where `image_url` is the
        original URL returned by DashScope, used downstream to infer MIME type.
    """
    api_key = _read_api_key()
    payload = _build_payload(
        prompt,
        model=model,
        size=size,
        reference_image_url=reference_image_url,
        n=n,
        prompt_extend=prompt_extend,
        negative_prompt=negative_prompt,
        watermark=watermark,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            DASHSCOPE_GENERATION_URL, json=payload, headers=headers
        )
        resp.raise_for_status()
        response_json: dict[str, Any] = resp.json()
        image_url, revised_prompt = _extract_image_url(response_json)
        img_resp = await client.get(image_url)
        img_resp.raise_for_status()
    encoded = base64.b64encode(img_resp.content).decode("utf-8")

    saved: str | None = None
    if save_path:
        saved = await asyncio.to_thread(_save_image, encoded, save_path)

    return GenerateImageResult(
        base64=encoded,
        saved_path=saved,
        model=model,
        revised_prompt=revised_prompt,
    ), image_url


# ---------------------------------------------------------------------------
# MIME type inference and ToolMessage construction
# ---------------------------------------------------------------------------


def _infer_mime_type(url: str) -> str:
    """Infer the image MIME type from a URL's path extension.

    Falls back to `image/png` when the extension is absent or unrecognised.

    Args:
        url: Image URL to inspect.

    Returns:
        MIME type string, e.g. `"image/jpeg"`.
    """
    ext = Path(urlparse(url).path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def _build_tool_message(
    result: GenerateImageResult,
    *,
    image_url: str,
    tool_call_id: str,
) -> ToolMessage:
    """Wrap a `GenerateImageResult` in a `ToolMessage` with an image content block.

    Args:
        result: The raw generation result containing base64 image data.
        image_url: The original image URL, used to infer the MIME type.
        tool_call_id: The tool call ID to attach to the message.

    Returns:
        A `ToolMessage` whose content is a single image block plus metadata in
        `additional_kwargs`.
    """
    mime_type = _infer_mime_type(image_url)
    return ToolMessage(
        content_blocks=[
            create_image_block(base64=result["base64"], mime_type=mime_type)
        ],
        name="generate_image",
        tool_call_id=tool_call_id,
        additional_kwargs={
            "saved_path": result["saved_path"],
            "model": result["model"],
            "revised_prompt": result["revised_prompt"],
            "mime_type": mime_type,
        },
    )


# ---------------------------------------------------------------------------
# Tool description
# ---------------------------------------------------------------------------

GENERATE_IMAGE_TOOL_DESCRIPTION = """\
Generate an image using the DashScope multimodal image generation API (Qwen Image model).

The tool calls the DashScope API, downloads the generated image, and returns the image
as a multimodal content block so the model can see it directly.  Metadata such as
`saved_path`, `model`, and `revised_prompt` are available in `additional_kwargs`.

The API key is read from the `DASHSCOPE_API_KEY` environment variable.

Usage notes:
- Provide a descriptive `prompt` in any language supported by the model.
- Set `save_path` to persist the image to disk (e.g. `/workspace/output.png`).
  The parent directories are created automatically.
- Use `reference_image_url` to supply a publicly accessible image URL that the
  model should use as a visual reference or editing base.
- `size` must be one of the supported dimension strings, e.g. `"1024*1024"` (default)
  or `"2048*2048"`.
- `negative_prompt` describes content the model should avoid.
- `prompt_extend=True` (default) allows the model to enrich the prompt for better results.
- Only the first generated image is returned even when `n > 1`.

Example:
  generate_image(
      prompt="A serene mountain lake at dawn, oil painting style",
      save_path="/workspace/lake.png",
      size="1024*1024",
  )
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def generate_image_tool(
    *,
    model: str = DEFAULT_MODEL,
    default_save_dir: str | None = None,
) -> BaseTool:
    """Create and return a `generate_image` `BaseTool` instance.

    The returned tool is a `StructuredTool` that exposes both synchronous and
    asynchronous implementations.

    Args:
        model: DashScope model identifier to use by default. Individual calls can
            override this via the `override_model` parameter on the tool itself.
        default_save_dir: Optional directory to prepend when the caller supplies a
            relative `save_path`. When `None` (the default), relative paths are
            resolved against the current working directory.

    Returns:
        A configured `StructuredTool` named `generate_image`.
    """

    def sync_generate_image(
        prompt: Annotated[str, "Text description of the image to generate."],
        runtime: ToolRuntime[None, None],
        save_path: Annotated[
            str | None,
            "Filesystem path where the generated image should be saved "
            "(e.g. '/workspace/output.png'). Directories are created automatically. "
            "If omitted, the image is saved to the current directory.",
        ] = None,
        size: Annotated[
            ImageSize,
            "Output dimensions as 'width*height'. Supported values: "
            "'256*256', '512*512', '1024*1024' (default), '1280*720', '720*1280', '2048*2048'.",
        ] = DEFAULT_SIZE,  # type: ignore[assignment]
        reference_image_url: Annotated[
            str | None,
            "URL of a publicly accessible reference image to condition the generation on.",
        ] = None,
        n: Annotated[
            int, "Number of candidate images to generate (only the first is returned)."
        ] = 1,
        prompt_extend: Annotated[
            bool, "Allow the model to enrich the prompt for better quality results."
        ] = True,
        negative_prompt: Annotated[
            str | None, "Description of content the model should avoid."
        ] = None,
        watermark: Annotated[
            bool, "Whether to embed a watermark in the generated image."
        ] = False,
        override_model: Annotated[
            str | None, "Override the default model for this single call."
        ] = None,
    ) -> ToolMessage | str:
        """Synchronous wrapper for the generate_image tool.

        Returns:
            `ToolMessage` containing the generated image as a multimodal content
            block, or an error string if the call fails.
        """
        resolved_path = _resolve_save_path(save_path, default_save_dir)
        used_model = override_model or model
        try:
            result, image_url = _call_generate_image(
                prompt,
                save_path=resolved_path,
                model=used_model,
                size=size,
                reference_image_url=reference_image_url,
                n=n,
                prompt_extend=prompt_extend,
                negative_prompt=negative_prompt,
                watermark=watermark,
            )
            return _build_tool_message(
                result, image_url=image_url, tool_call_id=runtime.tool_call_id
            )
        except OSError as e:
            return f"Error: {e}"
        except httpx.HTTPStatusError as e:
            return f"Error: DashScope API returned HTTP {e.response.status_code}: {e.response.text}"
        except ValueError as e:
            return f"Error: {e}"

    async def async_generate_image(
        prompt: Annotated[str, "Text description of the image to generate."],
        runtime: ToolRuntime[None, None],
        save_path: Annotated[
            str | None,
            "Filesystem path where the generated image should be saved "
            "(e.g. '/workspace/output.png'). Directories are created automatically. "
            "If omitted, the image is saved to the current directory.",
        ] = None,
        size: Annotated[
            ImageSize,
            "Output dimensions as 'width*height'. Supported values: "
            "'256*256', '512*512', '1024*1024' (default), '1280*720', '720*1280', '2048*2048'.",
        ] = DEFAULT_SIZE,  # type: ignore[assignment]
        reference_image_url: Annotated[
            str | None,
            "URL of a publicly accessible reference image to condition the generation on.",
        ] = None,
        n: Annotated[
            int, "Number of candidate images to generate (only the first is returned)."
        ] = 1,
        prompt_extend: Annotated[
            bool, "Allow the model to enrich the prompt for better quality results."
        ] = True,
        negative_prompt: Annotated[
            str | None, "Description of content the model should avoid."
        ] = None,
        watermark: Annotated[
            bool, "Whether to embed a watermark in the generated image."
        ] = False,
        override_model: Annotated[
            str | None, "Override the default model for this single call."
        ] = None,
    ) -> ToolMessage | str:
        """Asynchronous wrapper for the generate_image tool.

        Returns:
            `ToolMessage` containing the generated image as a multimodal content
            block, or an error string if the call fails.
        """
        resolved_path = _resolve_save_path(save_path, default_save_dir)
        used_model = override_model or model
        try:
            result, image_url = await _acall_generate_image(
                prompt,
                save_path=resolved_path,
                model=used_model,
                size=size,
                reference_image_url=reference_image_url,
                n=n,
                prompt_extend=prompt_extend,
                negative_prompt=negative_prompt,
                watermark=watermark,
            )
            return _build_tool_message(
                result, image_url=image_url, tool_call_id=runtime.tool_call_id
            )
        except OSError as e:
            return f"Error: {e}"
        except httpx.HTTPStatusError as e:
            return f"Error: DashScope API returned HTTP {e.response.status_code}: {e.response.text}"
        except ValueError as e:
            return f"Error: {e}"

    return StructuredTool.from_function(
        name="generate_image",
        description=GENERATE_IMAGE_TOOL_DESCRIPTION,
        func=sync_generate_image,
        coroutine=async_generate_image,
    )


def _resolve_save_path(save_path: str | None, default_save_dir: str | None) -> str:
    """Resolve a caller-supplied save path against an optional default directory.

    When `save_path` is an absolute path, it is returned unchanged.
    When `save_path` is relative and `default_save_dir` is set, the path is
    joined under `default_save_dir`. If `save_path` is `None`, a default
    filename is generated in the current directory.

    Args:
        save_path: Raw path supplied by the caller.
        default_save_dir: Optional base directory for relative paths.

    Returns:
        Resolved path string.
    """
    import time

    if save_path is None:
        save_path = f"image_{int(time.time())}.png"
    p = Path(save_path)
    if p.is_absolute() or default_save_dir is None:
        return str(p)
    return str(Path(default_save_dir) / p)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SIZE",
    "GENERATE_IMAGE_TOOL_DESCRIPTION",
    "GenerateImageResult",
    "generate_image_tool",
]
