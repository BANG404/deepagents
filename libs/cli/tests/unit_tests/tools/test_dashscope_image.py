"""Unit tests for the generate_image tool.

All tests are fully offline — no real HTTP calls are made. httpx is patched at
the transport level so the actual request/response cycle is exercised without
touching the network.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.stores import InMemoryStore

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from langchain_core.tools import StructuredTool

from deepagents_cli.dashscope_image import (
    _build_payload,
    _build_tool_message,
    _download_as_base64,
    _extract_image_url,
    _infer_mime_type,
    _resolve_save_path,
    _save_image,
    generate_image_tool,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_runtime(tool_call_id: str = "test-call-1") -> ToolRuntime[None, None]:
    """Create a minimal `ToolRuntime` for use in tool invocations."""
    return ToolRuntime(
        state=None,
        context=None,
        tool_call_id=tool_call_id,
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # minimal fake PNG header
FAKE_BASE64 = base64.b64encode(FAKE_IMAGE_BYTES).decode("utf-8")
FAKE_IMAGE_URL = "https://example.com/generated.png"
FAKE_REVISED_PROMPT = "A serene mountain lake at dawn, oil painting style, ultra detail"


def _make_dashscope_response(
    image_url: str = FAKE_IMAGE_URL,
    revised_prompt: str | None = FAKE_REVISED_PROMPT,
) -> dict[str, Any]:
    """Return a minimal DashScope-shaped response dict."""
    content: list[dict[str, str]] = [{"image": image_url}]
    if revised_prompt:
        content.append({"text": revised_prompt})
    return {
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        },
        "usage": {"image_count": 1},
        "request_id": "fake-request-id",
    }


class _FakeTransport(httpx.BaseTransport):
    """Fake synchronous transport that serves pre-canned responses in order."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:  # noqa: ARG002
        return next(self._responses)


class _FakeAsyncTransport(httpx.AsyncBaseTransport):
    """Fake async transport that serves pre-canned responses in order."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:  # noqa: ARG002
        return next(self._responses)


def _json_response(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=data)


def _bytes_response(content: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, content=content)


# Capture the real constructors before any patching so factories can use them
# without triggering recursion when `httpx.Client` itself is replaced by a mock.
_RealClient = httpx.Client
_RealAsyncClient = httpx.AsyncClient


def _make_sync_client_class(responses: list[httpx.Response]) -> type:
    """Return a drop-in `httpx.Client` factory backed by pre-canned responses.

    Each call to the factory (i.e. each `httpx.Client(...)` in production code)
    returns a fresh context-manager wrapper around a brand-new real client backed
    by the fake transport, so re-entering after close is never an issue.
    """
    transport = _FakeTransport(responses)

    class _FakeClientFactory:
        """Mimics `httpx.Client(...)` — ignores all constructor kwargs."""

        def __new__(cls, **_kwargs: object) -> object:
            # Use the real constructor to avoid recursing into the patch.
            client = _RealClient(transport=transport)

            class _CM:
                def __enter__(self_inner) -> httpx.Client:  # noqa: N805
                    return client.__enter__()

                def __exit__(self_inner, *args: object) -> None:  # noqa: N805
                    client.__exit__(*args)

            return _CM()

    return _FakeClientFactory


def _make_async_client_class(responses: list[httpx.Response]) -> type:
    """Return a drop-in `httpx.AsyncClient` factory backed by pre-canned responses."""
    transport = _FakeAsyncTransport(responses)

    class _FakeAsyncClientFactory:
        """Mimics `httpx.AsyncClient(...)` — ignores all constructor kwargs."""

        def __new__(cls, **_kwargs: object) -> object:
            client = _RealAsyncClient(transport=transport)

            class _ACM:
                async def __aenter__(self_inner) -> httpx.AsyncClient:  # noqa: N805
                    return await client.__aenter__()

                async def __aexit__(self_inner, *args: object) -> None:  # noqa: N805
                    await client.__aexit__(*args)

            return _ACM()

    return _FakeAsyncClientFactory


def _make_sync_client_class_with_transport(transport: httpx.BaseTransport) -> type:
    """Return a drop-in `httpx.Client` factory wrapping a custom transport.

    Unlike `_make_sync_client_class`, this helper accepts an already-constructed
    transport so callers can instrument it (e.g. to capture requests) before
    passing it in.
    """

    class _Factory:
        def __new__(cls, **_kw: object) -> object:
            client = _RealClient(transport=transport)

            class _CM:
                def __enter__(self_inner) -> httpx.Client:  # noqa: N805
                    return client.__enter__()

                def __exit__(self_inner, *args: object) -> None:  # noqa: N805
                    client.__exit__(*args)

            return _CM()

    return _Factory


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_text_only_prompt(self) -> None:
        payload = _build_payload(
            "a red apple",
            model="qwen-image-2.0-pro",
            size="1024*1024",
            reference_image_url=None,
            n=1,
            prompt_extend=True,
            negative_prompt=None,
            watermark=False,
        )
        messages = payload["input"]["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # Only the text block — no image block
        assert content == [{"text": "a red apple"}]

    def test_reference_image_prepended(self) -> None:
        payload = _build_payload(
            "edit this image",
            model="qwen-image-2.0-pro",
            size="1024*1024",
            reference_image_url="https://example.com/ref.jpg",
            n=1,
            prompt_extend=False,
            negative_prompt=None,
            watermark=True,
        )
        content = payload["input"]["messages"][0]["content"]
        assert content[0] == {"image": "https://example.com/ref.jpg"}
        assert content[1] == {"text": "edit this image"}

    def test_negative_prompt_included(self) -> None:
        payload = _build_payload(
            "sunset",
            model="qwen-image-2.0-pro",
            size="512*512",
            reference_image_url=None,
            n=2,
            prompt_extend=True,
            negative_prompt="blurry, low quality",
            watermark=False,
        )
        assert payload["parameters"]["negative_prompt"] == "blurry, low quality"
        assert payload["parameters"]["n"] == 2

    def test_negative_prompt_omitted_when_none(self) -> None:
        payload = _build_payload(
            "sunset",
            model="qwen-image-2.0-pro",
            size="512*512",
            reference_image_url=None,
            n=1,
            prompt_extend=True,
            negative_prompt=None,
            watermark=False,
        )
        assert "negative_prompt" not in payload["parameters"]

    def test_model_and_size_forwarded(self) -> None:
        payload = _build_payload(
            "test",
            model="custom-model",
            size="2048*2048",
            reference_image_url=None,
            n=1,
            prompt_extend=False,
            negative_prompt=None,
            watermark=True,
        )
        assert payload["model"] == "custom-model"
        assert payload["parameters"]["size"] == "2048*2048"
        assert payload["parameters"]["watermark"] is True
        assert payload["parameters"]["prompt_extend"] is False


# ---------------------------------------------------------------------------
# _extract_image_url
# ---------------------------------------------------------------------------


class TestExtractImageUrl:
    def test_extracts_url_and_revised_prompt(self) -> None:
        response = _make_dashscope_response()
        url, prompt = _extract_image_url(response)
        assert url == FAKE_IMAGE_URL
        assert prompt == FAKE_REVISED_PROMPT

    def test_extracts_url_without_revised_prompt(self) -> None:
        response = _make_dashscope_response(revised_prompt=None)
        url, prompt = _extract_image_url(response)
        assert url == FAKE_IMAGE_URL
        assert prompt is None

    def test_raises_on_empty_choices(self) -> None:
        response: dict[str, Any] = {"output": {"choices": []}}
        with pytest.raises(ValueError, match="no choices"):
            _extract_image_url(response)

    def test_raises_when_no_image_block(self) -> None:
        response: dict[str, Any] = {
            "output": {
                "choices": [{"message": {"content": [{"text": "only text, no image"}]}}]
            }
        }
        with pytest.raises(ValueError, match="no image URL"):
            _extract_image_url(response)

    def test_raises_on_missing_output_key(self) -> None:
        with pytest.raises(ValueError, match="no choices"):
            _extract_image_url({})


# ---------------------------------------------------------------------------
# _download_as_base64
# ---------------------------------------------------------------------------


class TestDownloadAsBase64:
    def test_returns_base64_encoded_content(self) -> None:
        transport = _FakeTransport([_bytes_response(FAKE_IMAGE_BYTES)])
        with httpx.Client(transport=transport) as client:
            result = _download_as_base64(FAKE_IMAGE_URL, client=client)
        assert result == FAKE_BASE64

    def test_raises_on_http_error(self) -> None:
        transport = _FakeTransport([httpx.Response(status_code=404)])
        with (
            httpx.Client(transport=transport) as client,
            pytest.raises(httpx.HTTPStatusError),
        ):
            _download_as_base64(FAKE_IMAGE_URL, client=client)


# ---------------------------------------------------------------------------
# _save_image
# ---------------------------------------------------------------------------


class TestSaveImage:
    def test_writes_file_and_returns_absolute_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "sub" / "out.png"
        returned = _save_image(FAKE_BASE64, str(dest))
        assert Path(returned).is_file()
        assert Path(dest).read_bytes() == FAKE_IMAGE_BYTES

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "image.png"
        _save_image(FAKE_BASE64, str(dest))
        assert dest.exists()

    def test_returned_path_is_absolute(self, tmp_path: Path) -> None:
        dest = tmp_path / "img.png"
        returned = _save_image(FAKE_BASE64, str(dest))
        assert Path(returned).is_absolute()


# ---------------------------------------------------------------------------
# _resolve_save_path
# ---------------------------------------------------------------------------


class TestResolveSavePath:
    def test_none_returns_none(self) -> None:
        assert _resolve_save_path(None, "/workspace") is None

    def test_absolute_path_unchanged(self) -> None:
        result = _resolve_save_path("/abs/path/img.png", "/workspace")
        assert result == "/abs/path/img.png"

    def test_relative_joined_with_default_dir(self) -> None:
        result = _resolve_save_path("img.png", "/workspace")
        assert result == "/workspace/img.png"

    def test_relative_no_default_dir(self) -> None:
        # Without a default_save_dir, relative paths pass through as-is
        result = _resolve_save_path("img.png", None)
        assert result == "img.png"


# ---------------------------------------------------------------------------
# generate_image_tool — sync
# ---------------------------------------------------------------------------


class TestGenerateImageToolSync:
    def _make_tool(self) -> StructuredTool:
        return generate_image_tool()

    def _patch_sync(
        self, responses: list[httpx.Response]
    ) -> AbstractContextManager[None]:  # type: ignore[type-arg]
        return patch(
            "deepagents_cli.dashscope_image.httpx.Client",
            _make_sync_client_class(responses),
        )

    def test_returns_tool_message_with_image_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = self._make_tool()
        runtime = _make_runtime("call-sync-1")
        with self._patch_sync(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = tool.invoke({"prompt": "a mountain lake", "runtime": runtime})
        assert isinstance(result, ToolMessage)
        assert result.name == "generate_image"
        assert result.tool_call_id == "call-sync-1"
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "image"
        assert result.content[0]["base64"] == FAKE_BASE64
        assert result.additional_kwargs["saved_path"] is None
        assert result.additional_kwargs["model"] == "qwen-image-2.0-pro"
        assert result.additional_kwargs["revised_prompt"] == FAKE_REVISED_PROMPT

    def test_saves_to_disk_when_save_path_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        dest = str(tmp_path / "out.png")
        tool = self._make_tool()
        runtime = _make_runtime("call-sync-save")
        with self._patch_sync(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = tool.invoke(
                {"prompt": "a mountain lake", "save_path": dest, "runtime": runtime}
            )
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["saved_path"] == dest
        assert Path(dest).read_bytes() == FAKE_IMAGE_BYTES

    def test_returns_error_string_when_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        tool = self._make_tool()
        runtime = _make_runtime()
        result = tool.invoke({"prompt": "test", "runtime": runtime})
        assert isinstance(result, str)
        assert "DASHSCOPE_API_KEY" in result

    def test_returns_error_string_on_http_4xx(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = self._make_tool()
        runtime = _make_runtime()
        with self._patch_sync([httpx.Response(status_code=401, text="Unauthorized")]):
            result = tool.invoke({"prompt": "test", "runtime": runtime})
        assert isinstance(result, str)
        assert "401" in result

    def test_override_model_used_in_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = self._make_tool()
        runtime = _make_runtime()
        with self._patch_sync(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = tool.invoke(
                {
                    "prompt": "a lake",
                    "override_model": "custom-model-v1",
                    "runtime": runtime,
                }
            )
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["model"] == "custom-model-v1"

    def test_default_size_is_1024(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        captured_bodies: list[bytes] = []

        class _CapturingTransport(httpx.BaseTransport):
            def __init__(self) -> None:
                self._inner = _FakeTransport(
                    [
                        _json_response(_make_dashscope_response()),
                        _bytes_response(FAKE_IMAGE_BYTES),
                    ]
                )

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                if "multimodal-generation" in str(request.url):
                    captured_bodies.append(bytes(request.content))
                return self._inner.handle_request(request)

        capturing_cls = _make_sync_client_class_with_transport(_CapturingTransport())
        tool = self._make_tool()
        with patch("deepagents_cli.dashscope_image.httpx.Client", capturing_cls):
            tool.invoke({"prompt": "test", "runtime": _make_runtime()})

        assert captured_bodies, (
            "Expected at least one generation request to be captured"
        )
        body = json.loads(captured_bodies[0])
        assert body["parameters"]["size"] == "1024*1024"

    def test_watermark_false_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        captured_bodies: list[bytes] = []

        class _CapturingTransport(httpx.BaseTransport):
            def __init__(self) -> None:
                self._inner = _FakeTransport(
                    [
                        _json_response(_make_dashscope_response()),
                        _bytes_response(FAKE_IMAGE_BYTES),
                    ]
                )

            def handle_request(self, request: httpx.Request) -> httpx.Response:
                if "multimodal-generation" in str(request.url):
                    captured_bodies.append(bytes(request.content))
                return self._inner.handle_request(request)

        capturing_cls = _make_sync_client_class_with_transport(_CapturingTransport())
        tool = self._make_tool()
        with patch("deepagents_cli.dashscope_image.httpx.Client", capturing_cls):
            tool.invoke({"prompt": "no watermark", "runtime": _make_runtime()})

        body = json.loads(captured_bodies[0])
        assert body["parameters"]["watermark"] is False


# ---------------------------------------------------------------------------
# generate_image_tool — async
# ---------------------------------------------------------------------------


class TestGenerateImageToolAsync:
    def _make_tool(self) -> StructuredTool:
        return generate_image_tool()

    def _patch_async(
        self, responses: list[httpx.Response]
    ) -> AbstractContextManager[None]:  # type: ignore[type-arg]
        return patch(
            "deepagents_cli.dashscope_image.httpx.AsyncClient",
            _make_async_client_class(responses),
        )

    async def test_async_returns_tool_message_with_image_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = self._make_tool()
        runtime = _make_runtime("call-async-1")
        with self._patch_async(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = await tool.ainvoke(
                {"prompt": "a mountain lake", "runtime": runtime}
            )
        assert isinstance(result, ToolMessage)
        assert result.name == "generate_image"
        assert result.tool_call_id == "call-async-1"
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "image"
        assert result.content[0]["base64"] == FAKE_BASE64
        assert result.additional_kwargs["model"] == "qwen-image-2.0-pro"

    async def test_async_saves_to_disk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        dest = str(tmp_path / "async_out.png")
        tool = self._make_tool()
        runtime = _make_runtime("call-async-save")
        with self._patch_async(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = await tool.ainvoke(
                {"prompt": "a mountain lake", "save_path": dest, "runtime": runtime}
            )
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["saved_path"] == dest
        assert Path(dest).read_bytes() == FAKE_IMAGE_BYTES  # noqa: ASYNC240

    async def test_async_returns_error_on_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        tool = self._make_tool()
        runtime = _make_runtime()
        result = await tool.ainvoke({"prompt": "test", "runtime": runtime})
        assert isinstance(result, str)
        assert "DASHSCOPE_API_KEY" in result

    async def test_async_returns_error_on_http_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = self._make_tool()
        runtime = _make_runtime()
        with self._patch_async([httpx.Response(status_code=429, text="Rate limited")]):
            result = await tool.ainvoke({"prompt": "test", "runtime": runtime})
        assert isinstance(result, str)
        assert "429" in result


# ---------------------------------------------------------------------------
# generate_image_tool — default_save_dir factory option
# ---------------------------------------------------------------------------


class TestGenerateImageToolDefaultSaveDir:
    def _patch_sync(
        self, responses: list[httpx.Response]
    ) -> AbstractContextManager[None]:  # type: ignore[type-arg]
        return patch(
            "deepagents_cli.dashscope_image.httpx.Client",
            _make_sync_client_class(responses),
        )

    def test_relative_save_path_resolved_against_default_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        tool = generate_image_tool(default_save_dir=str(tmp_path))
        runtime = _make_runtime()
        with self._patch_sync(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = tool.invoke(
                {"prompt": "test", "save_path": "relative.png", "runtime": runtime}
            )
        expected = str(tmp_path / "relative.png")
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["saved_path"] == expected
        assert Path(expected).exists()

    def test_absolute_save_path_ignores_default_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        abs_dest = str(tmp_path / "absolute.png")
        tool = generate_image_tool(default_save_dir=str(other_dir))
        runtime = _make_runtime()
        with self._patch_sync(
            [
                _json_response(_make_dashscope_response()),
                _bytes_response(FAKE_IMAGE_BYTES),
            ]
        ):
            result = tool.invoke(
                {"prompt": "test", "save_path": abs_dest, "runtime": runtime}
            )
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["saved_path"] == abs_dest


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestGenerateImageToolMetadata:
    def test_tool_name(self) -> None:
        tool = generate_image_tool()
        assert tool.name == "generate_image"

    def test_tool_has_description(self) -> None:
        tool = generate_image_tool()
        assert tool.description
        assert "DashScope" in tool.description

    def test_tool_has_async_coroutine(self) -> None:
        tool = generate_image_tool()
        # StructuredTool exposes coroutine attribute when one is provided
        assert tool.coroutine is not None


# ---------------------------------------------------------------------------
# _infer_mime_type
# ---------------------------------------------------------------------------


class TestInferMimeType:
    def test_png_url(self) -> None:
        assert _infer_mime_type("https://example.com/image.png") == "image/png"

    def test_jpg_url(self) -> None:
        assert _infer_mime_type("https://example.com/photo.jpg") == "image/jpeg"

    def test_jpeg_url(self) -> None:
        assert _infer_mime_type("https://example.com/photo.jpeg") == "image/jpeg"

    def test_gif_url(self) -> None:
        assert _infer_mime_type("https://example.com/anim.gif") == "image/gif"

    def test_webp_url(self) -> None:
        assert _infer_mime_type("https://example.com/img.webp") == "image/webp"

    def test_unknown_extension_falls_back_to_png(self) -> None:
        assert _infer_mime_type("https://example.com/img.bmp") == "image/png"

    def test_no_extension_falls_back_to_png(self) -> None:
        assert _infer_mime_type("https://example.com/image") == "image/png"

    def test_uppercase_extension_normalised(self) -> None:
        assert _infer_mime_type("https://example.com/image.PNG") == "image/png"


# ---------------------------------------------------------------------------
# _build_tool_message
# ---------------------------------------------------------------------------


class TestBuildToolMessage:
    def _make_result(self, saved_path: str | None = None) -> dict:  # type: ignore[type-arg]
        from deepagents_cli.dashscope_image import GenerateImageResult

        return GenerateImageResult(
            base64=FAKE_BASE64,
            saved_path=saved_path,
            model="qwen-image-2.0-pro",
            revised_prompt=FAKE_REVISED_PROMPT,
        )

    def test_returns_tool_message(self) -> None:
        result = _build_tool_message(
            self._make_result(), image_url=FAKE_IMAGE_URL, tool_call_id="tc-1"
        )
        assert isinstance(result, ToolMessage)

    def test_name_is_generate_image(self) -> None:
        result = _build_tool_message(
            self._make_result(), image_url=FAKE_IMAGE_URL, tool_call_id="tc-1"
        )
        assert result.name == "generate_image"

    def test_tool_call_id_attached(self) -> None:
        result = _build_tool_message(
            self._make_result(), image_url=FAKE_IMAGE_URL, tool_call_id="my-id"
        )
        assert result.tool_call_id == "my-id"

    def test_content_is_image_block(self) -> None:
        result = _build_tool_message(
            self._make_result(), image_url=FAKE_IMAGE_URL, tool_call_id="tc-1"
        )
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        block = result.content[0]
        assert block["type"] == "image"
        assert block["base64"] == FAKE_BASE64

    def test_mime_type_inferred_from_url(self) -> None:
        result = _build_tool_message(
            self._make_result(),
            image_url="https://example.com/out.webp",
            tool_call_id="tc-1",
        )
        assert result.content[0]["mime_type"] == "image/webp"
        assert result.additional_kwargs["mime_type"] == "image/webp"

    def test_additional_kwargs_contain_metadata(self) -> None:
        result = _build_tool_message(
            self._make_result(saved_path="/tmp/x.png"),
            image_url=FAKE_IMAGE_URL,
            tool_call_id="tc-1",
        )
        assert result.additional_kwargs["saved_path"] == "/tmp/x.png"
        assert result.additional_kwargs["model"] == "qwen-image-2.0-pro"
        assert result.additional_kwargs["revised_prompt"] == FAKE_REVISED_PROMPT
