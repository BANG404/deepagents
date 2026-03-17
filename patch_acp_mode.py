import re

with open('libs/cli/tests/unit_tests/test_main_acp_mode.py', 'r') as f:
    text = f.read()

text = re.sub(
    r'<<<<<<< HEAD\n\s*patch\(\n\s*"deepagents_cli\.config\.settings",\n\s*new=SimpleNamespace\(has_tavily=True, has_dashscope=False\),\n\s*\),\n=======\n\s*patch\("deepagents_cli\.config\.settings", new=SimpleNamespace\(has_tavily=True\)\),\n\s*patch\("deepagents_cli\.model_config\.save_recent_model", return_value=True\),\n>>>>>>> upstream/main',
    r'        patch(\n            "deepagents_cli.config.settings",\n            new=SimpleNamespace(has_tavily=True, has_dashscope=False),\n        ),\n        patch("deepagents_cli.model_config.save_recent_model", return_value=True),',
    text
)

text = re.sub(
    r'<<<<<<< HEAD\n\s*patch\(\n\s*"deepagents_cli\.config\.settings",\n\s*new=SimpleNamespace\(has_tavily=False, has_dashscope=False\),\n\s*\),\n=======\n\s*patch\("deepagents_cli\.config\.settings", new=SimpleNamespace\(has_tavily=False\)\),\n\s*patch\("deepagents_cli\.model_config\.save_recent_model", return_value=True\),\n>>>>>>> upstream/main',
    r'        patch(\n            "deepagents_cli.config.settings",\n            new=SimpleNamespace(has_tavily=False, has_dashscope=False),\n        ),\n        patch("deepagents_cli.model_config.save_recent_model", return_value=True),',
    text
)

with open('libs/cli/tests/unit_tests/test_main_acp_mode.py', 'w') as f:
    f.write(text)
