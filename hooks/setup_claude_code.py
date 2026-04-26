#!/usr/bin/env python3
"""Setup script: configure Claude Code hooks for GitMem0 auto-memory.

Run this once to enable automatic memory injection and extraction.
"""
import json
import sys
from pathlib import Path


def get_claude_settings_path(project_dir: Path | None = None) -> Path:
    """Find the Claude Code settings file.

    Args:
        project_dir: Project directory to check for .claude/ dir.
                     Defaults to Path.cwd() if not provided.
    """
    project_dir = project_dir or Path.cwd()
    project_settings = project_dir / ".claude" / "settings.json"
    if project_settings.parent.exists():
        return project_settings
    user_settings = Path.home() / ".claude" / "settings.json"
    return user_settings


def install_hooks(project_dir: Path | None = None) -> dict:
    """Install Stop hook for memory extraction.

    Args:
        project_dir: Project directory. Defaults to Path.cwd().

    Returns:
        {"hooks_installed": True, "settings_path": str}

    Idempotent: overwrites hooks section cleanly.
    """
    settings_path = get_claude_settings_path(project_dir)
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            settings = {}

    hooks_dir = Path(__file__).parent.resolve()
    extract_script = str(hooks_dir / "post_response.py")

    if "hooks" not in settings:
        settings["hooks"] = {}

    # Stop hook: extract memories when AI finishes responding
    settings["hooks"]["Stop"] = [
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f'python "{extract_script}"'
                }
            ]
        }
    ]

    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"hooks_installed": True, "settings_path": str(settings_path)}


def setup():
    """Backward-compatible wrapper for CLI usage."""
    result = install_hooks()
    print(f"GitMem0 hooks configured in: {result['settings_path']}")
    print("  - Stop: auto-extract memories when AI finishes responding")
    print("\nRestart Claude Code to activate hooks.")
    print("\nNote: For auto-context injection, add the memory system prompt")
    print("to your CLAUDE.md or system prompt. See gitmem0/prompt.py.")


if __name__ == "__main__":
    setup()
