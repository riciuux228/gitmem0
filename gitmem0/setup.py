"""GitMem0 one-click setup — config, DB, daemon, hooks.

All logic is in pure functions for testability. No Typer, no interactive I/O.
Interactive prompts are isolated in prompt_*() functions, skipped in non-interactive mode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

GITMEM0_DIR = Path.home() / ".gitmem0"
CONFIG_PATH = GITMEM0_DIR / "config.toml"
DB_PATH = GITMEM0_DIR / "gitmem0.db"

# Source CLAUDE.md bundled with the package
_CLAUDE_MD_SOURCE = Path(__file__).parent.parent / "CLAUDE.md"


# ── Backend registry ──────────────────────────────────────────────────────────

BACKEND_DEFAULTS: dict[str, dict] = {
    "mimo": {
        "label": "Xiaomi Token Plan (MiMo)",
        "requires_key": True,
        "key_prefix": "tp-",
        "default_model": "MiMo",
    },
    "openai": {
        "label": "OpenAI",
        "requires_key": True,
        "key_prefix": "sk-",
        "default_model": "gpt-4o-mini",
    },
    "claude": {
        "label": "Anthropic Claude",
        "requires_key": True,
        "key_prefix": "sk-ant-",
        "default_model": "claude-haiku-4-5-20251001",
    },
    "ollama": {
        "label": "Ollama (local)",
        "requires_key": False,
        "default_model": "qwen2.5:7b",
    },
}


# ── Data structures ───────────────────────────────────────────────────────────


class Environment(Enum):
    CLAUDE_CODE = "claude_code"
    GENERIC = "generic"


@dataclass
class SetupConfig:
    """Resolved configuration for setup."""
    backend: str = "ollama"
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    db_path: str = str(DB_PATH)
    install_hooks: bool = True
    start_daemon: bool = True
    non_interactive: bool = False


@dataclass
class SetupResult:
    """Result of running setup — returned to CLI for display."""
    config_path: Optional[str] = None
    db_path: Optional[str] = None
    daemon_running: bool = False
    hooks_installed: bool = False
    claude_md_installed: bool = False
    environment: str = "generic"
    backend: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Functions ──────────────────────────────────────────────────────────────────


def detect_environment(cwd: Path | None = None) -> Environment:
    """Detect whether we're inside a Claude Code project.

    Checks for .claude/ directory in cwd.
    """
    cwd = cwd or Path.cwd()
    if (cwd / ".claude").is_dir():
        return Environment.CLAUDE_CODE
    return Environment.GENERIC


def prompt_backend() -> str:
    """Interactive backend picker. Returns backend key."""
    print("\nLLM backend (for smarter memory scoring):")
    backends = list(BACKEND_DEFAULTS.items())
    for i, (key, info) in enumerate(backends, 1):
        extra = " (no key needed)" if not info["requires_key"] else ""
        print(f"  {i}. {info['label']}{extra}")

    while True:
        raw = input(f"\nChoice [1-{len(backends)}] (default: {len(backends)}=Ollama): ").strip()
        if not raw:
            return backends[-1][0]  # default: last = ollama
        try:
            idx = int(raw)
            if 1 <= idx <= len(backends):
                return backends[idx - 1][0]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(backends)}.")


def prompt_api_key(backend: str) -> str:
    """Interactive API key input."""
    info = BACKEND_DEFAULTS.get(backend, {})
    prefix = info.get("key_prefix", "")
    label = info.get("label", backend)
    key = input(f"\nAPI key for {label} (e.g. {prefix}xxxxx): ").strip()
    return key


def write_config(config: SetupConfig, config_path: Path | None = None) -> Path:
    """Write or merge ~/.gitmem0/config.toml.

    Preserves existing non-[llm] sections. Idempotent.
    """
    config_path = config_path or CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config to preserve non-llm sections
    existing: dict = {}
    if config_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            try:
                with open(config_path, "rb") as f:
                    existing = tomllib.load(f)
            except Exception:
                existing = {}

    lines: list[str] = []

    # [llm] section
    lines.append("[llm]")
    lines.append(f'backend = "{config.backend}"')
    lines.append(f'api_key = "{config.api_key}"')
    lines.append(f'model = "{config.model}"')
    if config.base_url:
        lines.append(f'base_url = "{config.base_url}"')

    # Preserve other sections (storage, decay, retrieval, etc.)
    for section_name, section_data in existing.items():
        if section_name == "llm":
            continue
        lines.append("")
        lines.append(f"[{section_name}]")
        if isinstance(section_data, dict):
            for key, val in section_data.items():
                if isinstance(val, str):
                    lines.append(f'{key} = "{val}"')
                elif isinstance(val, bool):
                    lines.append(f'{key} = {"true" if val else "false"}')
                elif isinstance(val, list):
                    items = ", ".join(f'"{v}"' for v in val)
                    lines.append(f'{key} = [{items}]')
                else:
                    lines.append(f'{key} = {val}')

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def init_database(db_path: str | None = None) -> str:
    """Initialize SQLite database via MemoryStore.

    MemoryStore auto-creates schema on first use. Idempotent.
    Returns the db_path used.
    """
    # Lazy import to avoid loading heavy modules at module level
    from gitmem0.store import MemoryStore

    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    MemoryStore(path)
    return str(path)


def start_daemon(timeout: int = 60) -> bool:
    """Start the daemon in background. Returns True if ready.

    Reuses client._start_daemon() logic.
    """
    from gitmem0.client import _start_daemon, _is_daemon_running

    if _is_daemon_running():
        return True
    return _start_daemon()


def verify_daemon() -> dict:
    """Send a stats request to verify daemon is working.

    Returns the daemon response dict.
    """
    from gitmem0.client import _send
    return _send({"action": "stats"})


def install_claude_md(project_dir: Path | None = None) -> Path | None:
    """Copy CLAUDE.md into the project root.

    Returns the path where CLAUDE.md was written, or None if source not found.
    """
    project_dir = project_dir or Path.cwd()
    source = _CLAUDE_MD_SOURCE
    if not source.exists():
        return None
    target = project_dir / "CLAUDE.md"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def run_setup(config: SetupConfig) -> SetupResult:
    """Full setup orchestrator. Returns SetupResult.

    Steps:
    1. detect_environment()
    2. write_config()         — FATAL on failure
    3. init_database()        — FATAL on failure
    4. start_daemon()         — non-fatal
    5. install_hooks()        — non-fatal (only if Claude Code project)
    6. install_claude_md()    — non-fatal (only if Claude Code project)
    7. verify_daemon()        — non-fatal
    """
    result = SetupResult(backend=config.backend)
    cwd = Path.cwd()

    # 1. Detect environment
    env = detect_environment(cwd)
    result.environment = env.value

    # 2. Write config (FATAL)
    try:
        path = write_config(config)
        result.config_path = str(path)
    except Exception as e:
        result.errors.append(f"Config write failed: {e}")
        return result

    # 3. Initialize database (FATAL)
    try:
        db = init_database(config.db_path)
        result.db_path = db
    except Exception as e:
        result.errors.append(f"Database init failed: {e}")
        return result

    # 4. Start daemon (non-fatal)
    if config.start_daemon:
        try:
            result.daemon_running = start_daemon()
            if not result.daemon_running:
                result.warnings.append("Daemon failed to start within timeout")
        except Exception as e:
            result.warnings.append(f"Daemon start error: {e}")

    # 5-6. Install hooks and CLAUDE.md (non-fatal, Claude Code only)
    if config.install_hooks and env == Environment.CLAUDE_CODE:
        try:
            # Lazy import of the refactored hook installer
            sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))
            from setup_claude_code import install_hooks as _install_hooks
            hook_result = _install_hooks(cwd)
            result.hooks_installed = hook_result.get("hooks_installed", False)
        except Exception as e:
            result.warnings.append(f"Hook install error: {e}")
        finally:
            sys.path.pop(0) if str(Path(__file__).parent.parent / "hooks") in sys.path else None

        try:
            claude_md_path = install_claude_md(cwd)
            result.claude_md_installed = claude_md_path is not None
        except Exception as e:
            result.warnings.append(f"CLAUDE.md install error: {e}")

    # 7. Verify daemon (non-fatal)
    if result.daemon_running:
        try:
            resp = verify_daemon()
            if not resp.get("ok", False):
                result.warnings.append(f"Daemon verification failed: {resp.get('error', 'unknown')}")
        except Exception as e:
            result.warnings.append(f"Daemon verification error: {e}")

    return result
