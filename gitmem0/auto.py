"""Auto-trigger layer for GitMem0.

Two modes:
  1. Daemon (recommended): python -m gitmem0.auto daemon  — stays resident, model loaded once
  2. One-shot:             python -m gitmem0.auto '<json>' — loads model each time (~30s)

The CLI always tries the daemon first. If daemon isn't running, it starts one
automatically, waits for it to be ready, then sends the request.

Usage:
    # Start daemon (runs in background, loads model once)
    python -m gitmem0.auto daemon

    # Send requests (auto-connects to daemon, or starts one)
    python -m gitmem0.auto '{"action":"query","message":"用户消息"}'
    python -m gitmem0.auto '{"action":"remember","content":"要记住的内容","type":"fact","importance":0.8}'
    python -m gitmem0.auto '{"action":"search","query":"搜索词"}'
    python -m gitmem0.auto '{"action":"extract","text":"对话内容"}'
    python -m gitmem0.auto '{"action":"stats"}'
    python -m gitmem0.auto '{"action":"stop"}'
"""

from __future__ import annotations

import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from gitmem0.context import ContextBuilder
from gitmem0.decay import DecayEngine
from gitmem0.embeddings import EmbeddingEngine
from gitmem0.entities import EntityManager
from gitmem0.extraction import ExtractionEngine
from gitmem0.models import MemoryUnit, MemoryType
from gitmem0.retrieval import RetrievalEngine
from gitmem0.store import MemoryStore
from gitmem0.versioning import VersionControl

# ── Daemon config ──────────────────────────────────────────────────────────
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19840
PID_FILE = Path.home() / ".gitmem0" / "daemon.pid"

# Global singleton — model loads ONCE per process
_instance: Optional["AutoMemory"] = None


class AutoMemory:
    """Automatic memory management — single process, model loaded once."""

    def __init__(self, db_path: Optional[str] = None, llm_judge=None):
        if db_path is None:
            db_path = str(Path.home() / ".gitmem0" / "gitmem0.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.store = MemoryStore(db_path)
        self.embeddings = EmbeddingEngine()
        self.entities = EntityManager(self.store)
        self.retrieval = RetrievalEngine(self.store, self.embeddings)
        self.extraction = ExtractionEngine(
            self.store, self.embeddings, self.entities, llm_judge=llm_judge
        )
        self.context_builder = ContextBuilder(self.retrieval, self.entities)
        self.versioning = VersionControl(self.store)
        self.decay = DecayEngine(self.store, self.embeddings, llm_judge=llm_judge)

    def query(self, message: str, token_budget: int = 1500) -> dict:
        context = self.auto_context(message, token_budget)
        return {"context": context, "has_memories": bool(context)}

    def remember(self, content: str, type: str = "fact", importance: float = 0.5,
                 source: str = "cli", tags: Optional[list] = None) -> dict:
        try:
            mem_type = MemoryType(type.lower())
        except ValueError:
            return {"error": f"Invalid type '{type}'. Use: {', '.join(t.value for t in MemoryType)}"}

        embedding = self.embeddings.embed(content)
        unit = MemoryUnit(
            content=content, type=mem_type, importance=importance,
            source=source, tags=tags or [], embedding=embedding,
        )
        unit = self.entities.link_memory_entities(unit)
        self.store.add_memory(unit)
        return {"id": unit.id, "content": content, "type": mem_type.value, "imp": importance}

    def search(self, query: str, top: int = 3) -> list[dict]:
        results = self.retrieval.search(query, top_n=top)
        return [{"id": m.id, "content": m.content, "type": m.type.value,
                 "imp": round(m.importance, 3), "conf": round(m.confidence, 3)}
                for m in results]

    def extract(self, text: str, source: str = "auto") -> dict:
        candidates = self.extraction.extract_from_text(text, source=source)
        stored = []
        for unit in candidates:
            self.store.add_memory(unit)
            stored.append({"id": unit.id, "content": unit.content, "type": unit.type.value})
        return {"extracted": len(stored), "memories": stored}

    def stats(self) -> dict:
        return self.store.stats()

    def auto_context(self, message: str, token_budget: int = 1500) -> str:
        if not message.strip():
            return ""
        try:
            return self.context_builder.build_context(message, token_budget=token_budget)
        except Exception:
            return ""

    def handle_request(self, req: dict) -> dict:
        """Dispatch a JSON request and return result."""
        action = req.get("action", "query")
        try:
            if action == "query":
                return self.query(req.get("message", ""), req.get("budget", 1500))
            elif action == "remember":
                return self.remember(
                    req["content"], req.get("type", "fact"),
                    req.get("importance", 0.5), req.get("source", "cli"),
                    req.get("tags"),
                )
            elif action == "search":
                return self.search(req.get("query", ""), req.get("top", 3))
            elif action == "extract":
                return self.extract(req.get("text", ""), req.get("source", "auto"))
            elif action == "stats":
                return self.stats()
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}


def _load_llm_judge():
    """Try to create an LLM judge from config or environment."""
    import os

    # 1. Environment variables (highest priority)
    api_key = os.environ.get("GITMEM0_LLM_API_KEY", "")
    base_url = os.environ.get("GITMEM0_LLM_BASE_URL", "")

    # 2. Config file
    if not api_key:
        config_path = Path.home() / ".gitmem0" / "config.toml"
        if config_path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    tomllib = None
            if tomllib is not None:
                with open(config_path, "rb") as f:
                    cfg = tomllib.load(f)
                llm_cfg = cfg.get("llm", {})
                api_key = api_key or llm_cfg.get("api_key", "")
                base_url = base_url or llm_cfg.get("base_url", "")

    if not api_key:
        return None

    if not base_url:
        base_url = "https://token-plan-cn.xiaomimimo.com/v1"

    try:
        from gitmem0.llm_judge import MiMoLLMJudge
        judge = MiMoLLMJudge(api_key=api_key, base_url=base_url)
        if judge.enabled:
            return judge
    except Exception:
        pass
    return None


def get_instance() -> AutoMemory:
    global _instance
    if _instance is None:
        llm_judge = _load_llm_judge()
        _instance = AutoMemory(llm_judge=llm_judge)
    return _instance


# ── Daemon server ──────────────────────────────────────────────────────────

def _handle_client(conn: socket.socket, auto: AutoMemory):
    """Handle a single client connection."""
    try:
        conn.settimeout(60)
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if not data:
            return
        req = json.loads(data.decode("utf-8").strip())
        result = auto.handle_request(req)
        resp = json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)
        conn.sendall((resp + "\n").encode("utf-8"))
    except Exception as e:
        err = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
        try:
            conn.sendall((err + "\n").encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def run_daemon():
    """Run the daemon server. Blocks until stopped."""
    auto = get_instance()
    # Write PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((DAEMON_HOST, DAEMON_PORT))
    server.listen(5)
    server.settimeout(1.0)  # check for shutdown every second

    # Also listen on stdin for "stop" command (for parent process control)
    def _shutdown_handler(sig, frame):
        server.close()
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, _shutdown_handler)

    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=_handle_client, args=(conn, auto), daemon=True)
            t.start()
        except socket.timeout:
            continue
        except OSError:
            break

    PID_FILE.unlink(missing_ok=True)


# ── Daemon client ──────────────────────────────────────────────────────────

def _is_daemon_running() -> bool:
    """Check if daemon is running by attempting a connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _start_daemon_background():
    """Start daemon as a background process."""
    # Clean up stale PID file
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if not handle:
                    PID_FILE.unlink(missing_ok=True)
                else:
                    kernel32.CloseHandle(handle)
            else:
                os.kill(pid, 0)
        except (ValueError, OSError, ProcessLookupError):
            PID_FILE.unlink(missing_ok=True)

    # If port is occupied, try to stop existing daemon
    if _is_daemon_running():
        try:
            _send_to_daemon({"action": "stop"})
            for _ in range(5):
                if not _is_daemon_running():
                    break
                time.sleep(0.5)
        except Exception:
            pass
        time.sleep(1)

    # Log file for debugging
    log_dir = Path.home() / ".gitmem0"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daemon.log"

    kwargs = {
        "stdout": open(log_file, "a"),
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "-m", "gitmem0.auto", "_daemon"],
        **kwargs,
    )
    # Wait for daemon to be ready
    for _ in range(60):  # up to 60 seconds
        time.sleep(1)
        if _is_daemon_running():
            return True
    return False


def _send_to_daemon(req: dict) -> dict:
    """Send a request to the daemon and return the response."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(60)
    s.connect((DAEMON_HOST, DAEMON_PORT))
    payload = json.dumps(req, ensure_ascii=False) + "\n"
    s.sendall(payload.encode("utf-8"))
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    s.close()
    return json.loads(data.decode("utf-8").strip())


def _client_main(req: dict):
    """Send request to daemon, auto-starting if needed."""
    if not _is_daemon_running():
        print(json.dumps({"ok": True, "data": {"_info": "Starting daemon (first call loads model)..."}}, ensure_ascii=False))
        if not _start_daemon_background():
            print(json.dumps({"ok": False, "error": "Failed to start daemon"}, ensure_ascii=False))
            sys.exit(1)
    result = _send_to_daemon(req)
    print(json.dumps(result, ensure_ascii=False, default=str))


# ── CLI entry point ────────────────────────────────────────────────────────

def _main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m gitmem0.auto '<json>' | daemon | stop"}, ensure_ascii=False))
        sys.exit(1)

    arg = sys.argv[1]

    # Special commands
    if arg == "daemon":
        run_daemon()
        return
    if arg == "stop":
        if _is_daemon_running():
            _send_to_daemon({"action": "stop"})
            # Wait for shutdown
            for _ in range(5):
                if not _is_daemon_running():
                    break
                time.sleep(0.5)
            PID_FILE.unlink(missing_ok=True)
            print(json.dumps({"ok": True, "data": {"stopped": True}}, ensure_ascii=False))
        else:
            print(json.dumps({"ok": True, "data": {"stopped": False, "reason": "not running"}}, ensure_ascii=False))
        return

    # Try JSON input
    try:
        req = json.loads(arg)
    except json.JSONDecodeError:
        # Fallback: treat as query message
        req = {"action": "query", "message": arg}

    # Handle "stop" action
    if req.get("action") == "stop":
        _main.__wrapped__() if hasattr(_main, "__wrapped__") else None
        # Direct stop
        if _is_daemon_running():
            _send_to_daemon(req)
            PID_FILE.unlink(missing_ok=True)
            print(json.dumps({"ok": True, "data": {"stopped": True}}, ensure_ascii=False))
        else:
            print(json.dumps({"ok": True, "data": {"stopped": False, "reason": "not running"}}, ensure_ascii=False))
        return

    _client_main(req)


if __name__ == "__main__":
    _main()
