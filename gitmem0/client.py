"""Thin client for GitMem0 daemon — minimal imports, fast startup.

Usage:
    python -m gitmem0.client '{"action":"query","message":"hello"}'
    python -m gitmem0.client '{"action":"stats"}'
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19840
PID_FILE = Path.home() / ".gitmem0" / "daemon.pid"


def _is_daemon_running() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _cleanup_stale_pid():
    """Remove PID file if the process is no longer running."""
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return  # Process exists
        else:
            os.kill(pid, 0)
            return  # Process exists
    except (ValueError, OSError, ProcessLookupError):
        pass
    # Process doesn't exist, clean up
    PID_FILE.unlink(missing_ok=True)


def _kill_stale_daemon():
    """Kill any process that's holding the daemon port."""
    if not _is_daemon_running():
        return
    # Try to send stop command
    try:
        _send({"action": "stop"})
        for _ in range(5):
            if not _is_daemon_running():
                return
            time.sleep(0.5)
    except Exception:
        pass
    # If still running, kill by PID
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 9 if sys.platform != "win32" else 15)
        except (ValueError, OSError):
            pass
    PID_FILE.unlink(missing_ok=True)


def _send(req: dict) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(60)
    s.connect((DAEMON_HOST, DAEMON_PORT))
    s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
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


def _start_daemon():
    # Clean up stale PID file first
    _cleanup_stale_pid()

    # If port is occupied by something else, try to kill it
    if _is_daemon_running():
        _kill_stale_daemon()
        time.sleep(1)

    # Log file for debugging startup issues
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
    subprocess.Popen([sys.executable, "-m", "gitmem0.auto", "_daemon"], **kwargs)
    for _ in range(60):
        time.sleep(1)
        if _is_daemon_running():
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m gitmem0.client '<json>'"}, ensure_ascii=False))
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "stop":
        if _is_daemon_running():
            _send({"action": "stop"})
            PID_FILE.unlink(missing_ok=True)
            print(json.dumps({"ok": True, "data": {"stopped": True}}, ensure_ascii=False))
        else:
            print(json.dumps({"ok": True, "data": {"stopped": False}}, ensure_ascii=False))
        return

    try:
        req = json.loads(arg)
    except json.JSONDecodeError:
        req = {"action": "query", "message": arg}

    if not _is_daemon_running():
        print(json.dumps({"ok": True, "data": {"_info": "Starting daemon..."}}, ensure_ascii=False))
        if not _start_daemon():
            print(json.dumps({"ok": False, "error": "Failed to start daemon"}, ensure_ascii=False))
            sys.exit(1)

    result = _send(req)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
