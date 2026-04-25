#!/usr/bin/env python3
"""Pre-response hook: auto-inject relevant memories before AI responds.

Connects to GitMem0 daemon (no model loading).
"""
import json
import os
import socket
import sys
from pathlib import Path

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19840


def _sanitize(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def _send_to_daemon(req: dict) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(30)
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


def main():
    if len(sys.argv) > 1:
        message = _sanitize(" ".join(sys.argv[1:]))
    else:
        message = _sanitize(sys.stdin.read().strip())

    if not message:
        sys.exit(0)

    # Try daemon first
    try:
        result = _send_to_daemon({"action": "query", "message": message})
        data = result.get("data", {})
        context = data.get("context", "")
        if context:
            print(json.dumps({
                "hook": "pre_response",
                "has_memories": True,
                "context": context,
            }, ensure_ascii=False))
        else:
            print(json.dumps({
                "hook": "pre_response",
                "has_memories": False,
            }))
    except (ConnectionRefusedError, OSError):
        # Daemon not running — skip
        print(json.dumps({
            "hook": "pre_response",
            "has_memories": False,
            "_note": "daemon not running",
        }))


if __name__ == "__main__":
    main()
