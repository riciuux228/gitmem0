#!/usr/bin/env python3
"""Post-response hook: auto-extract memories from conversation.

Connects to GitMem0 daemon (no model loading).
Filters out garbage before sending to extraction.
"""
import json
import re
import socket
import sys

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 19840


def _sanitize(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def _is_garbage(text: str) -> bool:
    """Reject text that shouldn't become memories."""
    if len(text) < 20:
        return True
    if len(text) > 5000:
        return True
    # JSON metadata
    if '{"session_id"' in text or '{"transcript_path"' in text:
        return True
    if '"hook_event_name"' in text or '"permission_mode"' in text:
        return True
    # File paths
    if re.search(r'[A-Z]:\\Users\\.*\\\.claude\\', text):
        return True
    # Markdown table fragments (from hook output)
    if text.startswith('| ') and '|' in text[2:5]:
        return True
    # Garbled encoding markers
    if '锛' in text or '鈥' in text or 'utf-8' in text.lower():
        return True
    # Truncated JSON
    if text.startswith('{') and '"ok"' not in text and '"data"' not in text:
        return True
    return False


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
        conversation = _sanitize(" ".join(sys.argv[1:]))
    else:
        conversation = _sanitize(sys.stdin.read().strip())

    if not conversation or _is_garbage(conversation):
        sys.exit(0)

    try:
        result = _send_to_daemon({"action": "extract", "text": conversation, "source": "hook"})
        data = result.get("data", {})
        stored = data.get("memories", [])
        print(json.dumps({
            "hook": "post_response",
            "extracted": len(stored),
            "memories": stored,
        }, ensure_ascii=False))
    except (ConnectionRefusedError, OSError):
        print(json.dumps({
            "hook": "post_response",
            "extracted": 0,
            "_note": "daemon not running",
        }))


if __name__ == "__main__":
    main()
