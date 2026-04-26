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


def _is_garbage_line(line: str) -> bool:
    """Reject a single line that shouldn't become memories."""
    stripped = line.strip()
    if len(stripped) < 8:
        return True
    # JSON metadata
    if '{"session_id"' in stripped or '{"transcript_path"' in stripped:
        return True
    if '"hook_event_name"' in stripped or '"permission_mode"' in stripped:
        return True
    # File paths
    if re.search(r'[A-Z]:\\Users\\.*\\\.claude\\', stripped):
        return True
    # Garbled encoding markers (actual mojibake, not legitimate "utf-8" mentions)
    if '锛' in stripped or '鈥' in stripped:
        return True
    # Truncated JSON
    if stripped.startswith('{') and '"ok"' not in stripped and '"data"' not in stripped:
        return True
    return False


def _is_garbage(text: str) -> bool:
    """Reject entire text only if it's globally unusable."""
    if len(text) < 8:
        return True
    if len(text) > 20000:
        return True
    return False


def _filter_garbage_segments(text: str) -> str:
    """Filter out garbage lines, keeping usable content."""
    lines = text.split('\n')
    kept = [line for line in lines if not _is_garbage_line(line)]
    return '\n'.join(kept)


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

    # Per-line garbage filtering — keep usable content, drop noise
    conversation = _filter_garbage_segments(conversation)
    if not conversation.strip():
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
