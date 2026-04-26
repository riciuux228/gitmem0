"""LLM Judge implementation for GitMem0.

Connects to any OpenAI-compatible API (including Xiaomi Token Plan MiMo)
to provide LLM-assisted memory scoring, classification, and summarization.

Falls back gracefully to rule-based defaults on any API error.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from gitmem0.extraction import LLMJudge
from gitmem0.models import MemoryType


# ── Prompts ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a memory judgment assistant for a personal AI memory system. "
    "Answer concisely. No explanations, just the requested output."
)

_PROMPT_SHOULD_REMEMBER = (
    "Is the following text worth remembering long-term? "
    "Answer only 'yes' or 'no'.\n\nText: {content}"
)

_PROMPT_SCORE_IMPORTANCE = (
    "Rate the importance of this text for a personal memory system (0.0 to 1.0). "
    "0.0 = trivial/temporary, 0.5 = moderately useful, 1.0 = critical to remember. "
    "Answer only with a number.\n\nText: {content}"
)

_PROMPT_INFER_TYPE = (
    "Classify this text into exactly one category:\n"
    "- preference: personal likes, dislikes, habits\n"
    "- instruction: rules, procedures, how-to\n"
    "- insight: lessons learned, reflections, conclusions\n"
    "- fact: objective information, data\n"
    "- event: something that happened, timeline entry\n\n"
    "Answer only with the category name.\n\nText: {content}"
)

_PROMPT_SUMMARIZE = (
    "Summarize these related memories into one concise paragraph. "
    "Keep all key information, remove redundancy.\n\n"
    "Memories:\n{memories}"
)

_TYPE_MAP = {
    "preference": MemoryType.PREFERENCE,
    "instruction": MemoryType.INSTRUCTION,
    "insight": MemoryType.INSIGHT,
    "fact": MemoryType.FACT,
    "event": MemoryType.EVENT,
}


# ── Implementation ──────────────────────────────────────────────────────────


class MiMoLLMJudge(LLMJudge):
    """LLM Judge using Xiaomi Token Plan (OpenAI-compatible API).

    All methods return None on failure, triggering rule-based fallback.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
        model: str = "MiMo",
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("GITMEM0_LLM_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # Lazy validation — don't crash at import time
        self._enabled = bool(self._api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _chat(self, prompt: str) -> Optional[str]:
        """Send a chat completion request, return response text or None."""
        if not self._enabled:
            return None

        url = f"{self._base_url}/chat/completions"
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError):
            return None

    # ── LLMJudge protocol ───────────────────────────────────────────────

    def should_remember(self, content: str) -> Optional[bool]:
        resp = self._chat(_PROMPT_SHOULD_REMEMBER.format(content=content))
        if resp is None:
            return None
        lower = resp.lower().strip().strip(".")
        if lower.startswith("yes"):
            return True
        if lower.startswith("no"):
            return False
        return None  # unclear → use rule-based default

    def score_importance(self, content: str, context: str = "") -> Optional[float]:
        resp = self._chat(_PROMPT_SCORE_IMPORTANCE.format(content=content))
        if resp is None:
            return None
        try:
            import re
            # Try to find a float between 0 and 1
            match = re.search(r"(0\.\d+|1\.0|0\.0|1|0)", resp)
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            # Fallback: any number
            match = re.search(r"(\d+\.?\d*)", resp)
            if match:
                val = float(match.group(1))
                if val > 1.0:
                    val = val / 10.0  # e.g. "7" → 0.7
                return max(0.0, min(1.0, val))
        except (ValueError, AttributeError):
            pass
        return None

    def infer_type(self, content: str) -> Optional[MemoryType]:
        resp = self._chat(_PROMPT_INFER_TYPE.format(content=content))
        if resp is None:
            return None
        lower = resp.lower().strip().strip(".")
        # Try exact match first
        if lower in _TYPE_MAP:
            return _TYPE_MAP[lower]
        # Try partial match
        for key, mem_type in _TYPE_MAP.items():
            if key in lower:
                return mem_type
        return None

    def summarize(self, memories: list[str]) -> Optional[str]:
        if not memories:
            return None
        if len(memories) == 1:
            return memories[0]
        numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories))
        return self._chat(_PROMPT_SUMMARIZE.format(memories=numbered))
