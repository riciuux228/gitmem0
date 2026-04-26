"""Memory extraction engine for GitMem0.

Extracts MemoryUnits from raw text using multi-signal importance scoring,
confidence assessment, type inference, and deduplication.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from gitmem0.embeddings import EmbeddingEngine
from gitmem0.entities import EntityManager
from gitmem0.models import (
    Entity,
    MemoryType,
    MemoryUnit,
    TYPE_IMPORTANCE_WEIGHTS,
)
from gitmem0.store import MemoryStore


# ── LLM Judge Protocol ─────────────────────────────────────────────────────


@runtime_checkable
class LLMJudge(Protocol):
    """Protocol for LLM-assisted memory judgment.

    Implement this to plug in an LLM (local or API) that augments
    the rule-based scoring. All methods are optional — implement only
    what you need; unimplemented methods should return None to fall
    back to rule-based defaults.
    """

    def score_importance(self, content: str, context: str = "") -> Optional[float]:
        """Score importance of a memory candidate (0.0-1.0).

        Return None to use rule-based default.
        """
        ...

    def should_remember(self, content: str) -> Optional[bool]:
        """Decide if content is worth remembering.

        Return None to use rule-based default.
        """
        ...

    def infer_type(self, content: str) -> Optional[MemoryType]:
        """Infer memory type from content.

        Return None to use rule-based default.
        """
        ...

    def summarize(self, memories: list[str]) -> Optional[str]:
        """Produce a coherent summary from multiple related memories.

        Return None to use simple concatenation fallback.
        """
        ...


# ── Markers and patterns ────────────────────────────────────────────────────

_EXPLICIT_MARKERS: list[str] = [
    "记住", "remember", "注意", "note that", "important",
    "don't forget", "切记", "务必", "keep in mind", "always",
]

_CERTAINTY_HIGH: list[str] = [
    "i am", "i prefer", "i like", "i love", "i always",
    "i need", "i want", "i will", "i use", "i require",
    "definitely", "certainly", "must", "exactly",
    "我", "我喜欢", "我偏好", "我需要",
]

_CERTAINTY_LOW: list[str] = [
    "maybe", "perhaps", "i think", "i guess", "might",
    "possibly", "could be", "not sure", "not certain",
    "probably", "seems like", "好像", "可能", "也许", "大概",
]

_ACTIONABILITY_PATTERNS: list[str] = [
    r"\b(?:always|never|should|must|make sure|ensure)\b",
    r"\b(?:prefer|like|want|need|require|expect)\b",
    r"\b(?:please|do\s+not|don'?t|avoid|use|prefer)\b",
    r"\b(?:going\s+to|will|plan\s+to|next\s+time)\b",
    r"\b(?:记住|喜欢|偏好|需要|务必|不要)\b",
]

_SEGMENT_RE = re.compile(r"[^.!?。！？\n]+[.!?。！?\n]*")
_DATE_TIME_RE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b(?:yesterday|today|tomorrow|昨天|今天|明天)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
    r"|\b(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+\.?\d*")

_EXPERIENCE_KEYWORDS: list[str] = [
    "fix", "bug", "error", "issue", "problem", "solution",
    "learned", "realized", "discovered", "turns out", "root cause",
    "deploy", "refactor", "debug", "test", "failure", "crash",
    "workaround", "gotcha", "pitfall", "lesson",
    "修", "修复", "解决", "发现", "原来", "根本原因",
    "部署", "重构", "调试", "测试", "失败", "崩溃",
]


class ExtractionEngine:
    """Extracts memory units from raw text."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        entity_manager: EntityManager,
        llm_judge: Optional[LLMJudge] = None,
    ) -> None:
        self._store = store
        self._embedding_engine = embedding_engine
        self._entity_manager = entity_manager
        self._llm_judge = llm_judge

    # ── Importance scoring ──────────────────────────────────────────────

    def score_importance(
        self,
        content: str,
        memory_type: MemoryType,
        entities: list[Entity],
    ) -> float:
        """Multi-signal importance scoring.

        Returns a weighted average of six signals on 0.0-1.0 scale.
        """
        explicit = self._score_explicit(content)
        type_weight = TYPE_IMPORTANCE_WEIGHTS.get(memory_type, 0.5)
        novelty = self._score_novelty(content, memory_type)
        specificity = self._score_specificity(content, entities)
        actionability = self._score_actionability(content)
        experience = self._score_experience(content)

        return (
            explicit * 0.25
            + type_weight * 0.15
            + novelty * 0.15
            + specificity * 0.15
            + actionability * 0.15
            + experience * 0.15
        )

    def _score_explicit(self, content: str) -> float:
        lower = content.lower()
        for marker in _EXPLICIT_MARKERS:
            if marker in lower:
                return 1.0
        return 0.0

    def _score_novelty(self, content: str, memory_type: Optional[MemoryType] = None) -> float:
        """Novelty scoring with type-aware logic.

        For EVENT/EXPERIENCE: repeated mention = more important (inverted).
        For other types: novelty = 1.0 - max_similarity.
        """
        existing = self._store.list_memories(limit=200)
        if not existing:
            return 1.0

        query_emb = self._embedding_engine.embed(content)
        candidates: list[tuple[str, list[float]]] = [
            (m.id, m.embedding) for m in existing if m.embedding is not None
        ]
        if not candidates:
            return 1.0

        results = self._embedding_engine.most_similar(
            query_emb, candidates, top_k=1
        )
        if not results:
            return 1.0

        max_sim = results[0][1]

        # EVENT/EXPERIENCE: repeated mention signals importance, not redundancy
        if memory_type in (MemoryType.EVENT, MemoryType.EXPERIENCE):
            return 0.5 + 0.5 * max_sim

        return max(0.0, 1.0 - max_sim)

    def _score_specificity(
        self, content: str, entities: list[Entity]
    ) -> float:
        score = 0.0

        # Length factor (longer = more specific, capped)
        length_factor = min(len(content) / 200.0, 1.0)
        score += length_factor * 0.3

        # Entity factor
        entity_factor = min(len(entities) / 5.0, 1.0)
        score += entity_factor * 0.3

        # Numbers / dates
        has_numbers = bool(_NUMBER_RE.search(content))
        has_dates = bool(_DATE_TIME_RE.search(content))
        score += (0.2 if has_numbers else 0.0) + (0.2 if has_dates else 0.0)

        return min(score, 1.0)

    def _score_content_specificity(self, content: str) -> float:
        """Content-only specificity for confidence assessment (no entity dependency)."""
        score = 0.0

        # Length factor — use 100 chars as the midpoint instead of 200
        length_factor = min(len(content) / 100.0, 1.0)
        score += length_factor * 0.3

        # Numbers / dates add concrete detail
        has_numbers = bool(_NUMBER_RE.search(content))
        has_dates = bool(_DATE_TIME_RE.search(content))
        score += (0.2 if has_numbers else 0.0) + (0.2 if has_dates else 0.0)

        # Declarative clarity: short but clear "I prefer/like/am" statements
        # are inherently specific in meaning even if short in length
        if re.search(r"\b(?:I\s+(?:prefer|like|love|am|want|need|will|use))\b",
                      content, re.IGNORECASE):
            score += 0.3

        return min(score, 1.0)

    def _score_actionability(self, content: str) -> float:
        lower = content.lower()
        for pattern in _ACTIONABILITY_PATTERNS:
            if re.search(pattern, lower):
                return 1.0
        return 0.0

    def _score_experience(self, content: str) -> float:
        """Score whether content contains experience/lesson-learned signals."""
        lower = content.lower()
        for kw in _EXPERIENCE_KEYWORDS:
            if kw in lower:
                return 1.0
        return 0.0

    # ── Confidence assessment ───────────────────────────────────────────

    def assess_confidence(
        self, content: str, source: str = "conversation"
    ) -> float:
        """Assess how confident we should be in this memory.

        Returns geometric mean of source, certainty, and specificity factors.
        """
        source_factor = 1.0 if source == "user_explicit" else 0.7
        certainty_factor = self._score_certainty(content)
        specificity_factor = self._score_content_specificity(content)

        # Geometric mean
        product = source_factor * certainty_factor * specificity_factor
        if product <= 0.0:
            return 0.0
        return product ** (1.0 / 3.0)

    def _score_certainty(self, content: str) -> float:
        lower = content.lower()
        for phrase in _CERTAINTY_LOW:
            if phrase in lower:
                return 0.3
        for phrase in _CERTAINTY_HIGH:
            if phrase in lower:
                return 1.0
        return 0.6

    # ── Type inference ──────────────────────────────────────────────────

    def infer_type(self, content: str) -> MemoryType:
        """Infer memory type from content using pattern matching."""
        lower = content.lower()

        if re.search(r"\b(?:prefer|like|喜欢|偏好)\b", lower):
            return MemoryType.PREFERENCE

        if re.search(r"\b(?:remember|note|记住|注意)\b", lower):
            return MemoryType.INSTRUCTION

        if _DATE_TIME_RE.search(content):
            return MemoryType.EVENT

        # Experience: debugging, fixing, lessons learned
        if any(kw in lower for kw in _EXPERIENCE_KEYWORDS):
            return MemoryType.EXPERIENCE

        if re.search(r"\b(?:because|therefore|所以|thus|hence|consequently)\b", lower):
            return MemoryType.INSIGHT

        return MemoryType.FACT

    # ── Deduplication ───────────────────────────────────────────────────

    def is_duplicate(self, unit: MemoryUnit, threshold: float = 0.85) -> bool:
        """Check if a very similar memory already exists."""
        existing = self._store.list_memories(limit=200)
        if not existing:
            return False

        query_emb: Optional[list[float]] = None
        if unit.embedding is not None:
            query_emb = unit.embedding
        else:
            query_emb = self._embedding_engine.embed(unit.content)

        candidates: list[tuple[str, list[float]]] = [
            (m.id, m.embedding) for m in existing if m.embedding is not None
        ]
        if not candidates:
            return False

        results = self._embedding_engine.most_similar(
            query_emb, candidates, top_k=1
        )
        if not results:
            return False

        return results[0][1] >= threshold

    # ── Main extraction pipeline ───────────────────────────────────────

    def extract_from_text(
        self, text: str, source: str = "conversation"
    ) -> list[MemoryUnit]:
        """Extract memory units from raw text.

        Returns a list of MemoryUnits that have NOT been stored yet.
        The caller is responsible for persisting them.
        """
        segments = _split_segments(text)
        memories: list[MemoryUnit] = []

        for segment in segments:
            segment = segment.strip()
            if len(segment) < 10:
                continue

            # LLM judge: should we remember this at all?
            if self._llm_judge is not None:
                should = self._llm_judge.should_remember(segment)
                if should is False:
                    continue

            # Type inference (LLM override if available)
            mem_type = self.infer_type(segment)
            if self._llm_judge is not None:
                llm_type = self._llm_judge.infer_type(segment)
                if llm_type is not None:
                    mem_type = llm_type

            entities = self._entity_manager.extract_entities(segment)

            # Importance scoring (LLM override if available)
            importance = self.score_importance(segment, mem_type, entities)
            if self._llm_judge is not None:
                llm_imp = self._llm_judge.score_importance(segment, text[:200])
                if llm_imp is not None:
                    # Blend: 60% LLM + 40% rules (LLM is advisor, not dictator)
                    importance = llm_imp * 0.6 + importance * 0.4

            confidence = self.assess_confidence(segment, source)

            if importance < 0.15 or confidence < 0.15:
                continue

            embedding = self._embedding_engine.embed(segment)

            unit = MemoryUnit(
                content=segment,
                type=mem_type,
                importance=importance,
                confidence=confidence,
                source=source,
                entities=[e.id for e in entities],
                embedding=embedding,
            )

            # Link entities and store relations
            unit = self._entity_manager.link_memory_entities(unit)

            # Deduplicate
            if self.is_duplicate(unit):
                continue

            memories.append(unit)

        return memories


# ── Helpers ──────────────────────────────────────────────────────────────────


def _split_segments(text: str) -> list[str]:
    """Split text into candidate segments by sentence or logical chunk.

    Merges short fragments back with their neighbors to avoid breaking
    code identifiers like Date.now() or paths like /usr/local/bin.
    """
    segments = _SEGMENT_RE.findall(text)
    # Also split on newlines that weren't already caught
    expanded: list[str] = []
    for seg in segments:
        parts = [p.strip() for p in re.split(r"\n+", seg) if p.strip()]
        expanded.extend(parts)

    # Merge fragments back if they look like broken code identifiers.
    # A segment that starts with lowercase or ( is likely a continuation
    # of a code reference like Date.now() or console.log()
    merged: list[str] = []
    for seg in expanded:
        seg = seg.strip()
        if not seg:
            continue
        if merged and (seg[0].islower() or seg[0] == '('):
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)
    return merged
