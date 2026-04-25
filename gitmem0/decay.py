"""Confidence decay and memory consolidation for GitMem0.

Memories naturally lose confidence over time (exponential decay) and
duplicate/near-duplicate memories are consolidated to keep the store lean.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from gitmem0.models import MemoryType, MemoryUnit

if TYPE_CHECKING:
    from gitmem0.embeddings import EmbeddingEngine
    from gitmem0.extraction import LLMJudge
    from gitmem0.store import MemoryStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DecayEngine:
    """Manages confidence decay, archival, cleanup, and consolidation."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_engine: EmbeddingEngine,
        decay_lambda: float = 0.01,
        active_threshold: float = 0.3,
        llm_judge: Optional["LLMJudge"] = None,
    ) -> None:
        self._store = store
        self._engine = embedding_engine
        self._lambda = decay_lambda
        self._active_threshold = active_threshold
        self._min_confidence = 0.1
        self._llm_judge = llm_judge

    # ── Confidence decay ────────────────────────────────────────────────

    def compute_decayed_confidence(self, unit: MemoryUnit) -> float:
        """Return confidence after exponential decay based on time since last access.

        Formula: initial_confidence * exp(-lambda * days_since_last_access)
        """
        now = _utcnow()
        days_since_last_access = (now - unit.accessed_at).total_seconds() / 86400.0
        return unit.confidence * math.exp(-self._lambda * days_since_last_access)

    def apply_decay(self) -> dict[str, int]:
        """Decay confidence for all L1 memories; archive or clean up as needed.

        Returns {"decayed": N, "archived": N, "cleaned": N}.
        """
        l1_memories = self._store.list_memories(layer="L1")
        decayed = 0
        archived = 0
        cleaned = 0

        for unit in l1_memories:
            new_confidence = self.compute_decayed_confidence(unit)
            if new_confidence >= unit.confidence:
                # Not actually decaying (freshly accessed or lambda too small)
                continue

            decayed += 1
            unit.confidence = new_confidence
            self._store.update_memory(unit)

            if new_confidence <= self._min_confidence:
                self._store.delete_memory(unit.id)
                cleaned += 1
            elif new_confidence <= self._active_threshold:
                self._store.move_to_layer(unit.id, "L2")
                archived += 1

        return {"decayed": decayed, "archived": archived, "cleaned": cleaned}

    # ── Consolidation ───────────────────────────────────────────────────

    def find_similar_groups(self, threshold: float = 0.85) -> list[list[str]]:
        """Group L1 memories whose embeddings exceed the similarity threshold.

        Uses single-linkage clustering: memories are grouped if any pair in the
        group exceeds the threshold.
        Returns list of groups, each group being a list of memory IDs.
        """
        l1_memories = self._store.list_memories(layer="L1")
        # Filter to memories that have embeddings
        with_emb: list[MemoryUnit] = [m for m in l1_memories if m.embedding is not None]

        if len(with_emb) < 2:
            return []

        # Union-Find
        parent: dict[str, str] = {m.id: m.id for m in with_emb}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Compare all pairs (fine for typical store sizes)
        for i in range(len(with_emb)):
            for j in range(i + 1, len(with_emb)):
                sim = self._engine.similarity(
                    with_emb[i].embedding, with_emb[j].embedding  # type: ignore[arg-type]
                )
                if sim > threshold:
                    union(with_emb[i].id, with_emb[j].id)

        # Collect groups
        groups_dict: dict[str, list[str]] = {}
        for m in with_emb:
            root = find(m.id)
            groups_dict.setdefault(root, []).append(m.id)

        # Return only groups with 2+ members
        return [g for g in groups_dict.values() if len(g) >= 2]

    def consolidate_group(self, memory_ids: list[str]) -> MemoryUnit | None:
        """Merge a group of similar memories into a single consolidated unit.

        Returns the new MemoryUnit (caller is responsible for adding to store).
        Returns None if fewer than 2 IDs provided or any ID cannot be loaded.
        """
        if len(memory_ids) < 2:
            return None

        memories: list[MemoryUnit] = []
        for mid in memory_ids:
            m = self._store.get_memory(mid)
            if m is None:
                return None
            memories.append(m)

        # Combine content — use LLM summarizer if available
        combined_content = None
        if self._llm_judge is not None:
            combined_content = self._llm_judge.summarize([m.content for m in memories])
        if combined_content is None:
            combined_content = " | ".join(m.content for m in memories)

        # Use the most important memory as the base (highest importance)
        base = max(memories, key=lambda m: m.importance)

        # Combine tags (deduplicated)
        all_tags: list[str] = []
        seen: set[str] = set()
        for m in memories:
            for tag in m.tags:
                if tag not in seen:
                    seen.add(tag)
                    all_tags.append(tag)
        all_tags.append("consolidated")

        # Combine entities (deduplicated)
        all_entities: list[str] = []
        seen_entities: set[str] = set()
        for m in memories:
            for e in m.entities:
                if e not in seen_entities:
                    seen_entities.add(e)
                    all_entities.append(e)

        # Compute new embedding from combined content
        new_embedding = self._engine.embed(combined_content)

        consolidated = MemoryUnit(
            content=combined_content,
            type=base.type,
            importance=base.importance,
            confidence=base.confidence,
            source=f"consolidation:{_utcnow().isoformat()}",
            entities=all_entities,
            supersedes=",".join(m.id for m in memories),
            tags=all_tags,
            embedding=new_embedding,
            layer="L1",
        )

        # Tag originals as consolidated
        for m in memories:
            if "consolidated" not in m.tags:
                m.tags.append("consolidated")
                self._store.update_memory(m)

        return consolidated

    def run_consolidation(self, threshold: float = 0.85, dry_run: bool = False) -> dict[str, int]:
        """Find and consolidate similar memory groups.

        If not dry_run, adds consolidated memories to store and moves originals to L2.
        Returns {"groups_found": N, "consolidated": N}.
        """
        groups = self.find_similar_groups(threshold)
        groups_found = len(groups)
        consolidated = 0

        for group_ids in groups:
            new_unit = self.consolidate_group(group_ids)
            if new_unit is None:
                continue

            if not dry_run:
                self._store.add_memory(new_unit)
                # Move originals to archive
                for mid in group_ids:
                    self._store.move_to_layer(mid, "L2")

            consolidated += 1

        return {"groups_found": groups_found, "consolidated": consolidated}

    # ── Layer management ────────────────────────────────────────────────

    def promote_from_archive(self, memory_id: str) -> bool:
        """Move a memory from L2 to L1 if it exists in L2.

        Updates access timestamp. Returns True if promotion succeeded.
        """
        unit = self._store.get_memory(memory_id)
        if unit is None or unit.layer != "L2":
            return False

        self._store.move_to_layer(memory_id, "L1")
        return True

    def get_layer_stats(self) -> dict[str, dict]:
        """Return counts and average confidence per layer.

        Returns {"L1": {"count": N, "avg_confidence": F}, "L2": {...}, ...}
        """
        stats: dict[str, dict] = {}
        for layer in ("L0", "L1", "L2"):
            memories = self._store.get_memories_by_layer(layer)
            count = len(memories)
            if count > 0:
                avg_conf = sum(m.confidence for m in memories) / count
            else:
                avg_conf = 0.0
            stats[layer] = {"count": count, "avg_confidence": round(avg_conf, 4)}

        return stats
