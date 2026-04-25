"""Context injection optimizer for GitMem0.

Builds LLM-ready context packages from retrieved memories with
Lost-in-the-Middle arrangement and token budget compression.
"""

from __future__ import annotations

from gitmem0.entities import EntityManager
from gitmem0.models import MemoryUnit
from gitmem0.retrieval import RetrievalEngine


class ContextBuilder:
    """Builds optimized context strings for LLM consumption."""

    def __init__(
        self, retrieval_engine: RetrievalEngine, entity_manager: EntityManager
    ) -> None:
        self._retrieval = retrieval_engine
        self._entities = entity_manager

    # ── Lost in the Middle arrangement ────────────────────────────────

    def arrange_memories(
        self, memories: list[MemoryUnit]
    ) -> list[MemoryUnit]:
        """Rearrange memories for LLM consumption using Lost-in-the-Middle.

        Given memories sorted by relevance (highest first):
        - Position 0: highest relevance (primacy effect)
        - Positions 1..n-2: remaining in descending order
        - Position n-1: second highest relevance (recency effect)

        For 1-2 memories: return as-is.
        For 3+ memories: swap second and last.
        """
        if len(memories) <= 2:
            return list(memories)
        result = list(memories)
        result[1], result[-1] = result[-1], result[1]
        return result

    # ── Token budget management ───────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate. Uses len/4 for English, len/2 for CJK-heavy."""
        if not text:
            return 0
        cjk_count = sum(
            1 for ch in text
            if "一" <= ch <= "鿿"
            or "㐀" <= ch <= "䶿"
            or "\U00020000" <= ch <= "\U0002a6df"
        )
        ratio = cjk_count / len(text)
        if ratio > 0.3:
            return len(text) // 2
        return len(text) // 4

    def compress_memories(
        self, memories: list[MemoryUnit], token_budget: int
    ) -> list[MemoryUnit]:
        """Compress memories to fit within token_budget.

        Pass 1: Remove memories with confidence < 0.3.
        Pass 2: Group by entity, keep only the most recent per entity.
        Pass 3: Truncate to top-N by importance until budget is met.
        """
        if not memories:
            return []

        total = sum(self.estimate_tokens(m.content) for m in memories)
        if total <= token_budget:
            return list(memories)

        # Pass 1: drop low-confidence
        filtered = [m for m in memories if m.confidence >= 0.3]
        total = sum(self.estimate_tokens(m.content) for m in filtered)
        if total <= token_budget:
            return filtered

        # Pass 2: group by entity, keep most recent per entity
        by_entity: dict[str, list[MemoryUnit]] = {}
        no_entity: list[MemoryUnit] = []
        for m in filtered:
            if m.entities:
                for eid in m.entities:
                    by_entity.setdefault(eid, []).append(m)
            else:
                no_entity.append(m)

        deduped_ids: set[str] = set()
        deduped: list[MemoryUnit] = []
        for eid, group in by_entity.items():
            best = max(group, key=lambda m: m.created_at)
            if best.id not in deduped_ids:
                deduped_ids.add(best.id)
                deduped.append(best)
        for m in no_entity:
            if m.id not in deduped_ids:
                deduped_ids.add(m.id)
                deduped.append(m)

        total = sum(self.estimate_tokens(m.content) for m in deduped)
        if total <= token_budget:
            return deduped

        # Pass 3: truncate by importance
        ranked = sorted(deduped, key=lambda m: m.importance, reverse=True)
        result: list[MemoryUnit] = []
        running = 0
        for m in ranked:
            cost = self.estimate_tokens(m.content)
            if running + cost > token_budget:
                break
            result.append(m)
            running += cost
        return result

    # ── Context package generation ────────────────────────────────────

    def build_context(
        self,
        query: str,
        token_budget: int = 2000,
        max_memories: int = 8,
    ) -> str:
        """Build a structured context string for an LLM prompt.

        1. Search for relevant memories
        2. Compress to fit token budget
        3. Arrange with Lost-in-the-Middle
        4. Format as structured text
        """
        memories = self._retrieval.search(query, top_n=max_memories)
        if not memories:
            return ""

        memories = self.compress_memories(memories, token_budget)
        memories = self.arrange_memories(memories)

        lines: list[str] = ["[Relevant Memories]"]
        for m in memories:
            lines.append(f"- {m.content}")

        # Collect entity names from the selected memories
        entity_names: list[str] = []
        seen: set[str] = set()
        for m in memories:
            for eid in m.entities:
                if eid not in seen:
                    seen.add(eid)
                    entity = self._entities._store.get_entity(eid)
                    if entity:
                        entity_names.append(entity.name)

        if entity_names:
            lines.append("[Related Entities]")
            for name in entity_names:
                ctx = self._entities.get_entity_context(name)
                lines.append(f"- {ctx}")

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────

    def context_stats(self, query: str) -> dict:
        """Return stats about the context that would be built for a query."""
        memories = self._retrieval.search(query, top_n=8)
        if not memories:
            return {
                "memories_found": 0,
                "total_tokens": 0,
                "compressed": False,
                "entities_included": 0,
            }

        raw_tokens = sum(self.estimate_tokens(m.content) for m in memories)
        compressed = self.compress_memories(memories, 2000)
        compressed_tokens = sum(
            self.estimate_tokens(m.content) for m in compressed
        )

        entity_ids: set[str] = set()
        for m in compressed:
            entity_ids.update(m.entities)

        return {
            "memories_found": len(memories),
            "total_tokens": compressed_tokens,
            "compressed": len(compressed) < len(memories),
            "entities_included": len(entity_ids),
        }
