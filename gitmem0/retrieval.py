"""Two-stage retrieval engine for GitMem0.

Stage 1 (recall): fast candidate gathering via BM25, semantic search,
entity lookup, and recency, fused with Reciprocal Rank Fusion.
Stage 2 (rerank): multi-signal scoring over the candidates.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from gitmem0.embeddings import EmbeddingEngine
from gitmem0.models import (
    DEFAULT_SCORING_WEIGHTS,
    MemoryUnit,
)
from gitmem0.store import MemoryStore

# RRF constant (standard k=60)
_RRF_K = 60


def compute_confidence(unit: MemoryUnit, decay_lambda: float = 0.01) -> float:
    """Apply time-based decay to a memory unit's confidence.

    confidence * exp(-lambda * days_since_access)
    """
    now = datetime.now(timezone.utc)
    days = (now - unit.accessed_at).total_seconds() / 86400.0
    return unit.confidence * math.exp(-decay_lambda * days)


def _extract_key_terms(query: str) -> list[str]:
    """Pull plausible entity / keyword tokens from a query string."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}", query)
    # Drop very common stop words (kept minimal to stay fast)
    stops = {
        "the", "a", "an", "is", "it", "in", "on", "of", "for", "and",
        "to", "was", "are", "be", "this", "that", "with", "from", "by",
        "do", "does", "did", "has", "have", "had", "can", "could", "would",
        "should", "will", "what", "when", "where", "who", "how", "which",
        "my", "me", "i", "we", "our", "your", "their",
    }
    return [t for t in tokens if t.lower() not in stops]


class RetrievalEngine:
    """Two-stage retrieval: fast recall then multi-signal rerank."""

    def __init__(self, store: MemoryStore, embedding_engine: EmbeddingEngine) -> None:
        self._store = store
        self._emb = embedding_engine

    # ------------------------------------------------------------------
    # Stage 1 — Recall
    # ------------------------------------------------------------------

    def recall(self, query: str, limit: int = 20) -> list[str]:
        """Return candidate memory IDs ranked by RRF fusion."""
        sources: list[list[str]] = []

        # (a) BM25 via FTS5
        fts_results = self._store.search_fts(query, limit)
        if fts_results:
            sources.append([mid for mid, _ in fts_results])

        # (b) Semantic search (only if embeddings are available)
        if self._emb.is_available():
            sem_ids = self._semantic_search(query, limit)
            if sem_ids:
                sources.append(sem_ids)

        # (c) Entity-based
        entity_ids = self._entity_search(query)
        if entity_ids:
            sources.append(entity_ids)

        # (d) Recency — last 30 days
        recency_ids = self._recency_search(days=30, limit=limit)
        if recency_ids:
            sources.append(recency_ids)

        if not sources:
            return []

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for ranked_list in sources:
            for rank, mid in enumerate(ranked_list):
                rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (_RRF_K + rank)

        fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return fused[:limit]

    def _semantic_search(self, query: str, limit: int) -> list[str]:
        """Embed query, compare against stored embeddings, return ranked IDs."""
        query_emb = self._emb.embed(query)
        # Build candidate list from memories that have embeddings
        # We scan all memories — in a production system this would be ANN-indexed.
        all_memories = self._store.list_memories(limit=5000)
        candidates: list[tuple[str, list[float]]] = [
            (m.id, m.embedding) for m in all_memories if m.embedding is not None
        ]
        if not candidates:
            return []
        results = self._emb.most_similar(query_emb, candidates, top_k=limit)
        return [mid for mid, _ in results]

    def _entity_search(self, query: str) -> list[str]:
        """Extract key terms, match entities, return related memory IDs."""
        terms = _extract_key_terms(query)
        if not terms:
            return []

        seen_entity_ids: set[str] = set()
        memory_ids: list[str] = []

        for term in terms:
            entity = self._store.get_entity_by_name(term)
            if entity is None:
                continue
            if entity.id in seen_entity_ids:
                continue
            seen_entity_ids.add(entity.id)

            # Collect memories that reference this entity
            all_memories = self._store.list_memories(limit=5000)
            for m in all_memories:
                if entity.id in m.entities:
                    memory_ids.append(m.id)

            # Walk relations to get connected entity memories
            relations = self._store.get_relations(entity.id)
            for rel in relations:
                connected_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == entity.id
                    else rel.source_entity_id
                )
                for m in all_memories:
                    if connected_id in m.entities and m.id not in memory_ids:
                        memory_ids.append(m.id)

        return memory_ids

    def _recency_search(self, days: int = 30, limit: int = 20) -> list[str]:
        """Return IDs of memories accessed within the last N days."""
        now = datetime.now(timezone.utc)
        all_memories = self._store.list_memories(limit=5000)
        recent: list[tuple[float, str]] = []
        for m in all_memories:
            age_days = (now - m.accessed_at).total_seconds() / 86400.0
            if age_days <= days:
                recent.append((age_days, m.id))
        recent.sort(key=lambda x: x[0])
        return [mid for _, mid in recent[:limit]]

    # ------------------------------------------------------------------
    # Stage 2 — Rerank
    # ------------------------------------------------------------------

    def rerank(
        self, query: str, candidate_ids: list[str], top_n: int = 5
    ) -> list[MemoryUnit]:
        """Score candidates with multi-signal ranking, return top_n units."""
        if not candidate_ids:
            return []

        weights = DEFAULT_SCORING_WEIGHTS

        # Precompute query embedding for semantic similarity
        query_emb: Optional[list[float]] = None
        if self._emb.is_available():
            query_emb = self._emb.embed(query)

        # Precompute BM25 scores for normalization
        fts_results = self._store.search_fts(query, limit=1000)
        bm25_map: dict[str, float] = {mid: score for mid, score in fts_results}

        # Extract query entity terms for entity relevance
        query_terms = {t.lower() for t in _extract_key_terms(query)}

        # Load candidate units
        units: list[MemoryUnit] = []
        for cid in candidate_ids:
            u = self._store.get_memory(cid)
            if u is not None:
                units.append(u)

        if not units:
            return []

        # Normalize BM25 scores to [0, 1] across candidates
        raw_bm25 = [bm25_map.get(u.id, 0.0) for u in units]
        bm25_max = max(raw_bm25) if raw_bm25 and max(raw_bm25) > 0 else 1.0
        norm_bm25 = {units[i].id: raw_bm25[i] / bm25_max for i in range(len(units))}

        scored: list[tuple[float, MemoryUnit]] = []
        already_selected: list[list[float]] = []

        for unit in units:
            # semantic_sim
            sem_sim = 0.0
            if query_emb is not None and unit.embedding is not None:
                sem_sim = self._emb.similarity(query_emb, unit.embedding)

            # bm25_score (normalised)
            bm25 = norm_bm25.get(unit.id, 0.0)

            # entity_relevance — fraction of memory entities appearing in query terms
            ent_rel = 0.0
            if unit.entities:
                # Resolve entity names for the unit's entity IDs
                entity_names: set[str] = set()
                for eid in unit.entities:
                    ent = self._store.get_entity(eid)
                    if ent:
                        entity_names.add(ent.name.lower())
                if entity_names:
                    shared = len(entity_names & query_terms)
                    ent_rel = shared / len(entity_names)

            # recency_score
            now = datetime.now(timezone.utc)
            days_since = (now - unit.accessed_at).total_seconds() / 86400.0
            recency = 1.0 / (1.0 + days_since)

            # importance
            importance = unit.importance

            # confidence (with decay)
            confidence = compute_confidence(unit)

            # redundancy_penalty — max similarity to any already-selected memory
            red_penalty = 0.0
            if already_selected and unit.embedding is not None:
                emb = np.asarray(unit.embedding, dtype=np.float32)
                for sel_emb in already_selected:
                    sim = self._emb.similarity(unit.embedding, sel_emb)
                    if sim > red_penalty:
                        red_penalty = sim

            final_score = (
                weights["semantic"] * sem_sim
                + weights["bm25"] * bm25
                + weights["entity"] * ent_rel
                + weights["recency"] * recency
                + weights["importance"] * importance
                + weights["confidence"] * confidence
                - weights.get("redundancy", 0.15) * red_penalty
            )

            scored.append((final_score, unit))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [unit for _, unit in scored[:top_n]]

        # Touch access stats on returned memories
        for unit in results:
            unit.touch()
            self._store.update_memory(unit)

        return results

    # ------------------------------------------------------------------
    # Combined search
    # ------------------------------------------------------------------

    def search(self, query: str, top_n: int = 5) -> list[MemoryUnit]:
        """Main entry point: recall then rerank."""
        candidates = self.recall(query, limit=20)
        return self.rerank(query, candidates, top_n=top_n)
