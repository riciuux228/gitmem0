"""End-to-end integration tests for GitMem0.

Tests the full pipeline: extraction → storage → retrieval → consolidation.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gitmem0.models import (
    Entity,
    EntityType,
    MemoryType,
    MemoryUnit,
    Relation,
)
from gitmem0.store import MemoryStore
from gitmem0.entities import EntityManager
from gitmem0.extraction import ExtractionEngine
from gitmem0.embeddings import EmbeddingEngine
from gitmem0.retrieval import RetrievalEngine
from gitmem0.decay import DecayEngine
from gitmem0.context import ContextBuilder


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "integration.db")


@pytest.fixture
def store(tmp_db):
    return MemoryStore(tmp_db)


@pytest.fixture
def embedding_engine():
    return EmbeddingEngine()


@pytest.fixture
def entity_manager(store):
    return EntityManager(store)


@pytest.fixture
def extraction_engine(store, embedding_engine, entity_manager):
    return ExtractionEngine(store, embedding_engine, entity_manager)


@pytest.fixture
def retrieval_engine(store, embedding_engine):
    return RetrievalEngine(store, embedding_engine)


@pytest.fixture
def decay_engine(store, embedding_engine):
    return DecayEngine(store, embedding_engine)


# ── Full Pipeline Tests ─────────────────────────────────────────────


class TestFullPipeline:
    """Test extract → store → retrieve → context pipeline."""

    def test_extract_store_retrieve(self, extraction_engine, store, retrieval_engine):
        """Extract memories from text, store them, then retrieve."""
        text = "I prefer Python for scripting. Docker is great for deployment."
        memories = extraction_engine.extract_from_text(text)

        # Store extracted memories
        for m in memories:
            store.add_memory(m)

        # Retrieve by query
        results = retrieval_engine.search("Python", top_n=3)
        assert len(results) >= 1
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_chinese_extraction(self, extraction_engine, store):
        """Extract memories from Chinese text."""
        text = "我喜欢用Python写后端。React适合做前端界面。"
        memories = extraction_engine.extract_from_text(text)
        assert len(memories) >= 1

        # Store and verify
        for m in memories:
            store.add_memory(m)
        stats = store.stats()
        assert stats["total_memories"] >= 1

    def test_entity_extraction_chinese(self, entity_manager):
        """Extract entities from Chinese text with English tech terms."""
        entities = entity_manager.extract_entities("我用Python写了三年后端")
        names = [e.name for e in entities]
        assert "Python" in names

    def test_relation_extraction_chinese(self, entity_manager):
        """Extract relations from Chinese text."""
        entities = entity_manager.extract_entities("我喜欢Python和Rust")
        relations = entity_manager.extract_relations("我喜欢Python和Rust", entities)

        # Should have at least co_occurs relation
        assert len(relations) >= 1
        types = [r.type for r in relations]
        assert "co_occurs" in types or "prefers" in types

    def test_entity_linking(self, extraction_engine, store):
        """Entities are linked to memories after extraction."""
        text = "Python is great for data science."
        memories = extraction_engine.extract_from_text(text)
        for m in memories:
            store.add_memory(m)

        # Check that memories have entity links
        stored = store.list_memories()
        has_entities = any(m.entities for m in stored)
        assert has_entities


# ── Search Precision Tests ──────────────────────────────────────────


class TestSearchPrecision:
    """Test that search returns relevant results."""

    def test_content_search_cjk(self, store):
        """LIKE-based search works for CJK text."""
        store.add_memory(MemoryUnit(content="用户偏好用Python写脚本"))
        store.add_memory(MemoryUnit(content="React适合做前端"))
        store.add_memory(MemoryUnit(content="Python和Django搭配很好"))

        results = store.search_content("Python")
        assert len(results) >= 2

    def test_content_search_multi_token(self, store):
        """Content search with multiple tokens uses AND logic."""
        store.add_memory(MemoryUnit(content="Python Django web development"))
        store.add_memory(MemoryUnit(content="Python scripting"))
        store.add_memory(MemoryUnit(content="Django REST framework"))

        results = store.search_content("Python Django")
        ids_set = set(results)
        # Only the first memory has both tokens
        assert len(ids_set) == 1

    def test_retrieval_rerank(self, store, embedding_engine, retrieval_engine):
        """Reranking puts more relevant results first."""
        store.add_memory(MemoryUnit(
            content="I prefer Python for data science",
            importance=0.9,
            confidence=0.9,
        ))
        store.add_memory(MemoryUnit(
            content="Docker is great for deployment",
            importance=0.7,
            confidence=0.7,
        ))
        store.add_memory(MemoryUnit(
            content="Python scripting is fun",
            importance=0.5,
            confidence=0.5,
        ))

        results = retrieval_engine.search("Python", top_n=2)
        assert len(results) >= 1
        # Top result should be Python-related
        assert "Python" in results[0].content


# ── Decay and Consolidation Tests ───────────────────────────────────


class TestDecayConsolidation:
    """Test decay and consolidation pipeline."""

    def test_decay_moves_stale_to_l2(self, decay_engine, store):
        """Memories with low decayed confidence move to L2."""
        # confidence=0.8, days=100: 0.8 * exp(-0.01*100) = 0.8 * 0.368 = 0.294
        # Below active_threshold (0.3) → archived to L2
        m = MemoryUnit(content="stale memory", confidence=0.8)
        m.accessed_at = datetime.now(timezone.utc) - timedelta(days=100)
        store.add_memory(m)

        result = decay_engine.apply_decay()
        assert result["decayed"] >= 1
        assert result["archived"] >= 1

        # Check it's in L2
        stored = store.get_memory(m.id)
        assert stored.layer == "L2"

    def test_consolidation_finds_duplicates(self, decay_engine, store, embedding_engine):
        """Consolidation groups similar memories."""
        # Add two very similar memories
        store.add_memory(MemoryUnit(
            content="I prefer Python for scripting",
            embedding=embedding_engine.embed("I prefer Python for scripting"),
        ))
        store.add_memory(MemoryUnit(
            content="I like Python for scripting",
            embedding=embedding_engine.embed("I like Python for scripting"),
        ))

        groups = decay_engine.find_similar_groups(threshold=0.85)
        # Should find at least one group
        assert len(groups) >= 1

    def test_consolidation_merge(self, decay_engine, store, embedding_engine):
        """Consolidation merges similar memories into one."""
        m1 = MemoryUnit(
            content="I prefer Python for scripting",
            embedding=embedding_engine.embed("I prefer Python for scripting"),
        )
        m2 = MemoryUnit(
            content="I like Python for scripting",
            embedding=embedding_engine.embed("I like Python for scripting"),
        )
        store.add_memory(m1)
        store.add_memory(m2)

        result = decay_engine.run_consolidation(threshold=0.85)
        assert result["consolidated"] >= 1

        # Originals should be in L2
        assert store.get_memory(m1.id).layer == "L2"
        assert store.get_memory(m2.id).layer == "L2"

        # Should have a new consolidated memory in L1
        l1 = store.list_memories(layer="L1")
        consolidated = [m for m in l1 if "consolidated" in m.tags]
        assert len(consolidated) >= 1


# ── Context Builder Tests ───────────────────────────────────────────


class TestContextBuilder:
    """Test context building for LLM consumption."""

    def test_context_with_memories(self, store, embedding_engine, retrieval_engine):
        """Context includes relevant memories."""
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval_engine, em)

        store.add_memory(MemoryUnit(content="I prefer Python", importance=0.9))
        store.add_memory(MemoryUnit(content="Docker is great", importance=0.7))

        context = ctx.build_context("user preferences Python")
        assert "Python" in context
        assert "Relevant Memories" in context

    def test_context_empty(self, store, embedding_engine, retrieval_engine):
        """Context handles empty store gracefully."""
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval_engine, em)

        context = ctx.build_context("test query")
        # Empty store returns empty string
        assert context == ""


# ── Garbage Filtering Tests ─────────────────────────────────────────


class TestGarbageFiltering:
    """Test hook garbage filtering."""

    def test_is_garbage_short(self):
        from hooks.post_response import _is_garbage
        assert _is_garbage("short") is True

    def test_is_garbage_long(self):
        from hooks.post_response import _is_garbage
        assert _is_garbage("x" * 6000) is True

    def test_is_garbage_json_metadata(self):
        from hooks.post_response import _is_garbage
        assert _is_garbage('{"session_id":"abc123","transcript_path":"test"}') is True

    def test_is_garbage_file_path(self):
        from hooks.post_response import _is_garbage
        assert _is_garbage('C:\\Users\\test\\.claude\\settings.json has hooks') is True

    def test_is_garbage_valid(self):
        from hooks.post_response import _is_garbage
        text = "I prefer using dark mode for all my development work."
        assert _is_garbage(text) is False

    def test_is_garbage_garbled(self):
        from hooks.post_response import _is_garbage
        assert _is_garbage("some text with 锛 garbled encoding") is True


# ── Entity Graph Tests ──────────────────────────────────────────────


class TestEntityGraph:
    """Test entity graph operations."""

    def test_co_occurrence_relations(self, entity_manager):
        """Entities in same text get co_occurs relation."""
        entities = entity_manager.extract_entities("Python and Docker are great")
        relations = entity_manager.extract_relations(
            "Python and Docker are great", entities
        )
        types = [r.type for r in relations]
        assert "co_occurs" in types

    def test_graph_traversal(self, entity_manager, store):
        """BFS traversal finds connected entities."""
        e1 = Entity(name="User", type=EntityType.PERSON)
        e2 = Entity(name="Python", type=EntityType.TECHNOLOGY)
        e3 = Entity(name="Django", type=EntityType.TECHNOLOGY)
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)
        store.add_relation(Relation(source_entity_id=e1.id, target_entity_id=e2.id, type="uses"))
        store.add_relation(Relation(source_entity_id=e2.id, target_entity_id=e3.id, type="powers"))

        neighbors = entity_manager.get_entity_neighbors(e1.id, depth=2)
        assert len(neighbors) >= 2

    def test_find_path(self, entity_manager, store):
        """Shortest path between entities."""
        e1 = Entity(name="A", type=EntityType.PERSON)
        e2 = Entity(name="B", type=EntityType.TECHNOLOGY)
        e3 = Entity(name="C", type=EntityType.TECHNOLOGY)
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)
        store.add_relation(Relation(source_entity_id=e1.id, target_entity_id=e2.id, type="uses"))
        store.add_relation(Relation(source_entity_id=e2.id, target_entity_id=e3.id, type="related"))

        path = entity_manager.find_path(e1.id, e3.id)
        assert path is not None
        assert len(path) == 3
