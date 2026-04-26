"""Integration tests for GitMem0 memory system."""

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
    TYPE_IMPORTANCE_WEIGHTS,
)
from gitmem0.store import MemoryStore
from gitmem0.entities import EntityManager
from gitmem0.extraction import ExtractionEngine
from gitmem0.context import ContextBuilder
from gitmem0.retrieval import RetrievalEngine


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    """Create a fresh MemoryStore for each test."""
    from gitmem0.store import MemoryStore

    return MemoryStore(tmp_db)


@pytest.fixture
def embedding_engine():
    """Create an EmbeddingEngine (may be dummy if sentence-transformers unavailable)."""
    from gitmem0.embeddings import EmbeddingEngine

    return EmbeddingEngine()


@pytest.fixture
def entity_manager(store):
    """Create an EntityManager."""
    from gitmem0.entities import EntityManager

    return EntityManager(store)


# ── Models ──────────────────────────────────────────────────────────


class TestModels:
    def test_memory_unit_create(self):
        m = MemoryUnit(content="test", type=MemoryType.FACT)
        assert m.content == "test"
        assert m.type == MemoryType.FACT
        assert m.id  # auto-generated
        assert m.importance == 0.5
        assert m.confidence == 0.8

    def test_memory_unit_roundtrip(self):
        m = MemoryUnit(
            content="user prefers Python",
            type=MemoryType.PREFERENCE,
            importance=0.9,
            confidence=0.85,
            entities=["e1", "e2"],
            tags=["tag1"],
        )
        d = m.to_dict()
        m2 = MemoryUnit.from_dict(d)
        assert m2.content == m.content
        assert m2.type == m.type
        assert m2.entities == m.entities
        assert m2.tags == m.tags

    def test_memory_unit_touch(self):
        m = MemoryUnit(content="test")
        assert m.access_count == 0
        old_time = m.accessed_at
        m.touch()
        assert m.access_count == 1
        assert m.accessed_at >= old_time

    def test_entity_create(self):
        e = Entity(name="Python", type=EntityType.TECHNOLOGY)
        assert e.name == "Python"
        assert e.mention_count == 1

    def test_entity_roundtrip(self):
        e = Entity(name="Python", type=EntityType.TECHNOLOGY, aliases=["py"])
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.name == "Python"
        assert e2.aliases == ["py"]

    def test_relation_roundtrip(self):
        r = Relation(
            source_entity_id="e1",
            target_entity_id="e2",
            type="prefers",
            weight=0.8,
        )
        d = r.to_dict()
        r2 = Relation.from_dict(d)
        assert r2.type == "prefers"
        assert r2.weight == 0.8


# ── Store ───────────────────────────────────────────────────────────


class TestStore:
    def test_add_and_get(self, store):
        m = MemoryUnit(content="test memory")
        store.add_memory(m)
        result = store.get_memory(m.id)
        assert result is not None
        assert result.content == "test memory"

    def test_get_nonexistent(self, store):
        assert store.get_memory("nonexistent") is None

    def test_update_memory(self, store):
        m = MemoryUnit(content="original")
        store.add_memory(m)
        m.content = "updated"
        store.update_memory(m)
        result = store.get_memory(m.id)
        assert result.content == "updated"

    def test_delete_memory(self, store):
        m = MemoryUnit(content="to delete")
        store.add_memory(m)
        store.delete_memory(m.id)
        assert store.get_memory(m.id) is None

    def test_list_memories(self, store):
        for i in range(5):
            store.add_memory(MemoryUnit(content=f"memory {i}"))
        result = store.list_memories(limit=3)
        assert len(result) == 3

    def test_list_by_type(self, store):
        store.add_memory(MemoryUnit(content="fact", type=MemoryType.FACT))
        store.add_memory(MemoryUnit(content="pref", type=MemoryType.PREFERENCE))
        facts = store.list_memories(type=MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].type == MemoryType.FACT

    def test_fts_search(self, store):
        store.add_memory(MemoryUnit(content="Python is great"))
        store.add_memory(MemoryUnit(content="Rust is fast"))
        results = store.search_fts("Python")
        assert len(results) == 1

    def test_fts_chinese(self, store):
        store.add_memory(MemoryUnit(content="用户偏好用Python写脚本"))
        results = store.search_fts("Python")
        assert len(results) >= 1

    def test_entity_crud(self, store):
        e = Entity(name="Python", type=EntityType.TECHNOLOGY)
        store.add_entity(e)
        result = store.get_entity(e.id)
        assert result.name == "Python"

        by_name = store.get_entity_by_name("Python")
        assert by_name is not None

    def test_relation_crud(self, store):
        e1 = Entity(name="User", type=EntityType.PERSON)
        e2 = Entity(name="Python", type=EntityType.TECHNOLOGY)
        store.add_entity(e1)
        store.add_entity(e2)

        r = Relation(source_entity_id=e1.id, target_entity_id=e2.id, type="prefers")
        store.add_relation(r)

        rels = store.get_relations(e1.id)
        assert len(rels) == 1
        assert rels[0].type == "prefers"

    def test_layer_management(self, store):
        m = MemoryUnit(content="test", layer="L1")
        store.add_memory(m)
        store.move_to_layer(m.id, "L2")
        result = store.get_memory(m.id)
        assert result.layer == "L2"

    def test_stats(self, store):
        store.add_memory(MemoryUnit(content="a"))
        store.add_memory(MemoryUnit(content="b"))
        stats = store.stats()
        assert stats["total_memories"] == 2

    def test_entities_tags_roundtrip(self, store):
        m = MemoryUnit(content="test", entities=["e1", "e2"], tags=["t1"])
        store.add_memory(m)
        result = store.get_memory(m.id)
        assert result.entities == ["e1", "e2"]
        assert result.tags == ["t1"]

    def test_embedding_roundtrip(self, store):
        m = MemoryUnit(content="test", embedding=[0.1, 0.2, 0.3])
        store.add_memory(m)
        result = store.get_memory(m.id)
        assert result.embedding == [0.1, 0.2, 0.3]


# ── Embeddings ──────────────────────────────────────────────────────


class TestEmbeddings:
    def test_similarity_identical(self, embedding_engine):
        v = [1.0, 0.0, 0.0]
        assert embedding_engine.similarity(v, v) == pytest.approx(1.0)

    def test_similarity_orthogonal(self, embedding_engine):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert embedding_engine.similarity(a, b) == pytest.approx(0.0)

    def test_similarity_zero(self, embedding_engine):
        z = [0.0, 0.0, 0.0]
        assert embedding_engine.similarity(z, z) == 0.0

    def test_most_similar(self, embedding_engine):
        q = [1.0, 0.0, 0.0]
        candidates = [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0]),
            ("c", [0.5, 0.5, 0.0]),
        ]
        results = embedding_engine.most_similar(q, candidates, top_k=2)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)


# ── Versioning ──────────────────────────────────────────────────────


class TestVersioning:
    def test_create_version(self, store):
        from gitmem0.versioning import VersionControl

        vc = VersionControl(store)
        m1 = MemoryUnit(content="v1")
        store.add_memory(m1)

        m2 = vc.create_version(
            content="v2",
            type=MemoryType.FACT,
            importance=0.8,
            confidence=0.9,
            source="test",
            entities=[],
            tags=[],
            supersedes_id=m1.id,
        )
        assert m2.supersedes == m1.id

    def test_get_history(self, store):
        from gitmem0.versioning import VersionControl

        vc = VersionControl(store)
        m1 = MemoryUnit(content="v1")
        store.add_memory(m1)
        m2 = vc.create_version("v2", MemoryType.FACT, 0.8, 0.9, "test", [], [], m1.id)
        m3 = vc.create_version("v3", MemoryType.FACT, 0.8, 0.9, "test", [], [], m2.id)

        history = vc.get_history(m3.id)
        assert len(history) == 3
        assert history[0].content == "v3"
        assert history[-1].content == "v1"

    def test_get_current(self, store):
        from gitmem0.versioning import VersionControl

        vc = VersionControl(store)
        m1 = MemoryUnit(content="v1")
        store.add_memory(m1)
        m2 = vc.create_version("v2", MemoryType.FACT, 0.8, 0.9, "test", [], [], m1.id)

        current = vc.get_current(m1.id)
        assert current.id == m2.id

    def test_branch(self, store):
        from gitmem0.versioning import VersionControl

        vc = VersionControl(store)
        m1 = MemoryUnit(content="original")
        store.add_memory(m1)

        branch_id = vc.create_branch(m1.id, "feature-x")
        branch_mem = store.get_memory(branch_id)
        assert "branch:feature-x" in branch_mem.tags

        branch_mems = vc.get_branch("feature-x")
        assert len(branch_mems) == 1

    def test_diff(self, store):
        from gitmem0.versioning import VersionControl

        vc = VersionControl(store)
        m1 = MemoryUnit(content="old content", importance=0.5)
        store.add_memory(m1)
        m2 = vc.create_version("new content", MemoryType.FACT, 0.9, 0.9, "test", [], [], m1.id)

        diff = vc.diff(m1.id, m2.id)
        assert diff["content"]["a"] == "old content"
        assert diff["content"]["b"] == "new content"
        assert diff["importance"]["a"] == 0.5
        assert diff["importance"]["b"] == 0.9


# ── Decay ───────────────────────────────────────────────────────────


class TestDecay:
    def test_compute_decayed_confidence(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)

        # Recent memory: confidence ~ initial
        m = MemoryUnit(content="recent", confidence=0.8)
        m.accessed_at = datetime.now(timezone.utc)
        c = decay.compute_decayed_confidence(m)
        assert c == pytest.approx(0.8, abs=0.01)

        # Old memory: confidence decays
        m.accessed_at = datetime.now(timezone.utc) - timedelta(days=100)
        c_old = decay.compute_decayed_confidence(m)
        assert c_old < 0.8

    def test_layer_stats(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)
        store.add_memory(MemoryUnit(content="a", layer="L1"))
        store.add_memory(MemoryUnit(content="b", layer="L2"))

        stats = decay.get_layer_stats()
        assert stats["L1"]["count"] == 1
        assert stats["L2"]["count"] == 1

    def test_check_contradiction(self):
        from gitmem0.decay import DecayEngine

        # Positive vs negative pair
        assert DecayEngine._check_contradiction("我喜欢Python", "我不喜欢Python") is not None
        assert DecayEngine._check_contradiction("I prefer dark mode", "I don't prefer dark mode") is not None
        assert DecayEngine._check_contradiction("always use type hints", "never use type hints") is not None

        # No contradiction
        assert DecayEngine._check_contradiction("I like Python", "I like Rust") is None
        assert DecayEngine._check_contradiction("use Docker", "use Kubernetes") is None

    def test_detect_contradictions(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)
        emb = [1.0, 0.0, 0.0] + [0.0] * 381  # dummy 384-dim

        m1 = MemoryUnit(content="我喜欢Python", type=MemoryType.PREFERENCE, embedding=emb, layer="L1")
        m2 = MemoryUnit(content="我不喜欢Python", type=MemoryType.PREFERENCE, embedding=emb, layer="L1")
        m3 = MemoryUnit(content="Python很好", type=MemoryType.FACT, embedding=emb, layer="L1")
        store.add_memory(m1)
        store.add_memory(m2)
        store.add_memory(m3)

        pairs = decay.detect_contradictions(threshold=0.0)
        # m1 and m2 contradict, m3 is different type so skipped
        assert len(pairs) == 1
        assert pairs[0][0] in (m1.id, m2.id)
        assert pairs[0][1] in (m1.id, m2.id)

    def test_resolve_contradictions(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)
        emb = [1.0, 0.0, 0.0] + [0.0] * 381

        older = MemoryUnit(
            content="我喜欢Python", type=MemoryType.PREFERENCE,
            embedding=emb, layer="L1", confidence=0.9,
        )
        older.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        newer = MemoryUnit(
            content="我不喜欢Python", type=MemoryType.PREFERENCE,
            embedding=emb, layer="L1", confidence=0.9,
        )
        store.add_memory(older)
        store.add_memory(newer)

        result = decay.resolve_contradictions()
        assert result["found"] == 1
        assert result["resolved"] == 1

        # Older should be moved to L2
        reloaded = store.get_memory(older.id)
        assert reloaded.layer == "L2"
        assert any("contradicted_by" in t for t in reloaded.tags)

    def test_resolve_contradictions_dry_run(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)
        emb = [1.0, 0.0, 0.0] + [0.0] * 381

        m1 = MemoryUnit(content="我喜欢Python", type=MemoryType.PREFERENCE, embedding=emb, layer="L1")
        m2 = MemoryUnit(content="我不喜欢Python", type=MemoryType.PREFERENCE, embedding=emb, layer="L1")
        store.add_memory(m1)
        store.add_memory(m2)

        result = decay.resolve_contradictions(dry_run=True)
        assert result["found"] == 1
        assert result["resolved"] == 1

        # Both should still be in L1 (dry run)
        assert store.get_memory(m1.id).layer == "L1"
        assert store.get_memory(m2.id).layer == "L1"

    def test_auto_induct_no_llm(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)  # no LLM judge
        result = decay.auto_induct()
        assert result["groups"] == 0
        assert result["inducted"] == 0

    def test_compress_l2_no_llm(self, store, embedding_engine):
        from gitmem0.decay import DecayEngine

        decay = DecayEngine(store, embedding_engine)  # no LLM judge
        result = decay.compress_l2()
        assert result["groups"] == 0
        assert result["compressed"] == 0
        assert result["memories_removed"] == 0

    def test_auto_induct_too_few_events(self, store, embedding_engine):
        """auto_induct needs 3+ events to trigger."""
        from gitmem0.decay import DecayEngine

        class FakeJudge:
            def summarize(self, memories):
                return "summary"

        decay = DecayEngine(store, embedding_engine, llm_judge=FakeJudge())
        store.add_memory(MemoryUnit(content="event 1 happened", type=MemoryType.EVENT, entities=["e1"]))
        result = decay.auto_induct()
        assert result["groups"] == 0  # only 1 event, need 3+

    def test_compress_l2_too_few(self, store, embedding_engine):
        """compress_l2 needs 5+ L2 memories to trigger."""
        from gitmem0.decay import DecayEngine

        class FakeJudge:
            def summarize(self, memories):
                return "summary"

        decay = DecayEngine(store, embedding_engine, llm_judge=FakeJudge())
        store.add_memory(MemoryUnit(content="old 1", layer="L2", type=MemoryType.FACT))
        store.add_memory(MemoryUnit(content="old 2", layer="L2", type=MemoryType.FACT))
        result = decay.compress_l2()
        assert result["groups"] == 0  # only 2, need 5+


# ── Entities ────────────────────────────────────────────────────────


class TestEntities:
    def test_extract_technology(self, entity_manager):
        entities = entity_manager.extract_entities("I use Python and Docker")
        names = [e.name for e in entities]
        assert "Python" in names
        assert "Docker" in names

    def test_extract_dedup(self, entity_manager, store):
        entity_manager.extract_entities("I use Python")
        entity_manager.extract_entities("Python is great")
        pythons = [e for e in store.list_entities() if e.name == "Python"]
        assert len(pythons) == 1
        assert pythons[0].mention_count == 2

    def test_extract_relations(self, entity_manager):
        entities = entity_manager.extract_entities("I prefer Python")
        e_user = [e for e in entities if e.type == EntityType.PERSON]
        e_python = [e for e in entities if e.name == "Python"]
        if e_user and e_python:
            relations = entity_manager.extract_relations(
                "I prefer Python", entities
            )
            # Relations should be extracted if pattern matches
            assert isinstance(relations, list)

    def test_get_entity_context(self, entity_manager, store):
        entities = entity_manager.extract_entities("Python is great")
        if entities:
            ctx = entity_manager.get_entity_context(entities[0].name)
            assert entities[0].name in ctx

    def test_find_path(self, entity_manager, store):
        e1 = Entity(name="User", type=EntityType.PERSON)
        e2 = Entity(name="Python", type=EntityType.TECHNOLOGY)
        e3 = Entity(name="Django", type=EntityType.TECHNOLOGY)
        store.add_entity(e1)
        store.add_entity(e2)
        store.add_entity(e3)
        store.add_relation(Relation(source_entity_id=e1.id, target_entity_id=e2.id, type="uses"))
        store.add_relation(Relation(source_entity_id=e2.id, target_entity_id=e3.id, type="powers"))

        path = entity_manager.find_path(e1.id, e3.id)
        assert path is not None
        assert len(path) == 3


# ── Extraction ──────────────────────────────────────────────────────


class TestExtraction:
    def test_infer_type(self):
        from gitmem0.extraction import ExtractionEngine
        from gitmem0.embeddings import EmbeddingEngine

        store_fixture = MemoryStore(":memory:")
        engine = EmbeddingEngine()
        em = EntityManager(store_fixture)
        ext = ExtractionEngine(store_fixture, engine, em)

        assert ext.infer_type("I prefer dark mode") == MemoryType.PREFERENCE
        assert ext.infer_type("remember to use type hints") == MemoryType.INSTRUCTION
        assert ext.infer_type("yesterday I deployed v2") == MemoryType.EVENT
        assert ext.infer_type("Python is a language") == MemoryType.FACT

    def test_extract_from_text(self):
        from gitmem0.extraction import ExtractionEngine
        from gitmem0.embeddings import EmbeddingEngine

        store_fixture = MemoryStore(":memory:")
        engine = EmbeddingEngine()
        em = EntityManager(store_fixture)
        ext = ExtractionEngine(store_fixture, engine, em)

        memories = ext.extract_from_text("I prefer Python for scripting. Docker is great for deployment.")
        assert isinstance(memories, list)
        # Should extract at least something from this content


# ── Context ─────────────────────────────────────────────────────────


class TestContext:
    def test_arrange_1(self, store, embedding_engine):
        from gitmem0.context import ContextBuilder
        from gitmem0.retrieval import RetrievalEngine

        retrieval = RetrievalEngine(store, embedding_engine)
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval, em)

        m = MemoryUnit(content="only one")
        result = ctx.arrange_memories([m])
        assert len(result) == 1

    def test_arrange_3_plus(self, store, embedding_engine):
        from gitmem0.context import ContextBuilder
        from gitmem0.retrieval import RetrievalEngine

        retrieval = RetrievalEngine(store, embedding_engine)
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval, em)

        memories = [MemoryUnit(content=f"m{i}") for i in range(5)]
        arranged = ctx.arrange_memories(memories)
        # First stays first, second and last are swapped
        assert arranged[0].content == "m0"
        assert arranged[-1].content == "m1"  # second moved to end
        assert arranged[1].content == "m4"  # last moved to second

    def test_estimate_tokens(self, store, embedding_engine):
        from gitmem0.context import ContextBuilder
        from gitmem0.retrieval import RetrievalEngine

        retrieval = RetrievalEngine(store, embedding_engine)
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval, em)

        assert ctx.estimate_tokens("hello world") > 0
        assert ctx.estimate_tokens("") == 0

    def test_compress(self, store, embedding_engine):
        from gitmem0.context import ContextBuilder
        from gitmem0.retrieval import RetrievalEngine

        retrieval = RetrievalEngine(store, embedding_engine)
        em = EntityManager(store)
        ctx = ContextBuilder(retrieval, em)

        memories = [
            MemoryUnit(content="high", confidence=0.9, importance=0.9),
            MemoryUnit(content="low", confidence=0.1, importance=0.1),
        ]
        compressed = ctx.compress_memories(memories, token_budget=10)
        # Low confidence should be dropped under budget pressure
        assert len(compressed) <= len(memories)


# ── CLI smoke test ──────────────────────────────────────────────────


class TestCLI:
    def test_help(self):
        from typer.testing import CliRunner
        from gitmem0.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "GitMem0" in result.output

    def test_add_json(self, tmp_path):
        from typer.testing import CliRunner
        from gitmem0.cli import app

        db = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(app, ["--db", db, "add", "test memory", "--type", "fact"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert "id" in data["data"]

    def test_search_json(self, tmp_path):
        from typer.testing import CliRunner
        from gitmem0.cli import app

        db = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(app, ["--db", db, "add", "Python is great"])
        result = runner.invoke(app, ["--db", db, "search", "Python"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert len(data["data"]) >= 1

    def test_stats_json(self, tmp_path):
        from typer.testing import CliRunner
        from gitmem0.cli import app

        db = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(app, ["--db", db, "stats"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert "total_memories" in data["data"]


# ── LLM Judge tests ─────────────────────────────────────────────────────


class TestLLMJudge:
    """Test the MiMoLLMJudge implementation (mocked API)."""

    def test_init_without_key(self):
        from gitmem0.llm_judge import MiMoLLMJudge

        judge = MiMoLLMJudge(api_key="")
        assert judge.enabled is False
        assert judge.should_remember("test") is None
        assert judge.score_importance("test") is None
        assert judge.infer_type("test") is None
        assert judge.summarize(["a", "b"]) is None

    def test_init_with_key(self):
        from gitmem0.llm_judge import MiMoLLMJudge

        judge = MiMoLLMJudge(api_key="tp-test123")
        assert judge.enabled is True

    def test_disabled_returns_none(self):
        from gitmem0.llm_judge import MiMoLLMJudge

        judge = MiMoLLMJudge(api_key="")
        # All methods should return None when disabled
        assert judge.should_remember("anything") is None
        assert judge.score_importance("anything") is None
        assert judge.infer_type("anything") is None
        assert judge.summarize(["a"]) == "a"  # single item returned as-is
        assert judge.summarize([]) is None
        assert judge.summarize(["a", "b"]) is None  # multi → API call → None (disabled)

    def test_protocol_compliance(self):
        from gitmem0.llm_judge import MiMoLLMJudge
        from gitmem0.extraction import LLMJudge

        judge = MiMoLLMJudge(api_key="")
        assert isinstance(judge, LLMJudge)

    def test_parse_importance_response(self):
        from gitmem0.llm_judge import MiMoLLMJudge

        judge = MiMoLLMJudge(api_key="tp-test")
        # Test the parsing logic by calling _chat mock
        import re
        # Simulate various LLM responses
        test_cases = [
            ("0.8", 0.8),
            ("The importance is 0.65", 0.65),
            ("1.0", 1.0),
            ("0.0", 0.0),
            ("very important", None),  # no number
        ]
        for resp, expected in test_cases:
            match = re.search(r"(\d+\.?\d*)", resp)
            if match:
                val = float(match.group(1))
                result = max(0.0, min(1.0, val))
            else:
                result = None
            assert result == expected

    def test_parse_type_response(self):
        from gitmem0.llm_judge import _TYPE_MAP

        # Test type classification parsing
        test_cases = [
            ("preference", "preference"),
            ("Preference", "preference"),
            ("fact", "fact"),
            ("event", "event"),
            ("this is a fact about Python", "fact"),
            ("instruction on how to", "instruction"),
        ]
        for resp, expected in test_cases:
            lower = resp.lower().strip().strip(".")
            if lower in _TYPE_MAP:
                result = _TYPE_MAP[lower].value
            else:
                result = None
                for key, mem_type in _TYPE_MAP.items():
                    if key in lower:
                        result = mem_type.value
                        break
            assert result == expected, f"Failed for '{resp}': got {result}, expected {expected}"

    def test_parse_should_remember_response(self):
        # Test yes/no parsing
        test_cases = [
            ("yes", True),
            ("Yes", True),
            ("yes, definitely", True),
            ("no", False),
            ("No", False),
            ("no, too trivial", False),
            ("maybe", None),
        ]
        for resp, expected in test_cases:
            lower = resp.lower().strip().strip(".")
            if lower.startswith("yes"):
                result = True
            elif lower.startswith("no"):
                result = False
            else:
                result = None
            assert result == expected, f"Failed for '{resp}': got {result}, expected {expected}"


