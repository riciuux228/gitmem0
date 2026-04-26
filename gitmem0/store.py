"""SQLite storage layer for GitMem0.

Provides MemoryStore with full CRUD for MemoryUnit, Entity, and Relation,
plus FTS5 full-text search and layer management.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gitmem0.models import Entity, EntityType, MemoryType, MemoryUnit, Relation


def _sanitize(s: str) -> str:
    """Remove surrogate characters that break UTF-8 encoding (Windows pipe issue)."""
    return s.encode("utf-8", errors="ignore").decode("utf-8")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT '',
    entities TEXT NOT NULL DEFAULT '[]',
    supersedes TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    embedding TEXT,
    layer TEXT NOT NULL DEFAULT 'L1'
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

CREATE TABLE IF NOT EXISTS relations (
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_entity_id, target_entity_id, type)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    id UNINDEXED,
    tokenize='trigram'
);

"""


def _row_to_memory(row: sqlite3.Row) -> MemoryUnit:
    return MemoryUnit(
        id=row["id"],
        content=row["content"],
        type=MemoryType(row["type"]),
        importance=row["importance"],
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
        accessed_at=datetime.fromisoformat(row["accessed_at"]),
        access_count=row["access_count"],
        source=row["source"],
        entities=json.loads(row["entities"]),
        supersedes=row["supersedes"],
        tags=json.loads(row["tags"]),
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        layer=row["layer"],
    )


def _row_to_entity(row: sqlite3.Row) -> Entity:
    return Entity(
        id=row["id"],
        name=row["name"],
        type=EntityType(row["type"]),
        aliases=json.loads(row["aliases"]),
        first_seen=datetime.fromisoformat(row["first_seen"]),
        last_seen=datetime.fromisoformat(row["last_seen"]),
        mention_count=row["mention_count"],
    )


def _row_to_relation(row: sqlite3.Row) -> Relation:
    return Relation(
        source_entity_id=row["source_entity_id"],
        target_entity_id=row["target_entity_id"],
        type=row["type"],
        weight=row["weight"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class _LRUCache:
    """Thread-safe LRU cache with max size."""

    def __init__(self, maxsize: int = 1000):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return None

    def put(self, key, value):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


class MemoryStore:
    """SQLite-backed storage for memories, entities, and relations."""

    def __init__(self, db_path: str | Path, cache_size: int = 1000) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        # L0 Hot Cache
        self._mem_cache = _LRUCache(maxsize=cache_size)
        self._list_cache = _LRUCache(maxsize=100)
        self._cache_dirty = True
        # In-memory hash table indexes
        self._idx_lock = threading.Lock()
        self._memory_index: dict[str, MemoryUnit] = {}
        self._entity_index: dict[str, Entity] = {}
        self._entity_name_index: dict[str, Entity] = {}
        self._entity_to_memories: dict[str, set[str]] = {}
        self._embedding_index: dict[str, list[float]] = {}
        # Phase 2: content inverted index + relations index + FTS cache + stats counters
        self._content_index: dict[str, set[str]] = {}  # token → {memory_id, ...}
        self._relation_index: dict[str, list[Relation]] = {}  # entity_id → [Relation, ...]
        self._fts_cache: dict[str, tuple[float, list[tuple[str, float]]]] = {}  # query → (ts, results)
        self._fts_cache_ttl = 300.0  # 5 minutes
        self._stats_counters: dict[str, int] = {"total_memories": 0, "total_entities": 0, "total_relations": 0}
        self._layer_counters: dict[str, int] = {}
        self._build_indexes()

    @staticmethod
    def _tokenize_content(text: str) -> list[str]:
        """Extract ASCII tokens (2+ chars) for inverted index."""
        return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_.+#-]{1,}", text)]

    def _build_indexes(self) -> None:
        """Build all in-memory indexes from SQLite. Called once at startup."""
        # Memory indexes
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        for row in rows:
            unit = _row_to_memory(row)
            self._memory_index[unit.id] = unit
            if unit.embedding is not None:
                self._embedding_index[unit.id] = unit.embedding
            for eid in unit.entities:
                self._entity_to_memories.setdefault(eid, set()).add(unit.id)
            # Content inverted index
            for token in self._tokenize_content(unit.content):
                self._content_index.setdefault(token, set()).add(unit.id)
            # Layer counters
            self._layer_counters[unit.layer] = self._layer_counters.get(unit.layer, 0) + 1
        # Entity indexes
        rows = self._conn.execute("SELECT * FROM entities").fetchall()
        for row in rows:
            entity = _row_to_entity(row)
            self._entity_index[entity.id] = entity
            self._entity_name_index[entity.name.lower()] = entity
            for alias in entity.aliases:
                self._entity_name_index[alias.lower()] = entity
        # Relation index
        rows = self._conn.execute("SELECT * FROM relations").fetchall()
        for row in rows:
            rel = _row_to_relation(row)
            self._relation_index.setdefault(rel.source_entity_id, []).append(rel)
            if rel.target_entity_id != rel.source_entity_id:
                self._relation_index.setdefault(rel.target_entity_id, []).append(rel)
        # Stats counters
        self._stats_counters["total_memories"] = len(self._memory_index)
        self._stats_counters["total_entities"] = len(self._entity_index)
        self._stats_counters["total_relations"] = len(rows)

    # ── Index update helpers ───────────────────────────────────────

    def _index_add_memory(self, unit: MemoryUnit) -> None:
        self._memory_index[unit.id] = unit
        if unit.embedding is not None:
            self._embedding_index[unit.id] = unit.embedding
        for eid in unit.entities:
            self._entity_to_memories.setdefault(eid, set()).add(unit.id)
        # Content inverted index
        for token in self._tokenize_content(unit.content):
            self._content_index.setdefault(token, set()).add(unit.id)
        # Stats counters
        self._stats_counters["total_memories"] = self._stats_counters.get("total_memories", 0) + 1
        self._layer_counters[unit.layer] = self._layer_counters.get(unit.layer, 0) + 1
        self._fts_cache.clear()

    def _index_update_memory(self, old: MemoryUnit | None, new: MemoryUnit) -> None:
        # Content index: diff tokens
        if old is not None:
            old_tokens = set(self._tokenize_content(old.content))
            new_tokens = set(self._tokenize_content(new.content))
            for token in old_tokens - new_tokens:
                s = self._content_index.get(token)
                if s:
                    s.discard(new.id)
                    if not s:
                        del self._content_index[token]
            for token in new_tokens - old_tokens:
                self._content_index.setdefault(token, set()).add(new.id)
            # Layer counter diff
            if old.layer != new.layer:
                self._layer_counters[old.layer] = max(0, self._layer_counters.get(old.layer, 1) - 1)
                self._layer_counters[new.layer] = self._layer_counters.get(new.layer, 0) + 1
        else:
            for token in self._tokenize_content(new.content):
                self._content_index.setdefault(token, set()).add(new.id)
        self._memory_index[new.id] = new
        # Update embedding index
        if new.embedding is not None:
            self._embedding_index[new.id] = new.embedding
        elif new.id in self._embedding_index:
            del self._embedding_index[new.id]
        # Diff entity lists
        if old is not None:
            old_entities = set(old.entities)
            new_entities = set(new.entities)
            for eid in old_entities - new_entities:
                s = self._entity_to_memories.get(eid)
                if s:
                    s.discard(new.id)
                    if not s:
                        del self._entity_to_memories[eid]
            for eid in new_entities - old_entities:
                self._entity_to_memories.setdefault(eid, set()).add(new.id)
        else:
            for eid in new.entities:
                self._entity_to_memories.setdefault(eid, set()).add(new.id)
        self._fts_cache.clear()

    def _index_delete_memory(self, id: str) -> None:
        unit = self._memory_index.pop(id, None)
        self._embedding_index.pop(id, None)
        if unit is not None:
            # Content inverted index
            for token in self._tokenize_content(unit.content):
                s = self._content_index.get(token)
                if s:
                    s.discard(id)
                    if not s:
                        del self._content_index[token]
            # Entity index
            for eid in unit.entities:
                s = self._entity_to_memories.get(eid)
                if s:
                    s.discard(id)
                    if not s:
                        del self._entity_to_memories[eid]
            # Stats counters
            self._stats_counters["total_memories"] = max(0, self._stats_counters.get("total_memories", 1) - 1)
            self._layer_counters[unit.layer] = max(0, self._layer_counters.get(unit.layer, 1) - 1)
        self._fts_cache.clear()

    def _index_add_entity(self, entity: Entity) -> None:
        self._entity_index[entity.id] = entity
        self._entity_name_index[entity.name.lower()] = entity
        for alias in entity.aliases:
            self._entity_name_index[alias.lower()] = entity
        self._stats_counters["total_entities"] = self._stats_counters.get("total_entities", 0) + 1

    def _index_update_entity(self, old: Entity | None, new: Entity) -> None:
        # Remove old name/aliases from name index
        if old is not None:
            self._entity_name_index.pop(old.name.lower(), None)
            for alias in old.aliases:
                self._entity_name_index.pop(alias.lower(), None)
        # Add new
        self._entity_index[new.id] = new
        self._entity_name_index[new.name.lower()] = new
        for alias in new.aliases:
            self._entity_name_index[alias.lower()] = new

    # ── Public index accessors ─────────────────────────────────────

    def get_entity_memories(self, entity_id: str) -> set[str]:
        """Return set of memory IDs that reference this entity. O(1) lookup."""
        return set(self._entity_to_memories.get(entity_id, set()))

    def get_embeddings(self) -> dict[str, list[float]]:
        """Return copy of embedding index. O(1) + O(n) copy."""
        return dict(self._embedding_index)

    def close(self) -> None:
        self._conn.close()

    def invalidate_cache(self) -> None:
        """Clear all caches. Called on any write operation."""
        self._mem_cache.clear()
        self._list_cache.clear()
        self._cache_dirty = True

    # ── Memory CRUD ──────────────────────────────────────────────

    def add_memory(self, unit: MemoryUnit) -> None:
        cur = self._conn.execute(
            """INSERT INTO memories
               (id, content, type, importance, confidence, created_at, accessed_at,
                access_count, source, entities, supersedes, tags, embedding, layer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                unit.id,
                _sanitize(unit.content),
                unit.type.value,
                unit.importance,
                unit.confidence,
                unit.created_at.isoformat(),
                unit.accessed_at.isoformat(),
                unit.access_count,
                unit.source,
                json.dumps(unit.entities),
                unit.supersedes,
                json.dumps(unit.tags),
                json.dumps(unit.embedding) if unit.embedding is not None else None,
                unit.layer,
            ),
        )
        self._conn.execute(
            "INSERT INTO memories_fts(rowid, content, id) VALUES (?, ?, ?)",
            (cur.lastrowid, _sanitize(unit.content), unit.id),
        )
        self._conn.commit()
        with self._idx_lock:
            self._index_add_memory(unit)
        self.invalidate_cache()

    def get_memory(self, id: str) -> Optional[MemoryUnit]:
        # O(1) dict lookup — no SQL needed
        return self._memory_index.get(id)

    def update_memory(self, unit: MemoryUnit) -> None:
        # Grab old unit from index for entity diff
        old_unit = self._memory_index.get(unit.id)
        # Sync FTS: delete old row, insert new after update
        old_row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (unit.id,)
        ).fetchone()
        self._conn.execute(
            """UPDATE memories SET
               content=?, type=?, importance=?, confidence=?, created_at=?,
               accessed_at=?, access_count=?, source=?, entities=?, supersedes=?,
               tags=?, embedding=?, layer=?
               WHERE id=?""",
            (
                _sanitize(unit.content),
                unit.type.value,
                unit.importance,
                unit.confidence,
                unit.created_at.isoformat(),
                unit.accessed_at.isoformat(),
                unit.access_count,
                unit.source,
                json.dumps(unit.entities),
                unit.supersedes,
                json.dumps(unit.tags),
                json.dumps(unit.embedding) if unit.embedding is not None else None,
                unit.layer,
                unit.id,
            ),
        )
        if old_row:
            self._conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?", (old_row["rowid"],)
            )
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, content, id) VALUES (?, ?, ?)",
                (old_row["rowid"], _sanitize(unit.content), unit.id),
            )
        self._conn.commit()
        with self._idx_lock:
            self._index_update_memory(old_unit, unit)
        self.invalidate_cache()

    def delete_memory(self, id: str) -> None:
        old_row = self._conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (id,)
        ).fetchone()
        self._conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        if old_row:
            self._conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?", (old_row["rowid"],)
            )
        self._conn.commit()
        with self._idx_lock:
            self._index_delete_memory(id)
        self.invalidate_cache()

    def list_memories(
        self,
        type: Optional[MemoryType] = None,
        layer: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        # From in-memory index, filter + sort
        units = list(self._memory_index.values())
        if type is not None:
            units = [u for u in units if u.type == type]
        if layer is not None:
            units = [u for u in units if u.layer == layer]
        units.sort(key=lambda u: u.created_at, reverse=True)
        return units[:limit]

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Escape special FTS5 characters to prevent syntax errors."""
        # FTS5 special chars: " * + - ( ) : ^ { } ~
        # Strategy: wrap each token in double quotes for literal matching
        tokens = query.split()
        if not tokens:
            return '""'
        escaped = []
        for t in tokens:
            # Remove any existing double quotes, then wrap
            t = t.replace('"', '')
            escaped.append(f'"{t}"')
        return " ".join(escaped)

    def search_fts(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        # FTS result cache — skip SQLite if query was seen within TTL
        now = time.monotonic()
        cached = self._fts_cache.get(query)
        if cached is not None:
            ts, results = cached
            if now - ts < self._fts_cache_ttl:
                return results[:limit]
            else:
                del self._fts_cache[query]
        safe_query = self._sanitize_fts(query)
        try:
            rows = self._conn.execute(
                """SELECT id, rank FROM memories_fts
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
            results = [(r["id"], r["rank"]) for r in rows]
        except Exception:
            results = []
        self._fts_cache[query] = (now, results)
        return results

    def search_content(self, query: str, limit: int = 20) -> list[str]:
        """Content search: in-memory inverted index for ASCII tokens, SQL LIKE fallback for CJK."""
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return []

        # Classify tokens: ASCII (use inverted index) vs CJK (use SQL LIKE)
        ascii_tokens = []
        cjk_tokens = []
        for t in tokens:
            if re.search(r"[A-Za-z]{2,}", t):
                ascii_tokens.append(t.lower())
            elif re.search(r"[一-鿿㐀-䶿　-〿]", t):
                cjk_tokens.append(t)
            else:
                ascii_tokens.append(t.lower())

        candidate_ids: set[str] | None = None

        # In-memory inverted index for ASCII tokens — O(tokens × ~matching_ids)
        if ascii_tokens:
            for i, token in enumerate(ascii_tokens):
                matching = self._content_index.get(token, set())
                if not matching:
                    # Try prefix match for partial tokens
                    matching = set()
                    for idx_token, mids in self._content_index.items():
                        if idx_token.startswith(token):
                            matching.update(mids)
                if i == 0:
                    candidate_ids = set(matching)
                else:
                    candidate_ids &= matching
                if not candidate_ids:
                    return []

        # SQL LIKE fallback for CJK tokens — only over candidates if ASCII narrowed it
        if cjk_tokens:
            if candidate_ids is not None and len(candidate_ids) <= limit:
                # Small candidate set: filter in Python
                results = []
                for mid in candidate_ids:
                    unit = self._memory_index.get(mid)
                    if unit and all(t in unit.content for t in cjk_tokens):
                        results.append(mid)
                        if len(results) >= limit:
                            break
                return results
            else:
                # No ASCII filter or too many candidates: SQL LIKE
                conditions = " AND ".join(["content LIKE ?" for _ in cjk_tokens])
                params = [f"%{t}%" for t in cjk_tokens]
                params.append(limit)
                rows = self._conn.execute(
                    f"SELECT id FROM memories WHERE {conditions} LIMIT ?",
                    params,
                ).fetchall()
                cjk_ids = [r["id"] for r in rows]
                if candidate_ids is not None:
                    return [mid for mid in cjk_ids if mid in candidate_ids]
                return cjk_ids

        if candidate_ids is not None:
            return list(candidate_ids)[:limit]
        return []

    # ── Entity CRUD ──────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        self._conn.execute(
            """INSERT INTO entities (id, name, type, aliases, first_seen, last_seen, mention_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.id,
                entity.name,
                entity.type.value,
                json.dumps(entity.aliases),
                entity.first_seen.isoformat(),
                entity.last_seen.isoformat(),
                entity.mention_count,
            ),
        )
        self._conn.commit()
        with self._idx_lock:
            self._index_add_entity(entity)

    def update_entity(self, entity: Entity) -> None:
        old_entity = self._entity_index.get(entity.id)
        self._conn.execute(
            """UPDATE entities SET
               name=?, type=?, aliases=?, first_seen=?, last_seen=?, mention_count=?
               WHERE id=?""",
            (
                entity.name,
                entity.type.value,
                json.dumps(entity.aliases),
                entity.first_seen.isoformat(),
                entity.last_seen.isoformat(),
                entity.mention_count,
                entity.id,
            ),
        )
        self._conn.commit()
        with self._idx_lock:
            self._index_update_entity(old_entity, entity)

    def get_entity(self, id: str) -> Optional[Entity]:
        # O(1) dict lookup
        return self._entity_index.get(id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        # O(1) dict lookup (case-insensitive via lowercase key)
        return self._entity_name_index.get(name.lower())

    def list_entities(
        self,
        type: Optional[EntityType] = None,
        limit: int = 50,
    ) -> list[Entity]:
        # From in-memory index, filter + sort
        entities = list(self._entity_index.values())
        if type is not None:
            entities = [e for e in entities if e.type == type]
        entities.sort(key=lambda e: e.mention_count, reverse=True)
        return entities[:limit]

    # ── Relation CRUD ────────────────────────────────────────────

    def add_relation(self, relation: Relation) -> None:
        # Remove existing relation with same key if replacing
        self._remove_relation_from_index(relation.source_entity_id, relation.target_entity_id, relation.type)
        self._conn.execute(
            """INSERT OR REPLACE INTO relations
               (source_entity_id, target_entity_id, type, weight, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                relation.source_entity_id,
                relation.target_entity_id,
                relation.type,
                relation.weight,
                relation.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        # Update relation index
        with self._idx_lock:
            self._relation_index.setdefault(relation.source_entity_id, []).append(relation)
            if relation.target_entity_id != relation.source_entity_id:
                self._relation_index.setdefault(relation.target_entity_id, []).append(relation)
            self._stats_counters["total_relations"] = self._stats_counters.get("total_relations", 0) + 1

    def get_relations(self, entity_id: str) -> list[Relation]:
        # O(1) dict lookup — no SQL needed
        return list(self._relation_index.get(entity_id, []))

    def _remove_relation_from_index(self, source_id: str, target_id: str, rel_type: str) -> None:
        """Remove a specific relation from the in-memory index."""
        for eid in (source_id, target_id):
            rels = self._relation_index.get(eid)
            if rels:
                filtered = [r for r in rels if not (
                    r.source_entity_id == source_id and r.target_entity_id == target_id and r.type == rel_type
                )]
                if len(filtered) < len(rels):
                    self._stats_counters["total_relations"] = max(0, self._stats_counters.get("total_relations", 1) - 1)
                if filtered:
                    self._relation_index[eid] = filtered
                else:
                    del self._relation_index[eid]

    def delete_relation(
        self, source_id: str, target_id: str, type: str
    ) -> None:
        self._conn.execute(
            """DELETE FROM relations
               WHERE source_entity_id = ? AND target_entity_id = ? AND type = ?""",
            (source_id, target_id, type),
        )
        self._conn.commit()
        with self._idx_lock:
            self._remove_relation_from_index(source_id, target_id, type)

    # ── Layer management ─────────────────────────────────────────

    def get_memories_by_layer(self, layer: str) -> list[MemoryUnit]:
        units = [u for u in self._memory_index.values() if u.layer == layer]
        units.sort(key=lambda u: u.created_at, reverse=True)
        return units

    def move_to_layer(self, memory_id: str, new_layer: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE memories SET layer = ?, accessed_at = ? WHERE id = ?""",
            (new_layer, now, memory_id),
        )
        self._conn.commit()
        # Update index: refresh the unit's layer and accessed_at
        unit = self._memory_index.get(memory_id)
        if unit is not None:
            unit.layer = new_layer
            unit.accessed_at = datetime.fromisoformat(now)
        self.invalidate_cache()

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict:
        # O(1) from in-memory counters — no SQL COUNT
        return {
            "total_memories": self._stats_counters.get("total_memories", 0),
            "layers": dict(self._layer_counters),
            "total_entities": self._stats_counters.get("total_entities", 0),
            "total_relations": self._stats_counters.get("total_relations", 0),
        }
