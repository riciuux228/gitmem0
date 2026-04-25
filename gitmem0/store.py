"""SQLite storage layer for GitMem0.

Provides MemoryStore with full CRUD for MemoryUnit, Entity, and Relation,
plus FTS5 full-text search and layer management.
"""

from __future__ import annotations

import json
import sqlite3
import threading
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
        self.invalidate_cache()

    def get_memory(self, id: str) -> Optional[MemoryUnit]:
        cached = self._mem_cache.get(("mem", id))
        if cached is not None:
            return cached
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (id,)
        ).fetchone()
        result = _row_to_memory(row) if row else None
        if result is not None:
            self._mem_cache.put(("mem", id), result)
        return result

    def update_memory(self, unit: MemoryUnit) -> None:
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
        self.invalidate_cache()

    def list_memories(
        self,
        type: Optional[MemoryType] = None,
        layer: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        cache_key = ("list", type.value if type else None, layer, limit)
        cached = self._list_cache.get(cache_key)
        if cached is not None:
            return cached
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if type is not None:
            query += " AND type = ?"
            params.append(type.value)
        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        result = [_row_to_memory(r) for r in rows]
        self._list_cache.put(cache_key, result)
        return result

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
        safe_query = self._sanitize_fts(query)
        try:
            rows = self._conn.execute(
                """SELECT id, rank FROM memories_fts
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
            return [(r["id"], r["rank"]) for r in rows]
        except Exception:
            # Fallback: return empty on any FTS error
            return []

    def search_content(self, query: str, limit: int = 20) -> list[str]:
        """LIKE-based keyword search on content. Works for CJK text where FTS5 fails."""
        # Split query into tokens (space-separated)
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return []

        # Each token must appear in content (AND logic)
        conditions = " AND ".join(["content LIKE ?" for _ in tokens])
        params = [f"%{t}%" for t in tokens]
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT id FROM memories WHERE {conditions} LIMIT ?",
            params,
        ).fetchall()
        return [r["id"] for r in rows]

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

    def update_entity(self, entity: Entity) -> None:
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

    def get_entity(self, id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (id,)
        ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()
        return _row_to_entity(row) if row else None

    def list_entities(
        self,
        type: Optional[EntityType] = None,
        limit: int = 50,
    ) -> list[Entity]:
        query = "SELECT * FROM entities WHERE 1=1"
        params: list = []
        if type is not None:
            query += " AND type = ?"
            params.append(type.value)
        query += " ORDER BY mention_count DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_entity(r) for r in rows]

    # ── Relation CRUD ────────────────────────────────────────────

    def add_relation(self, relation: Relation) -> None:
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

    def get_relations(self, entity_id: str) -> list[Relation]:
        rows = self._conn.execute(
            """SELECT * FROM relations
               WHERE source_entity_id = ? OR target_entity_id = ?""",
            (entity_id, entity_id),
        ).fetchall()
        return [_row_to_relation(r) for r in rows]

    def delete_relation(
        self, source_id: str, target_id: str, type: str
    ) -> None:
        self._conn.execute(
            """DELETE FROM relations
               WHERE source_entity_id = ? AND target_entity_id = ? AND type = ?""",
            (source_id, target_id, type),
        )
        self._conn.commit()

    # ── Layer management ─────────────────────────────────────────

    def get_memories_by_layer(self, layer: str) -> list[MemoryUnit]:
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE layer = ? ORDER BY created_at DESC",
            (layer,),
        ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def move_to_layer(self, memory_id: str, new_layer: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE memories SET layer = ?, accessed_at = ? WHERE id = ?""",
            (new_layer, now, memory_id),
        )
        self._conn.commit()
        self.invalidate_cache()

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict:
        layer_rows = self._conn.execute(
            "SELECT layer, COUNT(*) as cnt FROM memories GROUP BY layer"
        ).fetchall()
        layers = {r["layer"]: r["cnt"] for r in layer_rows}
        total_memories = sum(layers.values())

        total_entities = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM entities"
        ).fetchone()["cnt"]

        total_relations = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM relations"
        ).fetchone()["cnt"]

        return {
            "total_memories": total_memories,
            "layers": layers,
            "total_entities": total_entities,
            "total_relations": total_relations,
        }
