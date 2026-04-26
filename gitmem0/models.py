"""Core data models for GitMem0.

Memory Unit: immutable atom of memory (like a git commit)
Entity: node in the knowledge graph
Relation: edge in the knowledge graph
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    """Types of memory units."""
    FACT = "fact"              # Objective facts ("Python is a programming language")
    PREFERENCE = "preference"  # User preferences ("I prefer dark mode")
    EVENT = "event"            # Events that happened ("Deployed v2.0 on Monday")
    INSIGHT = "insight"        # Derived insights ("React is better for this UI")
    INSTRUCTION = "instruction"  # Directives ("Always use type hints")
    EXPERIENCE = "experience"  # Lessons learned ("Fixed auth bug: token expiry was wrong")


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    TECHNOLOGY = "technology"
    PROJECT = "project"
    CONCEPT = "concept"
    LOCATION = "location"
    ORGANIZATION = "organization"


# Importance weights for each memory type
TYPE_IMPORTANCE_WEIGHTS: dict[MemoryType, float] = {
    MemoryType.PREFERENCE: 0.9,
    MemoryType.INSTRUCTION: 0.9,
    MemoryType.EXPERIENCE: 0.85,
    MemoryType.INSIGHT: 0.8,
    MemoryType.FACT: 0.7,
    MemoryType.EVENT: 0.7,
}

# Default weights for multi-signal scoring
DEFAULT_SCORING_WEIGHTS = {
    "semantic": 0.25,
    "bm25": 0.15,
    "entity": 0.15,
    "recency": 0.10,
    "importance": 0.20,
    "confidence": 0.15,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class MemoryUnit:
    """An immutable atom of memory. Like a git commit, once created it never changes.

    When information evolves, a new MemoryUnit is created with `supersedes`
    pointing to the old one, forming a version chain.
    """

    id: str = field(default_factory=_new_id)
    content: str = ""
    type: MemoryType = MemoryType.FACT
    importance: float = 0.5          # 0.0-1.0, multi-signal auto-scored
    confidence: float = 0.8          # 0.0-1.0, decays over time
    created_at: datetime = field(default_factory=_utcnow)
    accessed_at: datetime = field(default_factory=_utcnow)
    access_count: int = 0
    source: str = ""                 # e.g. "conversation:2026-04-25"
    entities: list[str] = field(default_factory=list)  # entity IDs
    supersedes: Optional[str] = None # ID of the memory this replaces
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None  # vector embedding
    layer: str = "L1"                # L0 (hot), L1 (active), L2 (archive)

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = _utcnow()
        self.access_count += 1

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "source": self.source,
            "entities": self.entities,
            "supersedes": self.supersedes,
            "tags": self.tags,
            "layer": self.layer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryUnit:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            type=MemoryType(data["type"]),
            importance=data["importance"],
            confidence=data["confidence"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data["access_count"],
            source=data["source"],
            entities=data.get("entities", []),
            supersedes=data.get("supersedes"),
            tags=data.get("tags", []),
            layer=data.get("layer", "L1"),
        )


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    type: EntityType = EntityType.CONCEPT
    aliases: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=_utcnow)
    last_seen: datetime = field(default_factory=_utcnow)
    mention_count: int = 1

    def touch(self) -> None:
        self.last_seen = _utcnow()
        self.mention_count += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "aliases": self.aliases,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "mention_count": self.mention_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Entity:
        return cls(
            id=data["id"],
            name=data["name"],
            type=EntityType(data["type"]),
            aliases=data.get("aliases", []),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            mention_count=data.get("mention_count", 1),
        )


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_entity_id: str = ""
    target_entity_id: str = ""
    type: str = ""                   # "prefers", "works_on", "knows", etc.
    weight: float = 1.0
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict:
        return {
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "type": self.type,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relation:
        return cls(
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            type=data["type"],
            weight=data["weight"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )
