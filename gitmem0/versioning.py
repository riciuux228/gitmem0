"""Version control for GitMem0 memory units.

Provides version chains (like git branches), branching, cherry-picking,
and diffing between memory versions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from gitmem0.models import MemoryType, MemoryUnit, _new_id
from gitmem0.store import MemoryStore


class VersionControl:
    """Version control operations over a MemoryStore."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # ── Version chain operations ───────────────────────────────────

    def create_version(
        self,
        content: str,
        type: MemoryType,
        importance: float,
        confidence: float,
        source: str,
        entities: list[str],
        tags: list[str],
        supersedes_id: str,
    ) -> MemoryUnit:
        """Create a new MemoryUnit that supersedes the given one.

        Raises ValueError if *supersedes_id* does not exist in the store.
        """
        existing = self._store.get_memory(supersedes_id)
        if existing is None:
            raise ValueError(f"Memory {supersedes_id!r} not found")

        unit = MemoryUnit(
            content=content,
            type=type,
            importance=importance,
            confidence=confidence,
            source=source,
            entities=list(entities),
            tags=list(tags),
            supersedes=supersedes_id,
        )
        self._store.add_memory(unit)
        return unit

    def get_history(self, memory_id: str) -> list[MemoryUnit]:
        """Follow the *supersedes* chain backwards to build full history.

        Returns the list ordered newest-first (the requested id first).
        """
        history: list[MemoryUnit] = []
        current_id: Optional[str] = memory_id
        visited: set[str] = set()
        while current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)
            unit = self._store.get_memory(current_id)
            if unit is None:
                break
            history.append(unit)
            current_id = unit.supersedes
        return history

    def get_current(self, memory_id: str) -> MemoryUnit:
        """Given any ID in a version chain, find the latest version.

        Searches all memories whose supersedes chain includes *memory_id*.
        """
        # If the memory itself doesn't exist, fail fast.
        start = self._store.get_memory(memory_id)
        if start is None:
            raise ValueError(f"Memory {memory_id!r} not found")

        # Build a reverse index: supersedes_id -> list of memories that point to it.
        # Then walk forward from memory_id.
        current = start
        all_memories = self._store.list_memories(limit=999_999)
        forward_map: dict[str, list[MemoryUnit]] = {}
        for m in all_memories:
            if m.supersedes is not None:
                forward_map.setdefault(m.supersedes, []).append(m)

        # Follow forward links (there can be branching, take the newest).
        queue = [current]
        while queue:
            node = queue.pop()
            children = forward_map.get(node.id, [])
            if not children:
                # This is the leaf / current version.
                current = node
                continue
            # Pick the most recent child and continue.
            children.sort(key=lambda m: m.created_at, reverse=True)
            current = children[0]
            queue.append(current)

        return current

    def get_lineage(self, memory_id: str) -> list[str]:
        """Return all IDs in the version chain, newest first.

        Resolves to the current (newest) version first, then walks back.
        """
        current = self.get_current(memory_id)
        return [m.id for m in self.get_history(current.id)]

    # ── Branch operations ──────────────────────────────────────────

    def create_branch(self, memory_id: str, branch_name: str) -> str:
        """Create a copy of the memory with a new ID and a ``branch:<name>`` tag.

        Returns the new memory ID.
        """
        original = self._store.get_memory(memory_id)
        if original is None:
            raise ValueError(f"Memory {memory_id!r} not found")

        branch_tag = f"branch:{branch_name}"
        new_tags = list(original.tags)
        if branch_tag not in new_tags:
            new_tags.append(branch_tag)

        clone = MemoryUnit(
            content=original.content,
            type=original.type,
            importance=original.importance,
            confidence=original.confidence,
            source=original.source,
            entities=list(original.entities),
            supersedes=original.supersedes,
            tags=new_tags,
        )
        self._store.add_memory(clone)
        return clone.id

    def get_branch(self, branch_name: str) -> list[MemoryUnit]:
        """Return all memories tagged with ``branch:<name>``."""
        tag = f"branch:{branch_name}"
        return [m for m in self._store.list_memories(limit=999_999) if tag in m.tags]

    def cherry_pick(self, memory_id: str, target_branch: str) -> MemoryUnit:
        """Copy a memory into the target branch (adds the branch tag).

        Returns the new memory unit.
        """
        original = self._store.get_memory(memory_id)
        if original is None:
            raise ValueError(f"Memory {memory_id!r} not found")

        branch_tag = f"branch:{target_branch}"
        new_tags = list(original.tags)
        if branch_tag not in new_tags:
            new_tags.append(branch_tag)

        clone = MemoryUnit(
            content=original.content,
            type=original.type,
            importance=original.importance,
            confidence=original.confidence,
            source=original.source,
            entities=list(original.entities),
            supersedes=original.supersedes,
            tags=new_tags,
        )
        self._store.add_memory(clone)
        return clone

    # ── Diff ───────────────────────────────────────────────────────

    def diff(self, id1: str, id2: str) -> dict:
        """Return a diff between two memory versions.

        The result dict has keys:
        - ``content``: ``{"a": ..., "b": ...}``
        - ``type``: ``{"a": ..., "b": ...}``  (only if different)
        - ``importance``: ``{"a": ..., "b": ...}``  (only if different)
        - ``confidence``: ``{"a": ..., "b": ...}``  (only if different)
        - ``entities``: ``{"added": [...], "removed": [...]}``
        - ``tags``: ``{"added": [...], "removed": [...]}``
        """
        a = self._store.get_memory(id1)
        b = self._store.get_memory(id2)
        if a is None:
            raise ValueError(f"Memory {id1!r} not found")
        if b is None:
            raise ValueError(f"Memory {id2!r} not found")

        result: dict = {}

        if a.content != b.content:
            result["content"] = {"a": a.content, "b": b.content}

        if a.type != b.type:
            result["type"] = {"a": a.type.value, "b": b.type.value}

        if a.importance != b.importance:
            result["importance"] = {"a": a.importance, "b": b.importance}

        if a.confidence != b.confidence:
            result["confidence"] = {"a": a.confidence, "b": b.confidence}

        set_a_entities = set(a.entities)
        set_b_entities = set(b.entities)
        if set_a_entities != set_b_entities:
            result["entities"] = {
                "added": sorted(set_b_entities - set_a_entities),
                "removed": sorted(set_a_entities - set_b_entities),
            }

        set_a_tags = set(a.tags)
        set_b_tags = set(b.tags)
        if set_a_tags != set_b_tags:
            result["tags"] = {
                "added": sorted(set_b_tags - set_a_tags),
                "removed": sorted(set_a_tags - set_b_tags),
            }

        return result
