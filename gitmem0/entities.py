"""Entity knowledge graph manager for GitMem0.

Extracts entities and relations from text using rule-based pattern matching,
provides graph traversal operations, and links memories to entities.
"""

from __future__ import annotations

import re
from collections import deque

from gitmem0.models import Entity, EntityType, MemoryUnit, Relation
from gitmem0.store import MemoryStore

# ── Built-in technology vocabulary ─────────────────────────────────────

TECH_VOCABULARY: set[str] = {
    # Languages
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++",
    "C#", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Haskell", "Lua",
    "Perl", "R", "Dart", "Elixir", "Clojure", "Erlang", "OCaml",
    "Zig", "Julia", "Groovy", "Assembly",
    # Frameworks & Libraries
    "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "Django",
    "Flask", "FastAPI", "Express", "NestJS", "Spring", "Rails", "Laravel",
    "Gin", "Actix", "Tokio", "jQuery", "Bootstrap", "Tailwind",
    "TensorFlow", "PyTorch", "NumPy", "Pandas", "scikit-learn",
    # Tools & Platforms
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitLab",
    "GitHub", "Bitbucket", "Jira", "Confluence", "VS Code", "Vim",
    "Neovim", "Emacs", "IntelliJ", "PyCharm", "Webpack", "Vite",
    "esbuild", "Rollup", "Babel", "npm", "yarn", "pnpm", "pip",
    "Cargo", "Homebrew", "apt",
    # Databases & Infrastructure
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "SQLite", "DynamoDB", "Cassandra", "Kafka", "RabbitMQ", "Nginx",
    "Apache", "Caddy", "HAProxy", "gRPC", "GraphQL", "REST",
    # Cloud & Services
    "AWS", "GCP", "Azure", "Cloudflare", "Vercel", "Netlify",
    "Heroku", "DigitalOcean", "Stripe", "Twilio", "SendGrid",
    # Concepts
    "Linux", "Git", "CI/CD", "DevOps", "Microservices", "API",
    "OAuth", "JWT", "WebSocket", "HTTP", "TCP", "UDP", "DNS", "SSL", "TLS",
}

# Lowercase lookup for case-insensitive matching
_TECH_LOOKUP: dict[str, str] = {t.lower(): t for t in TECH_VOCABULARY}

# ── Relation extraction patterns ───────────────────────────────────────

_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # English: "X prefers/likes/uses/needs Y"
    (re.compile(
        r"(\w+)\s+(?:prefers?|likes?|uses?|needs?|loves?|wants?)\s+(\w[\w.+#-]*)",
        re.IGNORECASE,
    ), "prefers"),
    # English: "X works on/develops/maintains/builds Y"
    (re.compile(
        r"(\w+)\s+(?:works?\s+on|develops?|maintains?|builds?|created?|wrote)\s+(\w[\w.+#-]*)",
        re.IGNORECASE), "works_on"),
    # English: "X knows/learned/studied Y"
    (re.compile(
        r"(\w+)\s+(?:knows?|learned?|studied?|understands?|familiar\s+with)\s+(\w[\w.+#-]*)",
        re.IGNORECASE), "knows"),
    # English: "X is a Y"
    (re.compile(
        r"(\w+)\s+(?:is\s+an?|was\s+an?)\s+(\w[\w.+#-]*)",
        re.IGNORECASE), "is_a"),
    # English: "using X for/to Y", "with X"
    (re.compile(
        r"(?:using|use|used)\s+(\w[\w.+#-]*)\s+(?:for|to)\s+(\w[\w.+#-]*)",
        re.IGNORECASE), "uses"),
    # Chinese: "用X写/做/开发Y"
    (re.compile(
        r"用\s*([A-Za-z][A-Za-z0-9.+#-]*)\s*(?:写|做|开发|构建|搭建|重写)\s*([A-Za-z][A-Za-z0-9.+#-]*)",
    ), "uses"),
    # Chinese: "学习/学X"
    (re.compile(
        r"(?:学习|学|在学|正在学)\s*([A-Za-z][A-Za-z0-9.+#-]*)",
    ), "learning"),
    # Chinese: "用X" (standalone "using X")
    (re.compile(
        r"(?:用|使用|采用)\s*([A-Za-z][A-Za-z0-9.+#-]*)",
    ), "uses"),
    # Chinese: "喜欢/偏好X"
    (re.compile(
        r"(?:喜欢|偏好|爱用|常用|习惯用)\s*([A-Za-z][A-Za-z0-9.+#-]*)",
    ), "prefers"),
    # Chinese: "X后端/前端/项目"
    (re.compile(
        r"([A-Za-z][A-Za-z0-9.+#-]*)\s*(?:后端|前端|项目|框架|库|工具|平台|数据库)",
    ), "related_to"),
]


class EntityManager:
    """Manages entity extraction, relation extraction, and graph operations."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # ── Entity extraction ──────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text using pattern matching.

        Checks the store for existing entities (by name) and updates
        last_seen / mention_count if found, otherwise creates new ones.
        """
        found: dict[str, tuple[Entity, EntityType]] = {}

        # 1. Technology detection (case-insensitive against known vocab)
        # Use [A-Za-z0-9] instead of \w to avoid matching Chinese chars
        for word in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", text):
            canonical = _TECH_LOOKUP.get(word.lower())
            if canonical and canonical.lower() not in found:
                found[canonical.lower()] = (None, EntityType.TECHNOLOGY)
        # 2. Project detection: words after "project" or in quotes
        for m in re.finditer(r'(?:project\s+)(\w[\w-]*)', text, re.IGNORECASE):
            name = m.group(1)
            key = name.lower()
            if key not in found:
                found[key] = (None, EntityType.PROJECT)
        for m in re.finditer(r'"([^"]+)"', text):
            name = m.group(1).strip()
            if name and len(name) > 1:
                key = name.lower()
                if key not in found:
                    found[key] = (None, EntityType.CONCEPT)
        for m in re.finditer(r"'([^']+)'", text):
            name = m.group(1).strip()
            if name and len(name) > 1:
                key = name.lower()
                if key not in found:
                    found[key] = (None, EntityType.CONCEPT)
        # 3. Person detection: capitalized words (not sentence starts)
        words = text.split()
        for i, word in enumerate(words):
            cleaned = re.sub(r"[^A-Za-z]", "", word)
            if not cleaned:
                continue
            # Skip if it's a known tech word
            if cleaned.lower() in _TECH_LOOKUP:
                continue
            # Heuristic: capitalized word that isn't first in its sentence
            if cleaned[0].isupper() and len(cleaned) > 1 and cleaned[1:].islower():
                # Check it's not the first word after a period
                is_sentence_start = False
                if i > 0 and words[i - 1].endswith((".", "!", "?")):
                    is_sentence_start = True
                if not is_sentence_start and cleaned.lower() not in found:
                    # Only mark as person if it looks name-like (not common words)
                    if cleaned not in _COMMON_WORDS:
                        found[cleaned.lower()] = (None, EntityType.PERSON)
        # 4. Resolve against store: update existing or create new
        entities: list[Entity] = []
        for key, (_, etype) in found.items():
            # Try to find by exact name match (case-insensitive via the key)
            existing = self._find_entity_by_key(key)
            if existing:
                existing.touch()
                self._store.update_entity(existing)
                entities.append(existing)
            else:
                # Recover original casing from the key
                name = _TECH_LOOKUP.get(key, key.title())
                entity = Entity(name=name, type=etype)
                self._store.add_entity(entity)
                entities.append(entity)

        return entities

    def _find_entity_by_key(self, key: str) -> Entity | None:
        """Look up an entity by name key (case-insensitive). O(1) via name index."""
        entity = self._store.get_entity_by_name(key.title())
        if entity:
            return entity
        # Check tech vocabulary canonical form
        canonical = _TECH_LOOKUP.get(key)
        if canonical:
            entity = self._store.get_entity_by_name(canonical)
            if entity:
                return entity
        # Direct lowercase lookup — name index is already case-insensitive
        entity = self._store.get_entity_by_name(key)
        if entity:
            return entity
        return None

    # ── Relation extraction ────────────────────────────────────────────

    def extract_relations(
        self, text: str, entities: list[Entity]
    ) -> list[Relation]:
        """Extract relations from text using pattern matching."""
        # Build a name lookup (lowercase -> entity)
        name_lookup: dict[str, Entity] = {}
        for e in entities:
            name_lookup[e.name.lower()] = e
            for alias in e.aliases:
                name_lookup[alias.lower()] = e

        relations: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()

        for pattern, rel_type in _RELATION_PATTERNS:
            for m in pattern.finditer(text):
                # For patterns with 2 groups (subj + obj)
                if m.lastindex and m.lastindex >= 2:
                    subj_text = m.group(1).strip().rstrip(".,;:!?")
                    obj_text = m.group(2).strip().rstrip(".,;:!?")
                    subj = name_lookup.get(subj_text.lower())
                    obj = name_lookup.get(obj_text.lower())
                    if subj and obj and subj.id != obj.id:
                        key = (subj.id, obj.id, rel_type)
                        if key not in seen:
                            seen.add(key)
                            relations.append(Relation(
                                source_entity_id=subj.id,
                                target_entity_id=obj.id,
                                type=rel_type,
                            ))
                # For patterns with 1 group (e.g., "学习X", "用X")
                elif m.lastindex and m.lastindex >= 1:
                    obj_text = m.group(1).strip().rstrip(".,;:!?")
                    obj = name_lookup.get(obj_text.lower())
                    if obj:
                        # Link to all other entities in the same text
                        for other in entities:
                            if other.id != obj.id:
                                key = (other.id, obj.id, rel_type)
                                if key not in seen:
                                    seen.add(key)
                                    relations.append(Relation(
                                        source_entity_id=other.id,
                                        target_entity_id=obj.id,
                                        type=rel_type,
                                    ))

        # Co-occurrence: entities in the same segment are related
        if len(entities) >= 2:
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    key = (e1.id, e2.id, "co_occurs")
                    if key not in seen:
                        seen.add(key)
                        relations.append(Relation(
                            source_entity_id=e1.id,
                            target_entity_id=e2.id,
                            type="co_occurs",
                            weight=0.5,
                        ))

        return relations

    # ── Graph operations ───────────────────────────────────────────────

    def get_entity_neighbors(
        self, entity_id: str, depth: int = 1
    ) -> dict[str, dict]:
        """BFS traversal up to *depth* hops from *entity_id*.

        Returns ``{entity_id: {"entity": Entity, "relations": [Relation], "depth": int}}``.
        """
        visited: dict[str, dict] = {}
        queue: deque[tuple[str, int]] = deque()
        queue.append((entity_id, 0))
        visited[entity_id] = {"entity": None, "relations": [], "depth": 0}

        while queue:
            current_id, current_depth = queue.popleft()
            entity = self._store.get_entity(current_id)
            if entity is None:
                continue
            visited[current_id]["entity"] = entity

            if current_depth >= depth:
                continue

            rels = self._store.get_relations(current_id)
            visited[current_id]["relations"] = rels

            for rel in rels:
                neighbor_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == current_id
                    else rel.source_entity_id
                )
                if neighbor_id not in visited:
                    visited[neighbor_id] = {
                        "entity": None,
                        "relations": [],
                        "depth": current_depth + 1,
                    }
                    queue.append((neighbor_id, current_depth + 1))

        # Clean up root entry
        if entity_id in visited:
            visited[entity_id]["relations"] = self._store.get_relations(entity_id)

        return visited

    def get_entity_context(self, entity_name: str) -> str:
        """Return a human-readable summary of an entity and its relationships."""
        entity = self._store.get_entity_by_name(entity_name)
        if entity is None:
            return f"Entity '{entity_name}' not found."

        rels = self._store.get_relations(entity.id)
        parts: list[str] = []

        for rel in rels:
            if rel.source_entity_id == entity.id:
                other = self._store.get_entity(rel.target_entity_id)
            else:
                other = self._store.get_entity(rel.source_entity_id)
            if other is None:
                continue

            if rel.source_entity_id == entity.id:
                parts.append(f"{rel.type} {other.name}")
            else:
                parts.append(f"{rel.type} by {other.name}")

        rel_summary = ", ".join(parts) if parts else "no known relationships"
        return (
            f"{entity.name} ({entity.type.value}, "
            f"mentioned {entity.mention_count} times): {rel_summary}"
        )

    def find_path(
        self, entity_id1: str, entity_id2: str, max_depth: int = 3
    ) -> list[str] | None:
        """BFS shortest path between two entities.

        Returns list of entity IDs forming the path, or None if unreachable.
        """
        if entity_id1 == entity_id2:
            return [entity_id1]

        visited: set[str] = {entity_id1}
        queue: deque[list[str]] = deque()
        queue.append([entity_id1])

        while queue:
            path = queue.popleft()
            current_id = path[-1]

            if len(path) > max_depth + 1:
                continue

            rels = self._store.get_relations(current_id)
            for rel in rels:
                neighbor_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == current_id
                    else rel.source_entity_id
                )
                if neighbor_id == entity_id2:
                    return path + [neighbor_id]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(path + [neighbor_id])

        return None

    # ── Memory-entity linking ──────────────────────────────────────────

    def link_memory_entities(self, memory: MemoryUnit) -> MemoryUnit:
        """Extract entities and relations from a memory's content.

        Stores entities/relations in the store and updates
        ``memory.entities`` with entity IDs. Returns the updated memory.
        The caller is responsible for persisting the memory to the store.
        """
        entities = self.extract_entities(memory.content)
        relations = self.extract_relations(memory.content, entities)

        # Store relations
        for rel in relations:
            self._store.add_relation(rel)

        # Update memory's entity list (union with existing)
        existing_ids = set(memory.entities)
        for entity in entities:
            if entity.id not in existing_ids:
                memory.entities.append(entity.id)
                existing_ids.add(entity.id)

        return memory


# ── Helpers ────────────────────────────────────────────────────────────

_COMMON_WORDS: set[str] = {
    "The", "This", "That", "These", "Those", "There", "Then", "Their",
    "They", "What", "When", "Where", "Which", "While", "Who", "Why",
    "How", "All", "Each", "Every", "Both", "Few", "More", "Most",
    "Other", "Some", "Such", "Than", "Too", "Very", "Just", "Also",
    "Still", "Already", "Always", "Never", "Sometimes", "Often",
    "Here", "Now", "Today", "Tomorrow", "Yesterday", "Yes", "No",
    "Not", "But", "And", "Or", "Nor", "For", "Yet", "So", "If",
    "Because", "Since", "Although", "Though", "However", "Therefore",
    "Moreover", "Furthermore", "Nevertheless", "Instead", "Otherwise",
    "According", "Actually", "Basically", "Certainly", "Clearly",
    "Certainly", "Definitely", "Exactly", "Finally", "Fortunately",
    "Generally", "Hopefully", "Ideally", "Importantly", "Indeed",
    "Likewise", "Meanwhile", "Naturally", "Obviously", "Particularly",
    "Perhaps", "Possibly", "Previously", "Probably", "Really",
    "Regarding", "Similarly", "Specifically", "Subsequently", "Surely",
    "Ultimately", "Unfortunately", "Usually", "Without", "Within",
    "After", "Before", "During", "Between", "Above", "Below", "Under",
    "Over", "Through", "Across", "Along", "Around", "Against", "Toward",
    "Towards", "Upon", "Among", "Into", "Onto", "From", "With",
    "About", "Like", "Have", "Has", "Had", "Do", "Does", "Did",
    "Will", "Would", "Could", "Should", "May", "Might", "Must",
    "Can", "Need", "Dare", "Ought", "Shall", "Am", "Is", "Are",
    "Was", "Were", "Been", "Being", "Get", "Gets", "Got", "Gotten",
    "Make", "Makes", "Made", "Take", "Takes", "Took", "Taken",
    "Come", "Comes", "Came", "Give", "Gives", "Gave", "Given",
    "Know", "Knows", "Knew", "Think", "Thinks", "Thought",
    "Say", "Says", "Said", "Tell", "Tells", "Told",
    "See", "Sees", "Saw", "Seen", "Want", "Wants", "Wanted",
    "Use", "Uses", "Used", "Find", "Finds", "Found",
    "Work", "Works", "Worked", "Working", "Call", "Calls", "Called",
    "Try", "Tries", "Tried", "Keep", "Keeps", "Kept",
    "Let", "Lets", "Put", "Puts", "Mean", "Means", "Meant",
    "Become", "Becomes", "Became", "Leave", "Leaves", "Left",
    "Need", "Needs", "Needed", "Seem", "Seems", "Seemed",
    "Help", "Helps", "Helped", "Show", "Shows", "Showed", "Shown",
    "Hear", "Hears", "Heard", "Play", "Plays", "Played",
    "Run", "Runs", "Ran", "Move", "Moves", "Moved",
    "Live", "Lives", "Lived", "Believe", "Believes", "Believed",
    "Hold", "Holds", "Held", "Bring", "Brings", "Brought",
    "Happen", "Happens", "Happened", "Write", "Writes", "Wrote", "Written",
    "Provide", "Provides", "Provided", "Sit", "Sits", "Sat",
    "Stand", "Stands", "Stood", "Lose", "Loses", "Lost",
    "Pay", "Pays", "Paid", "Meet", "Meets", "Met",
    "Include", "Includes", "Included", "Continue", "Continues", "Continued",
    "Set", "Sets", "Learn", "Learns", "Learned", "Learning",
    "Change", "Changes", "Changed", "Lead", "Leads", "Led",
    "Understand", "Understands", "Understood", "Watch", "Watches", "Watched",
    "Follow", "Follows", "Followed", "Stop", "Stops", "Stopped",
    "Create", "Creates", "Created", "Speak", "Speaks", "Spoke", "Spoken",
    "Read", "Reads", "Allow", "Allows", "Allowed",
    "Add", "Adds", "Added", "Spend", "Spends", "Spent",
    "Grow", "Grows", "Grew", "Grown", "Open", "Opens", "Opened",
    "Walk", "Walks", "Walked", "Win", "Wins", "Won",
    "Offer", "Offers", "Offered", "Remember", "Remembers", "Remembered",
    "Love", "Loves", "Loved", "Consider", "Considers", "Considered",
    "Appear", "Appears", "Appeared", "Buy", "Bought",
    "Wait", "Waits", "Waited", "Serve", "Serves", "Served",
    "Die", "Dies", "Died", "Send", "Sends", "Sent",
    "Expect", "Expects", "Expected", "Build", "Builds", "Built",
    "Stay", "Stays", "Stayed", "Fall", "Falls", "Fell", "Fallen",
    "Cut", "Cuts", "Reach", "Reaches", "Reached",
    "Kill", "Kills", "Killed", "Remain", "Remains", "Remained",
    "Suggest", "Suggests", "Suggested", "Raise", "Raises", "Raised",
    "Pass", "Passes", "Passed", "Sell", "Sells", "Sold",
    "Require", "Requires", "Required", "Report", "Reports", "Reported",
    "Decide", "Decides", "Decided", "Pull", "Pulls", "Pulled",
}
