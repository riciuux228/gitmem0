"""GitMem0 CLI — AI-first command-line interface.

All commands output JSON by default (machine-readable, minimal tokens).
Use --format text for human debugging.
No interactive prompts — safe for AI to call via bash.
"""

from __future__ import annotations

import json
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Fix Windows terminal encoding: force UTF-8 output (only when running in terminal)
if sys.platform == "win32" and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer

from gitmem0.context import ContextBuilder
from gitmem0.decay import DecayEngine
from gitmem0.embeddings import EmbeddingEngine
from gitmem0.entities import EntityManager
from gitmem0.extraction import ExtractionEngine
from gitmem0.models import EntityType, MemoryType, MemoryUnit
from gitmem0.retrieval import RetrievalEngine
from gitmem0.store import MemoryStore
from gitmem0.versioning import VersionControl

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = Path.home() / ".gitmem0" / "gitmem0.db"
_db_override: Optional[str] = None

# ── Shared context ───────────────────────────────────────────────────────────


@dataclass
class GitMem0Context:
    store: MemoryStore
    embeddings: EmbeddingEngine
    retrieval: RetrievalEngine
    versioning: VersionControl
    decay: DecayEngine
    entities: EntityManager
    extraction: ExtractionEngine
    context_builder: ContextBuilder


# ── App setup ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="gitmem0",
    help="GitMem0: AI-first version-controlled memory system.",
    no_args_is_help=True,
)


def _load_config() -> dict:
    if _db_override is not None:
        return {"db_path": _db_override}
    config = {"db_path": str(_DEFAULT_DB_PATH)}
    config_path = Path.home() / ".gitmem0" / "config.toml"
    if config_path.exists():
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            with open(config_path, "rb") as f:
                loaded = tomllib.load(f)
            if "db_path" in loaded:
                config["db_path"] = loaded["db_path"]
    return config


def _build_ctx() -> GitMem0Context:
    cfg = _load_config()
    db_path = Path(cfg["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db_path)
    embeddings = EmbeddingEngine()
    return GitMem0Context(
        store=store,
        embeddings=embeddings,
        retrieval=RetrievalEngine(store, embeddings),
        versioning=VersionControl(store),
        decay=DecayEngine(store, embeddings),
        entities=EntityManager(store),
        extraction=ExtractionEngine(store, embeddings, EntityManager(store)),
        context_builder=ContextBuilder(
            RetrievalEngine(store, embeddings), EntityManager(store)
        ),
    )


def _get_ctx() -> GitMem0Context:
    if not hasattr(_get_ctx, "_ctx"):
        _get_ctx._ctx = _build_ctx()  # type: ignore[attr-defined]
    return _get_ctx._ctx  # type: ignore[attr-defined]


@app.callback()
def main(
    db: Optional[str] = typer.Option(None, "--db", help="Database path."),
) -> None:
    """GitMem0 — AI-first memory system."""
    global _db_override
    if db is not None:
        _db_override = db
        if hasattr(_get_ctx, "_ctx"):
            delattr(_get_ctx, "_ctx")


# ── Output helpers ────────────────────────────────────────────────────────────


def _ok(data=None) -> None:
    """Print success response and exit."""
    out = {"ok": True}
    if data is not None:
        out["data"] = data
    print(json.dumps(out, ensure_ascii=False, default=str))


def _err(msg: str) -> None:
    """Print error response and exit."""
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    raise typer.Exit(code=1)


def _mem_json(m: MemoryUnit) -> dict:
    """Minimal memory representation for JSON output."""
    return {
        "id": m.id,
        "content": m.content,
        "type": m.type.value,
        "imp": round(m.importance, 3),
        "conf": round(m.confidence, 3),
        "layer": m.layer,
        "supersedes": m.supersedes,
        "entities": m.entities,
        "tags": m.tags,
    }


def _entity_json(e) -> dict:
    return {
        "id": e.id,
        "name": e.name,
        "type": e.type.value,
        "mentions": e.mention_count,
    }


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command()
def add(
    content: str = typer.Argument(..., help="Memory content."),
    type: Optional[str] = typer.Option(None, "--type", "-t"),
    importance: float = typer.Option(0.5, "--importance", "-i"),
    source: str = typer.Option("cli", "--source", "-s"),
    tags: Optional[str] = typer.Option(None, "--tags"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="json (default) or text."),
) -> None:
    """Add a memory. Returns {ok, data: {id, content, type, imp, conf, entities}}."""
    ctx = _get_ctx()
    mem_type = MemoryType.FACT
    if type is not None:
        try:
            mem_type = MemoryType(type.lower())
        except ValueError:
            _err(f"Invalid type '{type}'. Use: {', '.join(t.value for t in MemoryType)}")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    embedding = ctx.embeddings.embed(content)

    unit = MemoryUnit(
        content=content,
        type=mem_type,
        importance=importance,
        source=source,
        tags=tag_list,
        embedding=embedding,
    )
    unit = ctx.entities.link_memory_entities(unit)
    ctx.store.add_memory(unit)

    if fmt == "text":
        print(f"OK {unit.id} type={mem_type.value} imp={importance:.2f} entities={len(unit.entities)}")
    else:
        _ok({"id": unit.id, "content": content, "type": mem_type.value,
             "imp": importance, "entities": unit.entities})


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    top: int = typer.Option(5, "--top", "-n"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="json (default) or text."),
) -> None:
    """Search memories. Returns {ok, data: [{id, content, type, imp, conf, ...}]}."""
    ctx = _get_ctx()
    try:
        results = ctx.retrieval.search(query, top_n=top)
    except Exception as e:
        _err(str(e))

    if not results:
        if fmt == "text":
            print("(no results)")
        else:
            _ok([])
        return

    if fmt == "text":
        for m in results:
            print(f"{m.id} [{m.type.value}] imp={m.importance:.2f} conf={m.confidence:.2f}")
            print(f"  {m.content}")
    else:
        _ok([_mem_json(m) for m in results])


@app.command("context")
def context_cmd(
    topic: str = typer.Argument(..., help="Topic / query."),
    budget: int = typer.Option(2000, "--budget", "-b"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="json (default) or text."),
) -> None:
    """Build optimized context for LLM injection. Returns {ok, data: {context, stats}}."""
    ctx = _get_ctx()
    try:
        result = ctx.context_builder.build_context(topic, token_budget=budget)
        stats = ctx.context_builder.context_stats(topic)
    except Exception as e:
        _err(str(e))

    if not result:
        if fmt == "text":
            print("(no context)")
        else:
            _ok({"context": "", "stats": stats})
        return

    if fmt == "text":
        print(result)
    else:
        _ok({"context": result, "stats": stats})


@app.command()
def history(
    memory_id: str = typer.Argument(..., help="Memory ID."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Show version chain. Returns {ok, data: [{id, content, type, supersedes, ...}]}."""
    ctx = _get_ctx()
    try:
        chain = ctx.versioning.get_history(memory_id)
    except Exception as e:
        _err(str(e))

    if not chain:
        _err(f"Memory '{memory_id}' not found.")

    if fmt == "text":
        for i, m in enumerate(chain):
            sup = f" <- {m.supersedes}" if m.supersedes else ""
            print(f"  [{i}] {m.id} {m.type.value}: {m.content}{sup}")
    else:
        _ok([_mem_json(m) for m in chain])


@app.command()
def diff(
    id1: str = typer.Argument(..., help="First memory ID."),
    id2: str = typer.Argument(..., help="Second memory ID."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Diff two memory versions. Returns {ok, data: {field: {a, b}, ...}}."""
    ctx = _get_ctx()
    try:
        result = ctx.versioning.diff(id1, id2)
    except ValueError as e:
        _err(str(e))

    if not result:
        if fmt == "text":
            print("(no differences)")
        else:
            _ok({})
        return

    if fmt == "text":
        for field, vals in result.items():
            if isinstance(vals, dict) and "a" in vals:
                print(f"  {field}: {vals['a']} -> {vals['b']}")
            elif isinstance(vals, dict) and "added" in vals:
                print(f"  {field}: +{vals['added']} -{vals['removed']}")
    else:
        _ok(result)


@app.command()
def entities(
    type: Optional[str] = typer.Option(None, "--type", "-t"),
    limit: int = typer.Option(20, "--limit", "-n"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """List entities. Returns {ok, data: [{id, name, type, mentions}]}."""
    ctx = _get_ctx()
    entity_type = None
    if type is not None:
        try:
            entity_type = EntityType(type.lower())
        except ValueError:
            _err(f"Invalid type '{type}'. Use: {', '.join(t.value for t in EntityType)}")

    results = ctx.store.list_entities(type=entity_type, limit=limit)

    if not results:
        if fmt == "text":
            print("(no entities)")
        else:
            _ok([])
        return

    if fmt == "text":
        for e in results:
            print(f"  {e.id} {e.name} [{e.type.value}] x{e.mention_count}")
    else:
        _ok([_entity_json(e) for e in results])


@app.command()
def relations(
    entity_name: str = typer.Argument(..., help="Entity name."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Show relations for an entity. Returns {ok, data: [{direction, type, other, weight}]}."""
    ctx = _get_ctx()
    entity = ctx.store.get_entity_by_name(entity_name)
    if entity is None:
        _err(f"Entity '{entity_name}' not found.")

    rels = ctx.store.get_relations(entity.id)
    if not rels:
        if fmt == "text":
            print(f"(no relations for {entity_name})")
        else:
            _ok([])
        return

    data = []
    for rel in rels:
        if rel.source_entity_id == entity.id:
            direction = "out"
            other = ctx.store.get_entity(rel.target_entity_id)
        else:
            direction = "in"
            other = ctx.store.get_entity(rel.source_entity_id)
        other_name = other.name if other else "?"
        data.append({"dir": direction, "type": rel.type, "other": other_name, "w": rel.weight})

    if fmt == "text":
        for r in data:
            print(f"  {r['dir']} {r['type']} {r['other']} (w={r['w']:.1f})")
    else:
        _ok(data)


@app.command()
def stats(
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Memory store statistics. Returns {ok, data: {memories, layers, entities, relations}}."""
    ctx = _get_ctx()
    data = ctx.store.stats()

    if fmt == "text":
        print(f"memories={data['total_memories']} entities={data['total_entities']} relations={data['total_relations']}")
        for layer, count in sorted(data["layers"].items()):
            print(f"  {layer}: {count}")
    else:
        _ok(data)


@app.command()
def decay(
    dry_run: bool = typer.Option(False, "--dry-run"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Run decay process. Returns {ok, data: {decayed, archived, cleaned}}."""
    ctx = _get_ctx()

    if dry_run:
        data = ctx.decay.get_layer_stats()
        if fmt == "text":
            for layer, info in data.items():
                print(f"  {layer}: count={info['count']} avg_conf={info['avg_confidence']:.3f}")
        else:
            _ok(data)
        return

    result = ctx.decay.apply_decay()
    if fmt == "text":
        print(f"decayed={result['decayed']} archived={result['archived']} cleaned={result['cleaned']}")
    else:
        _ok(result)


@app.command()
def consolidate(
    dry_run: bool = typer.Option(False, "--dry-run"),
    threshold: float = typer.Option(0.85, "--threshold", "-t"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Run consolidation. Returns {ok, data: {groups_found, consolidated}}."""
    ctx = _get_ctx()
    result = ctx.decay.run_consolidation(threshold=threshold, dry_run=dry_run)

    if fmt == "text":
        print(f"groups={result['groups_found']} consolidated={result['consolidated']}")
    else:
        _ok(result)


@app.command()
def contradictions(
    dry_run: bool = typer.Option(False, "--dry-run"),
    threshold: float = typer.Option(0.7, "--threshold", "-t"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Find and resolve contradicting memories. Returns {ok, data: {found, resolved}}."""
    ctx = _get_ctx()
    if dry_run:
        pairs = ctx.decay.detect_contradictions(threshold=threshold)
        data = [{"id_a": a, "id_b": b, "reason": r} for a, b, r in pairs]
        if fmt == "text":
            if not data:
                print("(no contradictions)")
            for d in data:
                print(f"  {d['id_a']} <-> {d['id_b']}: {d['reason']}")
        else:
            _ok({"found": len(data), "pairs": data})
    else:
        result = ctx.decay.resolve_contradictions()
        if fmt == "text":
            print(f"found={result['found']} resolved={result['resolved']}")
        else:
            _ok(result)


@app.command(name="auto-induct")
def auto_induct(
    dry_run: bool = typer.Option(False, "--dry-run"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Auto-induct events into insights. Returns {ok, data: {groups, inducted}}."""
    ctx = _get_ctx()
    result = ctx.decay.auto_induct(dry_run=dry_run)
    if fmt == "text":
        print(f"groups={result['groups']} inducted={result['inducted']}")
    else:
        _ok(result)


@app.command()
def compress(
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_group: int = typer.Option(10, "--max-group", "-g"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Compress L2 memories. Returns {ok, data: {groups, compressed, memories_removed}}."""
    ctx = _get_ctx()
    result = ctx.decay.compress_l2(max_group_size=max_group, dry_run=dry_run)
    if fmt == "text":
        print(f"groups={result['groups']} compressed={result['compressed']} removed={result['memories_removed']}")
    else:
        _ok(result)


@app.command()
def export(
    format: str = typer.Option("jsonl", "--format", "-f"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
) -> None:
    """Export all memories as JSONL."""
    ctx = _get_ctx()
    memories = ctx.store.list_memories(limit=999_999)

    if format == "jsonl":
        lines = [json.dumps(m.to_dict(), ensure_ascii=False, default=str) for m in memories]
        content = "\n".join(lines)
    elif format == "json":
        content = json.dumps([m.to_dict() for m in memories], ensure_ascii=False, indent=2, default=str)
    else:
        _err(f"Unknown format '{format}'.")

    if output:
        Path(output).write_text(content, encoding="utf-8")
        _ok({"exported": len(memories), "file": output})
    else:
        print(content)


@app.command(name="import")
def import_cmd(
    file: str = typer.Option(..., "--file", "-f"),
) -> None:
    """Import memories from JSONL. Returns {ok, data: {imported, skipped}}."""
    ctx = _get_ctx()
    path = Path(file)
    if not path.exists():
        _err(f"File not found: {file}")

    imported = 0
    skipped = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        try:
            unit = MemoryUnit.from_dict(data)
        except (KeyError, ValueError):
            skipped += 1
            continue
        if ctx.store.get_memory(unit.id) is not None:
            skipped += 1
            continue
        ctx.store.add_memory(unit)
        imported += 1

    _ok({"imported": imported, "skipped": skipped})


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="Memory ID to archive."),
) -> None:
    """Archive a memory to L2. Returns {ok, data: {id, layer}}."""
    ctx = _get_ctx()
    mem = ctx.store.get_memory(memory_id)
    if mem is None:
        _err(f"Memory '{memory_id}' not found.")
    ctx.store.move_to_layer(memory_id, "L2")
    _ok({"id": memory_id, "layer": "L2"})


@app.command()
def extract(
    text: str = typer.Argument(..., help="Text to extract memories from."),
    source: str = typer.Option("cli_extract", "--source", "-s"),
    auto: bool = typer.Option(True, "--auto/--confirm", "-y/-c", help="Auto-store (default) or confirm first."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Extract memories from text. Returns {ok, data: {extracted, memories: [...]}}."""
    ctx = _get_ctx()
    try:
        candidates = ctx.extraction.extract_from_text(text, source=source)
    except Exception as e:
        _err(str(e))

    if not candidates:
        _ok({"extracted": 0, "memories": []})
        return

    stored = []
    for unit in candidates:
        ctx.store.add_memory(unit)
        stored.append(_mem_json(unit))

    if fmt == "text":
        print(f"extracted={len(stored)}")
        for m in stored:
            print(f"  {m['id']} [{m['type']}] {m['content']}")
    else:
        _ok({"extracted": len(stored), "memories": stored})


@app.command()
def migrate(
    action: str = typer.Argument(..., help="Migration action: 're-embed'."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
) -> None:
    """Run migrations. Returns {ok, data: {total, re_embedded, skipped}}."""
    ctx = _get_ctx()

    if action == "re-embed":
        from gitmem0.migrate import re_embed_all

        result = re_embed_all(ctx.store, ctx.embeddings, verbose=verbose)
        if fmt == "text":
            print(f"total={result['total']} re_embedded={result['re_embedded']} skipped={result['skipped']}")
        else:
            _ok(result)
    else:
        _err(f"Unknown migration action '{action}'. Use: re-embed")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
