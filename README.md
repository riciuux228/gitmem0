# GitMem0

Git x Mem0: A pure-local, version-controlled memory system for LLMs.

English | [中文](README_zh.md)

GitMem0 gives AI agents persistent memory without external APIs or cloud services. It combines Git-style version tracking with Mem0's intelligent memory layer — all running on your machine.

## Features

- **Pure local** — no API keys, no cloud, everything stays on disk
- **Fast** — daemon architecture, model loads once, queries in <0.05s
- **Multi-signal retrieval** — semantic search + BM25 + entity graph + recency + importance scoring
- **Confidence decay** — memories naturally fade over time (exponential decay), archived to L2 when stale
- **Knowledge graph** — entities (person, technology, project...) and relations auto-extracted
- **Version control** — every memory is immutable like a git commit, with full diff/history
- **LLM-ready context** — Lost-in-the-Middle arrangement, token budget compression
- **Multilingual** — uses `paraphrase-multilingual-MiniLM-L12-v2`, supports 50+ languages
- **LLM Judge plugin** — optional LLM-assisted scoring with automatic fallback to rules

## Quick Start

```bash
pip install -e .

# First call auto-starts daemon (loads model, ~30s), subsequent calls <0.1s
python -m gitmem0.client '{"action":"remember","content":"I prefer dark mode","type":"preference","importance":0.9}'
python -m gitmem0.client '{"action":"query","message":"user preferences"}'
python -m gitmem0.client '{"action":"search","query":"dark mode"}'
python -m gitmem0.client '{"action":"stats"}'
```

## Architecture

```
User/LLM
    |
    v
client.py (thin, <0.1s startup)
    | TCP socket
    v
auto.py daemon (port 19840, model loaded once)
    |
    +---> extraction.py  (multi-signal importance + type inference)
    +---> retrieval.py   (two-stage: recall → rerank)
    +---> context.py     (Lost-in-the-Middle + token budget)
    +---> decay.py       (exponential decay + consolidation)
    +---> entities.py    (knowledge graph extraction)
    +---> store.py       (SQLite + FTS5 + L0 LRU cache)
    +---> embeddings.py  (sentence-transformers, 384d)
```

## CLI

```bash
# Typer CLI (alternative to client.py)
python -m gitmem0.cli add "Python is great for prototyping" --type fact --importance 0.7
python -m gitmem0.cli search "Python" --top 3
python -m gitmem0.cli context "what languages does the user like"
python -m gitmem0.cli extract "I always use type hints. Never skip tests."
python -m gitmem0.cli stats
python -m gitmem0.cli decay --dry-run
python -m gitmem0.cli consolidate --threshold 0.85
python -m gitmem0.cli export --format jsonl -o backup.jsonl
python -m gitmem0.cli migrate re-embed
```

## Claude Code Integration

GitMem0 works as a memory backend for Claude Code via hooks:

```bash
# Setup hooks
python hooks/setup_claude_code.py
```

This installs `UserPromptSubmit` and `Stop` hooks that automatically query and store memories during conversations.

## Memory Types

| Type | Example | Default Importance |
|------|---------|-------------------|
| `preference` | "I prefer dark mode" | 0.9 |
| `instruction` | "Always use type hints" | 0.9 |
| `insight` | "React is better for this UI" | 0.8 |
| `fact` | "Python is a programming language" | 0.7 |
| `event` | "Deployed v2.0 on Monday" | 0.4 |

## Configuration

Edit `~/.gitmem0/config.toml`:

```toml
[storage]
db_path = "gitmem0.db"
active_threshold = 0.3

[decay]
lambda_rate = 0.01  # ~70 day half-life

[retrieval]
weight_semantic = 0.25
weight_bm25 = 0.15
weight_entity = 0.15
weight_recency = 0.10
weight_importance = 0.20
weight_confidence = 0.15

[embedding]
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
dimension = 384
```

## LLM Judge Plugin

Implement the `LLMJudge` protocol to plug in any LLM:

```python
from gitmem0.extraction import LLMJudge, MemoryType

class MyJudge(LLMJudge):
    def score_importance(self, content, context="") -> float | None:
        # Return 0.0-1.0, or None to use rule-based default
        ...

    def should_remember(self, content) -> bool | None:
        # Return True/False, or None to use rule-based default
        ...

    def infer_type(self, content) -> MemoryType | None:
        # Return MemoryType, or None to use pattern matching
        ...

    def summarize(self, memories: list[str]) -> str | None:
        # Return summary, or None to use concatenation
        ...

# Pass to AutoMemory
from gitmem0.auto import AutoMemory
auto = AutoMemory(llm_judge=MyJudge())
```

Without an LLM Judge, the system falls back to rule-based scoring automatically.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
