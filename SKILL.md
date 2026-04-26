# GitMem0 — AI Agent Persistent Memory Skill

> Version: 1.0 | Applies to: Claude Code, Copilot CLI, Gemini CLI, any AI agent

GitMem0 gives you persistent memory across sessions. This document defines the **mandatory workflow** for any AI agent using GitMem0. Follow it exactly. Skipping any step = failure.

---

## Core Principle

**You have memory. Use it. Always.**

- Query memory BEFORE responding to the user
- Store memory AFTER completing significant work
- "Later" does not exist — if you think "I'll remember later", you won't

---

## Commands

All commands use this format:

```bash
python -m gitmem0.client '<json>' 2>/dev/null
```

The daemon auto-starts on first call (~30s), subsequent calls <0.05s.

### Query (retrieve relevant memories)

```bash
python -m gitmem0.client '{"action":"query","message":"<keywords describing what you need>"}' 2>/dev/null
```

Returns relevant memories, entities, and relations. Use this information in your response.

### Remember (store a memory)

```bash
python -m gitmem0.client '{"action":"remember","content":"<what happened or was decided>","type":"<type>","importance":<0.0-1.0>}' 2>/dev/null
```

### Search (keyword search)

```bash
python -m gitmem0.client '{"action":"search","query":"<exact keywords>"}' 2>/dev/null
```

### Stats

```bash
python -m gitmem0.client '{"action":"stats"}' 2>/dev/null
```

### Extract (analyze text for memories without storing)

```bash
python -m gitmem0.client '{"action":"extract","text":"<text to analyze>"}' 2>/dev/null
```

---

## Memory Types

| Type | When to use | Example | Default Importance |
|------|-------------|---------|-------------------|
| `instruction` | Rules, procedures, constraints | "Always use type hints" | 0.9 |
| `preference` | User likes/dislikes | "I prefer dark mode" | 0.9 |
| `insight` | Lessons learned, reflections | "React is better for this UI" | 0.8 |
| `fact` | Objective information | "The API uses JWT auth" | 0.7 |
| `event` | Something that happened | "Refactored auth module, all tests pass" | 0.4 |

---

## Importance Scale

| Value | Meaning | When to use |
|-------|---------|-------------|
| 0.9-1.0 | Critical | User instructions, core rules, breaking decisions |
| 0.7-0.8 | Important | Completed work, key findings, architecture decisions |
| 0.5-0.6 | Normal | General facts, routine observations |
| 0.3-0.4 | Minor | Temporary notes, minor events |

---

## Mandatory Workflow

### Step 1: Session Start — Query First

BEFORE responding to the user's first message, extract keywords and query:

```bash
python -m gitmem0.client '{"action":"query","message":"<keywords from user message>"}' 2>/dev/null
```

Check the results. If relevant memories exist, incorporate them into your response. If no results, proceed normally.

### Step 2: During Work — Query as Needed

Before making decisions or recommendations, check if you have relevant memories:

```bash
python -m gitmem0.client '{"action":"query","message":"<topic you are working on>"}' 2>/dev/null
```

### Step 3: After Work — Remember Immediately

After completing ANY significant task, IMMEDIATELY store what was done:

```bash
python -m gitmem0.client '{"action":"remember","content":"<concise summary of what was done and its result>","type":"event","importance":0.7}' 2>/dev/null
```

**Do NOT skip this step. Do NOT postpone it. Do it NOW.**

---

## What to Remember

**Store:**
- Decisions made and their rationale
- Work completed and its outcome
- User preferences and instructions
- Lessons learned from debugging
- Key facts about the project or user

**Skip:**
- Trivial questions ("what time is it?")
- Temporary debugging output
- Information already in the codebase
- Repetitive confirmations

---

## Response Checklist

Before EVERY response, verify:

```
[ ] Queried memory with relevant keywords?
[ ] If significant work was completed, remembered it?
[ ] Used memory results in my response (if relevant)?
```

If any box is unchecked, do it NOW before responding.

---

## Platform Integration

### Claude Code

**Option A: CLAUDE.md (recommended)**

Add to your project's `CLAUDE.md`:

```markdown
# Memory System — MANDATORY

Before EVERY response:
1. Query: python -m gitmem0.client '{"action":"query","message":"<keywords>"}' 2>/dev/null
2. After work: python -m gitmem0.client '{"action":"remember","content":"...","type":"event","importance":0.7}' 2>/dev/null

This is NOT optional. Skipping = failure.
```

**Option B: Hooks (automatic)**

```bash
python hooks/setup_claude_code.py
```

This installs a `Stop` hook that auto-extracts memories when the AI finishes responding.

### Copilot CLI

Add to your `AGENTS.md` or system prompt:

```markdown
You have persistent memory via GitMem0. Use these commands:
- Query: python -m gitmem0.client '{"action":"query","message":"..."}' 2>/dev/null
- Remember: python -m gitmem0.client '{"action":"remember","content":"...","type":"event","importance":0.7}' 2>/dev/null

Always query before responding. Always remember after completing work.
```

### Gemini CLI

Add to your `GEMINI.md`:

```markdown
## Memory

You have persistent memory. Use it EVERY response.

Before responding: python -m gitmem0.client '{"action":"query","message":"<keywords>"}' 2>/dev/null
After completing work: python -m gitmem0.client '{"action":"remember","content":"...","type":"event","importance":0.7}' 2>/dev/null
```

### Generic / System Prompt

```
You have access to a persistent memory system. Use it in every conversation.

BEFORE responding to any user message:
  Run: python -m gitmem0.client '{"action":"query","message":"<extracted keywords>"}' 2>/dev/null
  Use the results to inform your response.

AFTER completing any significant task (coding, analysis, decisions):
  Run: python -m gitmem0.client '{"action":"remember","content":"<what was done>","type":"event","importance":0.7}' 2>/dev/null

Memory types: instruction (rules), preference (likes), insight (lessons), fact (info), event (happened)
Importance: 0.3=minor, 0.5=normal, 0.7=important, 0.9=critical

This workflow is MANDATORY. Do not skip it. Do not postpone it.
```

---

## LLM Judge Plugin (Optional)

### Quick Setup (Xiaomi Token Plan / OpenAI-compatible)

Add to `~/.gitmem0/config.toml`:

```toml
[llm]
api_key = "tp-xxxxx"
base_url = "https://token-plan-cn.xiaomimimo.com/v1"
model = "MiMo"
```

Or set environment variables: `GITMEM0_LLM_API_KEY` and `GITMEM0_LLM_BASE_URL`.

The daemon auto-detects and enables LLM-assisted scoring. Falls back to rules if API is unavailable.

### Custom Implementation

```python
from gitmem0.extraction import LLMJudge, MemoryType

class MyJudge(LLMJudge):
    def score_importance(self, content, context="") -> float | None:
        """Return 0.0-1.0, or None to use rule-based default."""
        ...

    def should_remember(self, content) -> bool | None:
        """Return True/False, or None to use rule-based default."""
        ...

    def infer_type(self, content) -> MemoryType | None:
        """Return MemoryType, or None to use pattern matching."""
        ...

    def summarize(self, memories: list[str]) -> str | None:
        """Return summary, or None to use concatenation."""
        ...

# Pass to AutoMemory
from gitmem0.auto import AutoMemory
auto = AutoMemory(llm_judge=MyJudge())
```

Without an LLM Judge, the system uses rule-based scoring automatically.

---

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

[llm]
api_key = ""  # tp-xxxxx (Xiaomi Token Plan) or any OpenAI-compatible key
base_url = "https://token-plan-cn.xiaomimimo.com/v1"
model = "MiMo"
```

---

## Quick Start

```bash
# Install from PyPI
pip install gitmem0

# Or install from source (development)
pip install -e .

# First call loads model (~30s), subsequent calls <0.05s
python -m gitmem0.client '{"action":"remember","content":"I prefer dark mode","type":"preference","importance":0.9}'
python -m gitmem0.client '{"action":"query","message":"user preferences"}'
python -m gitmem0.client '{"action":"search","query":"dark mode"}'
python -m gitmem0.client '{"action":"stats"}'
```

---

## Anti-Patterns (DO NOT DO THESE)

| Wrong | Right |
|-------|-------|
| "I'll remember it later" | Remember it NOW |
| Responding without querying | Query FIRST, respond SECOND |
| Only using memory for "important" things | Use it for ALL significant work |
| Storing vague content | Be specific: what, why, result |
| Ignoring query results | Incorporate relevant memories into response |
| Treating memory as optional | It is MANDATORY |
