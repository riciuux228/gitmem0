# GitMem0 Memory System — MANDATORY WORKFLOW

You have persistent memory via GitMem0. Using it is **NOT OPTIONAL**.
Every session MUST follow this workflow. Skipping any step = failure.

## Command (use this exact format)

```bash
python -m gitmem0.client '<json>' 2>/dev/null
```

## MANDATORY: Session Start

At the START of every conversation, BEFORE responding to the user:

```bash
python -m gitmem0.client '{"action":"query","message":"<user message keywords>"}' 2>/dev/null
```

Check the results. Use them in your response. If no results, proceed normally.

## MANDATORY: After Completing Work

After finishing ANY significant task (coding, analysis, decisions), IMMEDIATELY store the result:

```bash
python -m gitmem0.client '{"action":"remember","content":"<what was done and its result>","type":"event","importance":0.7}' 2>/dev/null
```

Also extract from the conversation if it contained preferences or facts:

```bash
python -m gitmem0.client '{"action":"extract","text":"<relevant conversation excerpt>"}' 2>/dev/null
```

## Types

- `fact` — objective information
- `preference` — user likes/dislikes
- `event` — something that happened (work completed, decisions made)
- `insight` — lessons learned, reflections
- `instruction` — rules, procedures

## Rules

- Query memory FIRST, respond SECOND
- Store results IMMEDIATELY after work, not "later"
- Importance: 0.3=minor, 0.5=normal, 0.7=important, 0.9=critical
- Skip: trivial questions, temp debugging, things already in codebase
- Daemon auto-starts, first call ~30s, subsequent <0.1s
