# GitMem0 — Memory System (MANDATORY)

You have persistent memory via GitMem0. **This is NOT optional.**

Full specification: [SKILL.md](SKILL.md)

## Workflow (every response)

1. **Before responding:** `python -m gitmem0.client '{"action":"query","message":"<keywords>"}' 2>/dev/null`
2. **After completing work:** `python -m gitmem0.client '{"action":"remember","content":"<what was done>","type":"event","importance":0.7}' 2>/dev/null`

## Rules

- Query FIRST, respond SECOND
- Remember IMMEDIATELY, not "later"
- Types: `instruction` | `preference` | `insight` | `fact` | `event`
- Importance: 0.3=minor, 0.5=normal, 0.7=important, 0.9=critical
- Skip: trivial questions, temp debugging, things already in codebase
