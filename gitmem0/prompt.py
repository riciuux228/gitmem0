"""System prompt templates for memory-aware AI.

These prompts teach the AI when and how to use GitMem0 automatically.
The AI doesn't need to "decide" — the prompts create automatic behaviors.
"""

# ── Core memory prompt (injected into system prompt) ─────────────────────────

MEMORY_SYSTEM_PROMPT = """## Memory System (GitMem0)

You have access to a persistent memory system. It remembers facts, preferences,
events, and insights from past conversations. You MUST use it automatically.

### Automatic Behaviors

**At the start of each conversation:**
Before responding, run this command to check for relevant memories:
```
python -m gitmem0.auto context "<first few words of user's message>"
```
If it returns memories, use them to inform your response. Don't mention the
memory system unless the user asks about it.

**When you learn something worth remembering:**
If the user tells you something that would be useful in future conversations
(preferences, facts, project details, instructions), store it:
```
gitmem0 add "<what to remember>" --type <fact|preference|event|insight|instruction> --importance <0.3-1.0>
```

**At the end of a substantive conversation:**
If the conversation contained new information worth remembering:
```
python -m gitmem0.auto extract "<summary of key points from conversation>"
```

### What to Remember (high importance: 0.7-1.0)
- User preferences (tools, languages, style, workflow)
- Project details (name, tech stack, goals, constraints)
- Explicit instructions ("always do X", "never do Y")
- Key decisions and their rationale
- Recurring patterns in user behavior

### What NOT to Remember (skip or low importance: 0.1-0.3)
- One-off questions with no future relevance
- Temporary debugging info
- Information that's in the codebase already
- Casual conversation with no actionable content

### Memory Types
- `fact`: Objective information ("Python 3.12 released Oct 2023")
- `preference`: User preferences ("I prefer dark mode")
- `event`: Things that happened ("Deployed v2.0 on Monday")
- `insight`: Derived understanding ("React is better for this UI pattern")
- `instruction`: Directives ("Always use type hints in Python")

### Search When Needed
If you're unsure about something you might have discussed before:
```
gitmem0 search "<what you're looking for>" --top 3
```
"""

# ── Minimal prompt (for token-conscious setups) ──────────────────────────────

MEMORY_PROMPT_MINIMAL = """## Memory
You have persistent memory via gitmem0.
- Check memory: `python -m gitmem0.auto context "<topic>"`
- Store memory: `gitmem0 add "<content>" --type <type> --importance <0.3-1.0>`
- Search memory: `gitmem0 search "<query>"`
Remember: preferences, project info, instructions, decisions. Skip: one-offs, temp info.
"""

# ── Hook scripts (for Claude Code / other hook systems) ──────────────────────

# These are the actual shell commands that hooks execute.

HOOK_PRE_RESPONSE = 'python -m gitmem0.auto context "{message}"'
HOOK_POST_RESPONSE = 'python -m gitmem0.auto extract "{conversation}"'
HOOK_CYCLE = 'python -m gitmem0.auto cycle "{user_message}" "{assistant_response}"'

# ── Claude Code hook configuration ───────────────────────────────────────────

CLAUDE_CODE_HOOKS = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python -m gitmem0.auto context \"$CLAUDE_USER_MESSAGE\""
                    }
                ]
            }
        ],
        "PostToolUse": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python -m gitmem0.auto extract \"$CLAUDE_CONVERSATION\""
                    }
                ]
            }
        ]
    }
}


def get_memory_prompt(level: str = "full") -> str:
    """Get the memory system prompt.

    Args:
        level: "full" for complete prompt, "minimal" for token-saving version
    """
    if level == "minimal":
        return MEMORY_PROMPT_MINIMAL
    return MEMORY_SYSTEM_PROMPT
