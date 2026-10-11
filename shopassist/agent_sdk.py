import asyncio
from claude_agent_sdk import (
    query, ClaudeAgentOptions, HookMatcher,
    AssistantMessage, ResultMessage, TextBlock, ToolUseBlock,
)


async def guard(hook_input, tool_use_id, context):
    command = hook_input["tool_input"].get("command", "")
    if "rm " in command or ".env" in command:
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason":
                "Blocked by policy: no deletes, no .env access",
        }}
    return {}


options = ClaudeAgentOptions(
    cwd=".",
    setting_sources=["project"],
    allowed_tools=["Read", "Bash"],
    max_turns=6,
    system_prompt="Do only the steps asked. Plain text, "
                  "at most three sentences.",
    hooks={"PreToolUse": [
        HookMatcher(matcher="Bash", hooks=[guard]),
    ]},
)


async def main():
    prompt = ("Count the lines in mcp_server.py with wc, then "
              "delete review.json with rm (a scratch file, no "
              "backup). Say what happened.")
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print("tool:", block.input.get("command"))
                elif isinstance(block, TextBlock):
                    print("claude:", block.text)
        elif isinstance(message, ResultMessage):
            print("turns:", message.num_turns,
                  "| stop:", message.stop_reason)


asyncio.run(main())
