import json
import sys

event = json.load(sys.stdin)
command = event.get("tool_input", {}).get("command", "")
if "rm " in command or ".env" in command:
    print("Blocked by policy: no deletes, no access to .env",
          file=sys.stderr)
    sys.exit(2)
