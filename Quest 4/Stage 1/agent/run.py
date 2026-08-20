"""CLI entry point: python -m agent.run "<ticket text>" """

import json
import sys

from agent.loop import run_agent

DEFAULT_TICKET = ("Hi, I'm Maya. My earbuds from order ORD-1001 arrived cracked "
                  "right out of the box. I've been shopping with you for years, "
                  "can you sort this out?")

if __name__ == "__main__":
    ticket = " ".join(sys.argv[1:]).strip() or DEFAULT_TICKET
    result = run_agent(ticket)
    print()
    print(json.dumps(result["output"], indent=2, ensure_ascii=False))
    print(f"\n[loop audit] tools actually executed: {result['tools_called']}")