"""Build Hermes-like prompts: Hermes's real tool schemas and instruction text, synthetic per-session details.

Run with Hermes's own Python and a throwaway HERMES_HOME, e.g.:
  HERMES_HOME=$(mktemp -d) /path/to/hermes-agent/.venv/bin/python make_hermes_prompts.py out.json

Output: {"tools": [...], "stable": str, "context": str, "skills": str, "memory": [str, str], "volatile_template": str}
The system prompt follows Hermes's order (agent/system_prompt.py): stable, then context, then volatile.
"""
import json
import random
import sys

sys.path.insert(0, "/localhome/local-rochaudhari/roshan/hermes-agent")

import model_tools  # noqa: E402
from agent import prompt_builder as pb  # noqa: E402

WORDS = ("build test deploy cache session memory agent tool skill config server model token prompt slot "
         "profile gateway terminal browser file search patch review commit branch release docs").split()


def text(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def main():
    tools = model_tools.get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)

    parts = [pb.DEFAULT_AGENT_IDENTITY, pb.HERMES_AGENT_HELP_GUIDANCE, pb.MEMORY_GUIDANCE, pb.SESSION_SEARCH_GUIDANCE,
             pb.SKILLS_GUIDANCE, pb.TOOL_USE_ENFORCEMENT_GUIDANCE, pb.TASK_COMPLETION_GUIDANCE,
             pb.PARALLEL_TOOL_CALL_GUIDANCE, pb.ASYNC_HANDOFF_GUIDANCE, pb.STEER_CHANNEL_NOTE]
    stable = "\n\n".join(p for p in parts if isinstance(p, str) and p.strip())

    rng = random.Random(42)
    context = "# Project context (AGENTS.md)\n\n" + "\n".join(f"- {text(rng, 18)}." for _ in range(45))
    skills = "## Available skills\n\n" + "\n".join(f"- skill-{i:02d}: {text(rng, 12)}" for i in range(40))
    memory = ["## Memory\n\n" + "\n".join(f"- {text(random.Random(seed * 100 + i), 14)}" for i in range(30))
              for seed in (1, 2)]

    volatile_template = (
        "{skills}\n\n{memory}\n\nActive profile: default\n"
        "Conversation started: {date} (UTC-07:00, PDT)\nSession ID: {session_id}\n"
        "Model: qwen3.6\nProvider: local\nPlatform: cli\n\n"
        "## Runtime environment\n\nOS: Linux 5.15 x86_64\nShell: bash\nCurrent working directory: /home/user/project\n"
    )

    with open(sys.argv[1], "w") as f:
        json.dump({"tools": tools, "stable": stable, "context": context, "skills": skills, "memory": memory,
                   "volatile_template": volatile_template}, f)
    print(f"tools: {len(tools)}, tools json chars: {len(json.dumps(tools))}, stable chars: {len(stable)}, "
          f"context chars: {len(context)}, skills chars: {len(skills)}, memory chars: {len(memory[0])}")


if __name__ == "__main__":
    main()
