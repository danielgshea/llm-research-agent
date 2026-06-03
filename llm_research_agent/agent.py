"""LLM research agent.

A research analyst for the large language model landscape, built on the
LangChain DeepAgents framework. It investigates which models lead, how the major
benchmarks work and whether they're trustworthy, and who the major labs are —
and it maintains a long-term **scorecard** of the best models that improves over
time.

Storage in LangSmith (workspace-scoped API key resolves its own tenant — no
`LANGSMITH_WORKSPACE_ID` needed):
- **System prompt**: Prompt Hub prompt `llm-research-prompt`, pulled at
  construction. Its system message is used as the agent's system prompt.
- **Skills**: separate Context Hub *skill repos* (one per skill), pulled at
  construction and staged under `/skills/<name>/SKILL.md` so the SkillsMiddleware
  discovers them with progressive disclosure.
- **Memory**: Context Hub *agent repo* `llm-research-memory` (`AGENTS.md` +
  `scorecard.md`), mounted at `/memory/` and written back at runtime — the
  continuous-learning loop.
- **Shared workspace**: a directory mounted at `/shared/` that the `coder`
  subagent also writes to, so this agent can read what the coder produced for
  simple file handoffs (best-effort mount). The agent just sees a shared
  directory; it's backed by a shared LangSmith sandbox. See the
  `shared-workspace` skill.
- A `CompositeBackend` keeps scratch files ephemeral (`StateBackend`).
"""

import logging
import os
import pathlib
import tempfile

from deepagents import create_deep_agent, AsyncSubAgent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.context_hub import ContextHubBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.langsmith import LangSmithSandbox
from langchain.chat_models import init_chat_model
from langsmith import Client
from langsmith.sandbox import SandboxClient

from llm_research_agent.utils.budget import StepBudgetMiddleware
from llm_research_agent.utils.tools import fetch_page, web_search

logger = logging.getLogger(__name__)

# LangSmith resources (override via env to target staging vs production).
MEMORY_REPO = os.environ.get("LLM_RESEARCH_MEMORY_REPO", "llm-research-memory")
PROMPT_NAME = os.environ.get("LLM_RESEARCH_PROMPT_NAME", "llm-research-prompt")
# Each skill is its own Context Hub skill repo; handle == skill (and directory) name.
SKILL_HANDLES = [
    h.strip()
    for h in os.environ.get(
        "LLM_RESEARCH_SKILLS",
        "model-evaluation,benchmark-analysis,landscape-mapping,"
        "scorecard-maintenance,continuous-learning,code-generation,"
        "shared-workspace",
    ).split(",")
    if h.strip()
]

# The shared workspace at /shared/ is backed by a LangSmith sandbox the `coder`
# subagent also writes to; we attach by the same name to read those files for
# handoffs (see the `shared-workspace` skill).
SANDBOX_NAME = os.environ.get("LLM_RESEARCH_SANDBOX", "code-gen-agent-sandbox")

# Memory files loaded into context every run.
MEMORY_SOURCES = ["/memory/AGENTS.md", "/memory/scorecard.md"]

# Bounded research: hard step backstop + graceful final-answer turn budget.
RECURSION_LIMIT = 75
MAX_MODEL_TURNS = 25

client = Client()

# System prompt: `llm-research-prompt` is a ChatPromptTemplate (system +
# `{question}` human message); the agent brings its own conversation, so we take
# only the system message's text.
prompt = client.pull_prompt(PROMPT_NAME)
system_prompt = prompt.messages[0].prompt.template

# Skills: deepagents has no native "mount N skill repos" backend, so we pull each
# repo's SKILL.md into a temp dir as <name>/SKILL.md and root a FilesystemBackend
# there. virtual_mode=True keeps paths relative to root_dir (required for
# CompositeBackend routing) and blocks ../absolute escapes. Skills refresh on
# restart — they're curated, stable config; the evolving state is memory.
skills_root = pathlib.Path(tempfile.mkdtemp(prefix="llm_research_skills_"))
for handle in SKILL_HANDLES:
    skill_md = skills_root / handle / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(client.pull_skill(handle).files["SKILL.md"].content)
skills_backend = FilesystemBackend(root_dir=str(skills_root), virtual_mode=True)

model = init_chat_model(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    base_url="https://gateway.smith.langchain.com/anthropic",
)

# Mount the shared workspace at /shared/, backed by the sandbox the `coder`
# subagent also writes to. Per the `shared-workspace` skill: connect by name with
# get_sandbox() (NOT the auto-deleting `client.sandbox()` context manager), start
# it if it was idled, and never delete it — the coder agent owns its lifecycle.
# Best-effort: the backing sandbox may not exist yet when this agent starts, so a
# failure just disables the /shared/ route rather than breaking the agent.
shared_routes = {}
try:
    sandbox_client = SandboxClient()
    sandbox = sandbox_client.get_sandbox(SANDBOX_NAME)
    if sandbox.status != "ready":
        sandbox = sandbox_client.start_sandbox(SANDBOX_NAME)
    shared_routes["/shared/"] = LangSmithSandbox(sandbox)
except Exception as exc:  # noqa: BLE001 — shared workspace is optional at startup
    logger.warning(
        "Shared workspace %r unavailable (%s); /shared/ route disabled.",
        SANDBOX_NAME,
        exc,
    )

# Scratch files stay ephemeral in state; /skills serves the pulled skill repos;
# /memory persists to the LangSmith Context Hub; /shared/ (when available) is
# the directory shared with the coder subagent.
backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/skills/": skills_backend,
        "/memory/": ContextHubBackend(MEMORY_REPO),
        **shared_routes,
    },
)

async_subagents = [
    AsyncSubAgent(
        name="coder",
        description=(
            "Use PROACTIVELY for any task whose deliverable is runnable code — "
            "writing/generating scripts, programs, data visualizations, or "
            "notebooks, and reviewing or refactoring code. Prefer this over "
            "writing code inline with the file tools. Pass a complete, "
            "self-contained spec (it cannot see the conversation): the goal, the "
            "data/values to embed, expected inputs/outputs, output file path, and "
            "the language/libraries to use. See the `code-generation` skill."
        ),
        graph_id="coder",
        url="https://code-gen-agent-deployed-682e04b101d656aba3008a1dbef1884e.us.langgraph.app"
    ),
]

agent = create_deep_agent(
    model=model,
    tools=[web_search, fetch_page],
    system_prompt=system_prompt,
    backend=backend,
    skills=["/skills/"],
    memory=MEMORY_SOURCES,
    subagents=async_subagents,
    middleware=[StepBudgetMiddleware(max_model_turns=MAX_MODEL_TURNS)],
).with_config({"recursion_limit": RECURSION_LIMIT})
