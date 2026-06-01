"""LLM research agent.

A research analyst for the large language model landscape, built on the
LangChain DeepAgents framework. It investigates which models lead, how the major
benchmarks work and whether they're trustworthy, and who the major labs are —
and it maintains a long-term **scorecard** of the best models that improves over
time.

Architecture:
- **Tools**: Tavily `web_search` + `fetch_page` for sourced web research.
- **Skills** (`/skills/`): on-demand research playbooks, read from a LangSmith
  Context Hub repo via `ContextHubBackend`.
- **Memory** (`/memory/`): `AGENTS.md` + `scorecard.md`, also in the Context Hub.
  Loaded into context every run and written back via the agent's own
  `write_file`/`edit_file` tools — this is the continuous-learning loop.
- **System prompt**: pulled from its own Context Hub repo at construction time,
  so it's editable in the Hub rather than hard-coded here.
- A `CompositeBackend` keeps scratch files ephemeral (`StateBackend`) while
  routing `/skills/` and `/memory/` to the persistent Hub.
"""

import os

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.context_hub import ContextHubBackend
from langchain.chat_models import init_chat_model
from langsmith import Client

from llm_research_agent.utils.budget import StepBudgetMiddleware
from llm_research_agent.utils.tools import fetch_page, web_search

# Hard backstop: cap graph super-steps so a non-converging run can't run
# unbounded. StepBudgetMiddleware forces a clean final answer well before this.
RECURSION_LIMIT = 75
# Force a final, tool-free answer after this many model turns (~2 steps each).
MAX_MODEL_TURNS = 25

# Context Hub agent repos — the source of truth for skills, memory, and the
# system prompt. "-/<name>" resolves to the workspace owner; override via env
# vars to target a specific Hub/workspace (e.g. staging vs production).
SKILLS_REPO = os.environ.get("LLM_RESEARCH_SKILLS_REPO", "llm-research-skills")
MEMORY_REPO = os.environ.get("LLM_RESEARCH_MEMORY_REPO", "llm-research-memory")
PROMPT_REPO = os.environ.get("LLM_RESEARCH_PROMPT_REPO", "llm-research-prompt")

# Memory files loaded into context every run (AGENTS.md = role/learnings,
# scorecard.md = model rankings).
MEMORY_SOURCES = ["/memory/AGENTS.md", "/memory/scorecard.md"]


def _load_system_prompt() -> str:
    """Pull the system prompt from the Context Hub at construction time.

    The prompt is curated config, not agent-mutated state, so it lives in its own
    Hub repo and is fetched once when the graph is built — editing it is done in
    the Hub, with no code change here.
    """
    return Client().pull_agent(PROMPT_REPO).files["SYSTEM_PROMPT.md"].content


model = init_chat_model(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    base_url="https://gateway.smith.langchain.com/anthropic",
)

# Scratch files (todos, working notes, summarization offload) stay ephemeral in
# state; only /skills and /memory are persisted to the LangSmith Context Hub.
backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/skills/": ContextHubBackend(SKILLS_REPO),
        "/memory/": ContextHubBackend(MEMORY_REPO),
    },
)

agent = create_deep_agent(
    model=model,
    tools=[web_search, fetch_page],
    system_prompt=_load_system_prompt(),
    backend=backend,
    skills=["/skills/"],
    memory=MEMORY_SOURCES,
    middleware=[StepBudgetMiddleware(max_model_turns=MAX_MODEL_TURNS)],
).with_config({"recursion_limit": RECURSION_LIMIT})
