# LLM Research Agent

A research analyst for the **large language model landscape**, built on the
[LangChain **DeepAgents**](https://docs.langchain.com/oss/python/deepagents/overview)
framework and packaged as a
[LangGraph application](https://docs.langchain.com/oss/python/langgraph/application-structure)
(`langgraph.json`) so it runs in LangGraph Studio and deploys on LangSmith.

Ask it things like:

- "Which model is best for coding / reasoning / long-context / agents right now?"
- "What are the most important benchmarks — and can I trust them?"
- "Who are the major players, and what did they ship recently?"
- "How does model A compare to model B?"

It answers with **sourced, dated** analysis, is **skeptical of benchmark claims**,
and keeps a long-term **scorecard** of the best models that improves every time
it runs.

## How it works

- **Tools** — Tavily `web_search` (multi-result, with URLs + dates) and
  `fetch_page` (full-page extraction) for sourced research.
- **Skills** — five research playbooks loaded on demand via progressive
  disclosure, each stored as its own **LangSmith Context Hub skill repo**
  (`pull_skill`), pulled at startup:
  - `model-evaluation` — how to research and compare current models.
  - `benchmark-analysis` — the major benchmarks and how to judge their validity
    (contamination, saturation, self-reported vs independent, construct validity).
  - `landscape-mapping` — the major labs/model families and tracking releases.
  - `scorecard-maintenance` — the scorecard schema and how to update it safely.
  - `continuous-learning` — what to capture after each task so the agent improves.
- **Long-term memory** — `AGENTS.md` (standing context + learnings) and
  `scorecard.md` (ranked best-models-per-category, with scores, sources, dates,
  and caveats), in the Context Hub agent repo `llm-research-memory`. Loaded into
  context every run.
- **System prompt** — pulled from the **LangSmith Prompt Hub** prompt
  `llm-research-prompt` at startup (its system message), not hard-coded.
- **Continuous learning** — the agent updates memory and the scorecard with its
  own `write_file`/`edit_file` tools, which commit straight back to the Context
  Hub. Knowledge compounds across sessions instead of restarting each time.
- **Bounded research** — the prompt sets a search budget and stopping criteria,
  and a `StepBudgetMiddleware` forces a clean, tool-free final answer once the
  model-turn budget is nearly spent. A `recursion_limit` (`agent.py`) is the hard
  backstop. Together these prevent the agent from looping on hard-to-verify,
  fast-moving figures — the user always gets a finished, caveated answer rather
  than a `GraphRecursionError`.

### Architecture: state lives in LangSmith

This repo holds **only the agent code** — skills, memory, and the system prompt
all live in LangSmith (the single source of truth), so there are no local copies
to drift out of date. At construction the agent pulls the **Prompt Hub** prompt
`llm-research-prompt` (system message) and each **skill repo** (`pull_skill`),
staging the skills locally so a `CompositeBackend` can serve them:

```
/skills/   ─▶ FilesystemBackend(temp dir, virtual_mode=True)  # skill repos pulled at startup (progressive disclosure)
/memory/   ─▶ ContextHubBackend("llm-research-memory")        # AGENTS.md + scorecard.md (agent-written, persistent)
everything else ─▶ StateBackend()                             # ephemeral scratch (todos, working notes)
```

Skills and the prompt are pulled at startup (curated, stable — edit them in
LangSmith and restart/redeploy). Memory is the evolving state, written back to the
Context Hub at runtime. The key must be **scoped to the workspace** these repos
live in (then it resolves its own tenant — no `LANGSMITH_WORKSPACE_ID` needed).

## Project layout

```
llm-research/
├── llm_research_agent/
│   ├── agent.py              # builds the compiled graph (`agent`); pulls the prompt from the Hub
│   └── utils/
│       └── tools.py          # Tavily web_search + fetch_page
├── langgraph.json
├── pyproject.toml
└── .env.example
```

The Context Hub repos (skills, memory, prompt) are created/maintained in the Hub,
not from this repo. See **Editing content** below.

## 1. Install

```bash
uv sync
```

## 2. Configure keys

```bash
cp .env.example .env   # then fill in your keys
```

You need `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, and `LANGSMITH_API_KEY`. Leaving
`LANGSMITH_TRACING=true` sends traces to LangSmith. **Use a `LANGSMITH_API_KEY`
scoped to the workspace** that holds the prompt/skill/memory repos — it then
resolves its own tenant and needs no `LANGSMITH_WORKSPACE_ID`. (An
org/multi-workspace key resolves no tenant and 404s on the repos; set
`LANGSMITH_WORKSPACE_ID` for those.) Resource names are overridable via
`LLM_RESEARCH_PROMPT_NAME`, `LLM_RESEARCH_MEMORY_REPO`, and `LLM_RESEARCH_SKILLS`.

## 3. Run

```bash
uv run langgraph dev
```

This serves the `llm_research_agent` graph from `langgraph.json` in LangGraph
Studio. The system prompt, skills, and memory are read directly from LangSmith —
no per-invocation seeding needed.

## Editing content

All of the agent's editable content lives in LangSmith (the source of truth —
there are no local copies in this repo):

- **System prompt** — Prompt Hub prompt `llm-research-prompt` (its system
  message): edit in the LangSmith UI; pulled fresh each time the graph is built.
- **Skills** — one Context Hub **skill repo** per skill (`model-evaluation`,
  `benchmark-analysis`, `landscape-mapping`, `scorecard-maintenance`,
  `continuous-learning`), each with a `SKILL.md`: edit in the UI to refine a
  playbook; pulled at startup.
- **Memory + scorecard** — Context Hub agent repo `llm-research-memory`
  (`AGENTS.md` + `scorecard.md`): primarily owned by the agent — it writes back
  as it learns. Editable in the UI to correct or reset.

> Note: because content lives in LangSmith, it isn't version-controlled in git.
> Use each repo's own commit history (each write is a commit) to track/revert.
