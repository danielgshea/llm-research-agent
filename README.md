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
  disclosure, stored in the **LangSmith Context Hub**:
  - `model-evaluation` — how to research and compare current models.
  - `benchmark-analysis` — the major benchmarks and how to judge their validity
    (contamination, saturation, self-reported vs independent, construct validity).
  - `landscape-mapping` — the major labs/model families and tracking releases.
  - `scorecard-maintenance` — the scorecard schema and how to update it safely.
  - `continuous-learning` — what to capture after each task so the agent improves.
- **Long-term memory** — `AGENTS.md` (standing context + learnings) and
  `scorecard.md` (ranked best-models-per-category, with scores, sources, dates,
  and caveats), also in the Context Hub. They're loaded into context every run.
- **System prompt** — pulled from the Context Hub at startup, not hard-coded.
- **Continuous learning** — the agent updates memory and the scorecard with its
  own `write_file`/`edit_file` tools, which commit straight back to the Context
  Hub. Knowledge compounds across sessions instead of restarting each time.

### Architecture: everything lives in the Context Hub

This repo holds **only the agent code**. The agent's skills, memory, and system
prompt are stored in the **LangSmith Context Hub** — the single source of truth —
so there are no local copies to drift out of date. A `CompositeBackend` routes
the agent's filesystem by path:

```
/skills/   ─▶ ContextHubBackend(-/llm-research-skills)   # curated playbooks (persistent)
/memory/   ─▶ ContextHubBackend(-/llm-research-memory)   # AGENTS.md + scorecard.md (agent-written)
everything else ─▶ StateBackend()                        # ephemeral scratch (todos, working notes)
```

(The system prompt is pulled from a third repo, `-/llm-research-prompt`, at
construction time.)

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
`LANGSMITH_TRACING=true` sends traces to LangSmith. The Context Hub repos default
to the workspace owner (`-/llm-research-skills`, `-/llm-research-memory`,
`-/llm-research-prompt`); override with `LLM_RESEARCH_SKILLS_REPO`,
`LLM_RESEARCH_MEMORY_REPO`, and `LLM_RESEARCH_PROMPT_REPO`.

## 3. Run

```bash
uv run langgraph dev
```

This serves the `llm_research_agent` graph from `langgraph.json` in LangGraph
Studio. Skills, memory, and the system prompt are read directly from the Context
Hub — no per-invocation seeding needed.

## Editing content

All of the agent's editable content lives in the Context Hub (the source of
truth — there are no local copies in this repo):

- **System prompt** (`-/llm-research-prompt`, `SYSTEM_PROMPT.md`): edit in the
  Hub UI. It's pulled fresh each time the graph is constructed.
- **Skills** (`-/llm-research-skills`, `<name>/SKILL.md`): edit in the Hub UI to
  refine the research playbooks.
- **Memory + scorecard** (`-/llm-research-memory`, `AGENTS.md` + `scorecard.md`):
  primarily owned by the agent — it writes back as it learns. Editable in the Hub
  if you need to correct or reset it.

> Note: because content is Hub-only, it isn't version-controlled in git. Use the
> Context Hub's own commit history (each write is a commit) to track and revert
> changes.
