# LangSmith Engine demo — runbook

> **Internal demo asset. Do not merge `demo/engine-missing-dates` into `main`.**
> This branch carries a deliberately planted bug so LangSmith Engine has something
> real to discover, diagnose, and fix in front of a customer. `main` is the clean,
> fixed reference state.

> [!IMPORTANT]
> **Keep this file out of Engine's view.** Engine reads the connected source
> repository to diagnose problems. If it scans a file that names the bug, the
> "Engine figured it out on its own" narrative is spoiled. When you connect the
> repo in Engine setup, point the **Code repository subfolder** at
> `llm_research_agent/` (the agent source) rather than the repo root, so this
> runbook and other demo notes stay invisible to Engine.

---

## The planted bug

**File:** `llm_research_agent/utils/tools.py`, inside `web_search()`.

**The change (one character of intent, easy to miss in review):**

```python
# Correct (on main):
"published_date": r.get("published_date"),

# Planted on this branch:
"published_date": r.get("date"),
```

Tavily returns each result's date under the key `published_date`. Reading
`r.get("date")` returns `None` for **every** result, because `"date"` is not a
key Tavily emits. The `published_date` field in the tool's JSON output is
therefore always `null`.

### Why this is a realistic, easy-to-miss mistake

- The wrong key is a plausible guess — `date` reads as the "obvious" field name,
  and different APIs really do use different keys (`date`, `published`,
  `publishedAt`, `published_date`).
- Nothing crashes. The tool returns well-formed JSON; the only difference is one
  field is consistently `null`.
- Unit tests that assert "results have a `published_date` key" still pass — the
  key exists, it's just always empty.
- Code review slides past it: the surrounding code, the docstring (which still
  promises `published_date`), and the call site all look correct.

### Why it's a real, valuable problem

This agent researches a **fast-moving** LLM landscape and is explicitly
instructed (in its system prompt and `web_search` docstring) to **cite every
claim with a link and a date**, and to record dates in the long-term scorecard.
With dates silently stripped at the tool boundary, the agent can no longer date
anything. Across runs you get a consistent, recurring quality defect:

- answers omit publication dates, or
- the agent hedges / says recency is "unverified," or
- the scorecard accumulates undated entries,

even though the prompt demands dated, sourced claims. It degrades exactly the
capability this agent exists to provide, and it does so quietly.

---

## What Engine should surface

- **Category:** a correctness / answer-quality issue (not a hard tool failure).
  Select priorities like **Tool Call Failures** and a custom concern such as
  *"answers must cite publication dates"* during Engine setup to steer it.
- **Diagnosis (expected):** the agent consistently fails to provide publication
  dates despite instructions; root cause traced to `web_search` returning
  `published_date: null` because of the wrong Tavily result key.
- **Proposed fix:** restore `r.get("published_date")`.
- **Suggested evaluator:** asserts that answers citing sources include dates /
  that tool outputs carry non-null `published_date`. This is the regression
  guard that auto-reopens the issue if it returns.
- **Offline examples:** Engine generates ground-truth examples from the failing
  production traces (dated answers as the expected output).

---

## Running the demo

1. **Be on the bug branch.**
   ```bash
   git checkout demo/engine-missing-dates
   ```

2. **Seed recurring traces into a fresh, disposable tracing project.** Engine
   surfaces *recurring* patterns, so generate a batch (≈15–20 runs) over the
   golden questions. Use a unique project name per demo so Engine re-detects from
   a clean slate.
   ```bash
   LANGSMITH_PROJECT="engine-demo-$(date +%Y%m%d-%H%M)" \
     uv run --env-file .env python evals/seed_demo_traces.py
   ```
   (Seed script reuses the `llm-research-golden` questions; see that file.)

3. **Set up Engine on that project.** In LangSmith → Tracing → *(the demo
   project)* → **Engine** tab → connect the repo with subfolder
   `llm_research_agent/`, pick priorities (add the custom *"answers must cite
   publication dates"* concern), **Start Analyzing**. Allow up to ~20 min for the
   first scan.

4. **Walk the customer through the issue:** diagnosis → Linked traces (point out
   the `published_date: null` in tool outputs and the undated answers) →
   **Proposed Fix** → **Open PR** → **Suggested Evaluator** → **Offline
   examples**.

---

## Resetting between demos (zero re-authoring)

- Easiest: use a **new `LANGSMITH_PROJECT` name** each run (step 2). Engine
  analyzes per project, so a fresh project = a fresh detection from scratch.
- Alternatively, in Engine settings use **Delete all issues**.
- Discard or close the PR Engine opened during the demo; **do not merge it into
  `main`**. The `demo/engine-missing-dates` branch stays broken and ready for the
  next run.

## The fix (for reference / what "after" looks like)

`main` already has the correct line. To show the fixed state live, either check
out `main`, or apply Engine's PR against a throwaway branch:

```python
"published_date": r.get("published_date"),
```
