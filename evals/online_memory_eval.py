"""Online CODE evaluator: did the agent update its /memory on this run?

Upload to LangSmith as an ONLINE evaluator on the `llm-research` tracing project:

    set -a; . ./.env; set +a            # load LANGSMITH_API_KEY (workspace-scoped)
    langsmith evaluator upload evals/online_memory_eval.py \
        --function perform_eval --name memory_updated \
        --project llm-research --sampling-rate 1.0

Online code evaluators run SERVER-SIDE with NO internet access and stdlib only —
so this is pure trace inspection. (The hallucination / correctness judges call an
LLM, so they cannot be code evaluators online; configure those as LLM-as-judge
online evaluators with the `llm-research-hallucination-judge` Prompt Hub prompt.)

It reads the agent root run's output (`{"messages": [...]}`) and flags any
write_file / edit_file tool call targeting `/memory`. Score 100 = the agent
persisted a memory/scorecard update this run, 0 = it did not.
"""


def perform_eval(run):
    # Online code evaluators take exactly one positional arg: the run.
    outputs = (run or {}).get("outputs") or {}
    messages = outputs.get("messages") or []
    wrote = any(
        tc.get("name") in ("write_file", "edit_file")
        and str((tc.get("args") or {}).get("file_path", "")).startswith("/memory")
        for m in messages
        if isinstance(m, dict)
        for tc in (m.get("tool_calls") or [])
    )
    return {"memory_updated": 100 if wrote else 0}
