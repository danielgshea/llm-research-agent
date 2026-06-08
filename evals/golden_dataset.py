"""Pull the `llm-research-golden` dataset from LangSmith (the source of truth).

The dataset lives on LangSmith and is the canonical benchmark: its questions
(inputs) and reference answers (outputs) are maintained there, not hardcoded here.
Reference outputs are the agent's own vetted answers, so evals/run_eval.py grades
future changes against today's behavior.

This script reads the live examples and writes a git-trackable snapshot
(evals/golden_snapshot.json) so dataset drift shows up in a diff.

Schema (matches what evals/run_eval.py reads):
- inputs            : {"messages": [{"role": "user", "content": <question>}]}
- reference outputs : {"messages": [{"type": "ai", "content": <reference answer>}]}

Run:  uv run --env-file .env python evals/golden_dataset.py
"""

import json
import os
import pathlib

from langsmith import Client

DATASET = os.environ.get("EVAL_DATASET", "llm-research-golden")
SNAPSHOT = pathlib.Path(__file__).resolve().parent / "golden_snapshot.json"


def _question(inputs: dict) -> str:
    msgs = (inputs or {}).get("messages") or [{}]
    return msgs[-1].get("content", "")


def _answer(outputs: dict) -> str:
    msgs = (outputs or {}).get("messages") or [{}]
    return msgs[-1].get("content", "")


def main() -> None:
    client = Client()
    ds = client.read_dataset(dataset_name=DATASET)

    examples = sorted(client.list_examples(dataset_id=ds.id), key=lambda e: str(e.id))
    snapshot = [
        {"id": str(e.id), "inputs": e.inputs, "outputs": e.outputs}
        for e in examples
    ]
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str))

    print(f"Pulled {len(snapshot)} examples from '{DATASET}' ({ds.id}).")
    for e in examples:
        q, a = _question(e.inputs), _answer(e.outputs)
        print(f"  - {q[:70]!r} -> reference {len(a)} chars")
    print(f"\nSnapshot: {SNAPSHOT}")
    print(f"View: https://smith.langchain.com/datasets/{ds.id}")


if __name__ == "__main__":
    main()
