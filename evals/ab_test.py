"""A/B test models on the llm-research-golden evals.

Holds everything constant (prompt, skills, memory, tools, dataset, and the full
evaluator suite from evals/run_eval.py) and swaps ONLY the chat model, running one
LangSmith experiment per model so you can open the compare view and see which
model wins on each metric — and at what cost.

Models route through the LangSmith LLM Gateway (base_url /<provider>), authed with
the gateway key, so every model call is also traced in your workspace.

Preset matchups:
  anthropic-vs-openai : Anthropic flagship  vs  OpenAI flagship   (cross-vendor)
  openai-cost         : expensive OpenAI    vs  cheap OpenAI      (cost/quality)

Run:
  uv run --env-file .env python evals/ab_test.py anthropic-vs-openai
  uv run --env-file .env python evals/ab_test.py openai-cost

Override any model via env, e.g. AB_OPENAI_EXPENSIVE=gpt-5.1, AB_ANTHROPIC=claude-opus-4-8.
"""

import os
import sys

import dotenv

dotenv.load_dotenv()

from langchain.chat_models import init_chat_model
from langsmith import Client, evaluate

from llm_research_agent.agent import build_agent
from evals.run_eval import DATASET, MAX_CONCURRENCY, build_evaluators, make_target

GATEWAY = "https://gateway.smith.langchain.com"
# The gateway is authed by a LangSmith gateway key; ANTHROPIC_API_KEY is already
# set to that key in this repo's .env, so it's the fallback.
GATEWAY_KEY = os.environ.get("LANGSMITH_API_KEY_GATEWAY") or os.environ.get("ANTHROPIC_API_KEY")

# Model names are overridable via env so you can track provider renames without code changes.
ANTHROPIC = os.environ.get("AB_ANTHROPIC", "claude-sonnet-4-6")
OPENAI_EXPENSIVE = os.environ.get("AB_OPENAI_EXPENSIVE", "gpt-5")
OPENAI_CHEAP = os.environ.get("AB_OPENAI_CHEAP", "gpt-5-mini")

# Each arm: a short label (used in the experiment name + metadata) and a model spec.
MATCHUPS = {
    "anthropic-vs-openai": [
        {"label": "anthropic", "model": ANTHROPIC, "provider": "anthropic"},
        {"label": "openai", "model": OPENAI_EXPENSIVE, "provider": "openai"},
    ],
    "openai-cost": [
        {"label": "openai-expensive", "model": OPENAI_EXPENSIVE, "provider": "openai"},
        {"label": "openai-cheap", "model": OPENAI_CHEAP, "provider": "openai"},
    ],
}


def gateway_chat(model: str, provider: str):
    return init_chat_model(
        model=model,
        model_provider=provider,
        base_url=f"{GATEWAY}/{provider}",
        api_key=GATEWAY_KEY,
    )


def run_matchup(name: str) -> None:
    arms = MATCHUPS[name]
    client = Client()
    ds = client.read_dataset(dataset_name=DATASET)
    print(f"\n=== matchup '{name}' over '{DATASET}' ({ds.id}) ===")

    session_ids = []
    for arm in arms:
        chat_model = gateway_chat(arm["model"], arm["provider"])
        target = make_target(build_agent(chat_model))
        print(f"\n--- arm '{arm['label']}': {arm['provider']}:{arm['model']} ---")
        results = evaluate(
            target,
            data=DATASET,
            evaluators=build_evaluators(),
            experiment_prefix=f"ab-{name}-{arm['label']}",
            metadata={"matchup": name, "provider": arm["provider"], "model": arm["model"]},
            # Serial: arms share one Context Hub /memory backend; concurrent writes 409.
            max_concurrency=MAX_CONCURRENCY,
        )
        exp = getattr(results, "experiment_name", str(results))
        print(f"    experiment: {exp}")
        try:
            session_ids.append(str(client.read_project(project_name=exp).id))
        except Exception as e:  # noqa: BLE001 — compare URL is best-effort
            print(f"    (couldn't resolve session id for compare URL: {e})")

    print(f"\nCompare arms in the LangSmith UI:")
    if len(session_ids) == len(arms):
        sel = "".join(f"&selectedSessions={s}" for s in session_ids)
        print(f"  https://smith.langchain.com/datasets/{ds.id}/compare?{sel.lstrip('&')}")
    else:
        print(f"  Open dataset {ds.id} -> Experiments tab, select the 'ab-{name}-*' runs.")


def main() -> None:
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(MATCHUPS)
        print(f"No matchup given; running all: {', '.join(names)}")
    unknown = [n for n in names if n not in MATCHUPS]
    if unknown:
        sys.exit(f"Unknown matchup(s) {unknown}. Choose from: {', '.join(MATCHUPS)}")
    for name in names:
        run_matchup(name)


if __name__ == "__main__":
    main()
