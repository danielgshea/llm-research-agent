"""Step-budget middleware: land a clean final answer instead of looping forever.

The research agent can fail to converge — it keeps searching to nail down exact,
fast-moving figures and never settles into a final answer (we diagnosed a run
that did 38 model turns / 58 web calls before being interrupted). Two guards
address that:

1. A `recursion_limit` on the compiled graph (set in `agent.py`) — a hard
   backstop.
2. This middleware — the *graceful* layer. Rather than let the graph hit the
   recursion limit and raise `GraphRecursionError` (which the user would see and
   which can't be cleanly recovered without a checkpointer), it watches how many
   model turns have happened. Once the budget is nearly spent, it forces ONE
   final model call with the tools removed and an instruction to deliver the best
   answer from the evidence already gathered. The model returns a normal final
   message with no tool calls, the agent loop ends, and the user only ever sees a
   polished, caveated response.

The turn count is read from the message history (number of prior `AIMessage`s),
so it needs no extra state channel and is robust across sync/async execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.config import get_config

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

MIN_RECURSION_LIMIT = 25


class RecursionFloorMiddleware(AgentMiddleware):
    """Clamp the effective `recursion_limit` upward so caller overrides can't starve the agent."""

    def __init__(self, minimum: int = MIN_RECURSION_LIMIT) -> None:
        self.minimum = minimum

    def _apply_floor(self, config: RunnableConfig) -> None:
        """Raise `config['recursion_limit']` to `self.minimum` when present and lower."""
        current = config.get("recursion_limit", self.minimum)
        if current < self.minimum:
            config["recursion_limit"] = self.minimum

    def before_agent(self, state: Any, runtime: Runtime) -> None:
        """Clamp the live runtime config's recursion_limit upward."""
        self._apply_floor(get_config())

    async def abefore_agent(self, state: Any, runtime: Runtime) -> None:
        """Async variant of `before_agent`."""
        self._apply_floor(get_config())


_FINAL_ANSWER_INSTRUCTION = """

## Step budget reached — answer now

You have used most of your research budget for this task. Do NOT call any more
tools. Using only the information you have already gathered, write your best
final answer to the user now:
- Lead with the direct answer to their question.
- Cite the sources you already have, with dates.
- Be explicit about what you could not verify or look up — mark those clearly as
  unverified rather than guessing.
- Note anything that's still uncertain or due for follow-up.
A clear, well-caveated answer from current evidence is required now."""


class StepBudgetMiddleware(AgentMiddleware):
    """Force a clean final answer as the model-turn budget nears exhaustion.

    Args:
        max_model_turns: Number of model turns after which the next model call is
            forced to be a tool-free final answer. Keep comfortably below the
            graph's `recursion_limit` (≈2 graph steps per turn) so this fires
            before the hard limit.
    """

    def __init__(self, max_model_turns: int = 25) -> None:
        self.max_model_turns = max_model_turns

    def _over_budget(self, request: ModelRequest) -> bool:
        turns = sum(1 for m in request.messages if isinstance(m, AIMessage))
        return turns >= self.max_model_turns

    def _force_final(self, request: ModelRequest) -> ModelRequest:
        # Drop tools so the model can't call any, and append a strong instruction
        # to deliver the final answer from what it already has.
        return request.override(
            tools=[],
            system_message=append_to_system_message(
                request.system_message, _FINAL_ANSWER_INSTRUCTION
            ),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if self._over_budget(request):
            request = self._force_final(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self._over_budget(request):
            request = self._force_final(request)
        return await handler(request)
