"""Unit tests for `RecursionFloorMiddleware`."""

from llm_research_agent.utils.budget import (
    MIN_RECURSION_LIMIT,
    RecursionFloorMiddleware,
)


def test_clamps_below_minimum_to_floor() -> None:
    middleware = RecursionFloorMiddleware()
    config = {"recursion_limit": 1}
    middleware._apply_floor(config)
    assert config["recursion_limit"] == MIN_RECURSION_LIMIT


def test_passes_through_value_at_or_above_minimum() -> None:
    middleware = RecursionFloorMiddleware()
    for value in (MIN_RECURSION_LIMIT, 50, 100):
        config = {"recursion_limit": value}
        middleware._apply_floor(config)
        assert config["recursion_limit"] == value


def test_custom_minimum_is_honored() -> None:
    middleware = RecursionFloorMiddleware(minimum=10)
    config = {"recursion_limit": 1}
    middleware._apply_floor(config)
    assert config["recursion_limit"] == 10


def test_missing_recursion_limit_is_left_untouched() -> None:
    middleware = RecursionFloorMiddleware()
    config: dict = {}
    middleware._apply_floor(config)
    assert "recursion_limit" not in config
