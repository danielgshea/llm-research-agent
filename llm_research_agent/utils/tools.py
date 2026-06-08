"""Web research tools for the LLM research agent.

The agent researches the fast-moving large language model landscape — which
models lead, how benchmarks are run and whether they're trustworthy, and who the
major labs are. Its primary instrument is Tavily web search, plus a page-fetch
helper for reading leaderboards, model cards, papers, and announcements in full.

Both tools return *sourced* results (title + URL + content) so the agent can cite
every claim with a link and a date — essential when assessing benchmark validity
and dating fast-changing standings.
"""

import json
import os
from typing import Literal

from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(
    query: str,
    topic: Literal["general", "news", "finance"] = "general",
    max_results: int = 8,
) -> str:
    """Search the web for current information about LLMs, benchmarks, and labs.

    Use this as your primary discovery tool: finding new model releases,
    benchmark leaderboards (LMArena, Artificial Analysis, SWE-bench, GPQA, etc.),
    evaluation methodology, pricing, and lab announcements. Prefer `topic="news"`
    for recent releases and `topic="general"` for reference material and
    leaderboards.

    Returns multiple results so you can corroborate claims across sources and
    cite each one. ALWAYS keep the URLs so you can cite them and record sources
    in the scorecard.

    Args:
        query: The search query (e.g. "GPT-5.5 vs Claude Opus 4.8 SWE-bench score 2026").
        topic: Tavily search topic. Use "news" for recency-sensitive queries.
        max_results: How many results to return (default 8).

    Returns:
        A JSON string: a list of {title, url, published_date, content} objects,
        ranked by relevance.
    """
    response = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=False,
        topic=topic,
    )
    results = [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "published_date": r.get("date"),
            "content": r.get("content"),
        }
        for r in response.get("results", [])
    ]
    if not results:
        return f"No results found for query: {query!r}"
    return json.dumps(results, indent=2, ensure_ascii=False)


def fetch_page(url: str) -> str:
    """Fetch the full content of a single page (leaderboard, model card, paper, post).

    Use this after `web_search` when you need the complete text of a specific
    source — e.g. to read a benchmark's methodology section, a model card's
    eval table, or a release announcement — rather than the short snippet search
    returns.

    Args:
        url: The URL to extract (e.g. an Artificial Analysis or SWE-bench page).

    Returns:
        The extracted page content as text, or an error note if extraction fails.
    """
    response = tavily_client.extract(url, include_images=False)
    results = response.get("results", [])
    if not results:
        failed = response.get("failed_results", [])
        return f"Could not extract content from {url}. {failed}"
    return results[0]["raw_content"]
