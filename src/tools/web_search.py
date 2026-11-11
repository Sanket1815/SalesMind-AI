"""
Web search and page-fetch utilities used by the Research Agent.

Priority order:
1. Tavily (if TAVILY_API_KEY is set) — structured results, good for business intel
2. DuckDuckGo (duckduckgo-search) — no key required, best effort
"""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    body: Optional[str] = None


# --------------------------------------------------------------------------- #
# Tavily search (preferred)
# --------------------------------------------------------------------------- #

def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.tavily_api_key)
    raw = client.search(query=query, max_results=max_results, include_raw_content=False)
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
        )
        for r in raw.get("results", [])
    ]


# --------------------------------------------------------------------------- #
# DuckDuckGo search (fallback)
# --------------------------------------------------------------------------- #

def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
            )
    return results


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Run a web search and return structured results."""
    logger.debug("Web search", extra={"query": query, "max_results": max_results})
    try:
        if settings.tavily_api_key:
            return _search_tavily(query, max_results)
        return _search_duckduckgo(query, max_results)
    except Exception as exc:
        logger.warning("Search failed, returning empty", extra={"error": str(exc)})
        return []


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def fetch_page_text(url: str, max_chars: int = 4000) -> str:
    """Fetch a webpage and return its visible text (truncated)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SalesMindBot/1.0)"}
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except Exception as exc:
        logger.warning("Page fetch failed", extra={"url": url, "error": str(exc)})
        return ""


def company_research_queries(company_name: str, website: Optional[str]) -> list[str]:
    """Generate a set of targeted search queries for a company."""
    base = company_name.strip()
    return [
        f'"{base}" company overview funding employees',
        f'"{base}" B2B software technology stack',
        f'"{base}" news 2024 2025',
        f'"{base}" CEO leadership team site:linkedin.com OR site:crunchbase.com',
    ]
