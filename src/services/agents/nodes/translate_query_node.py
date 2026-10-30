import logging
import time
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from ..context import Context
from ..models import ArxivQueryTranslation
from ..prompts import ARXIV_QUERY_TRANSLATION_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)
load_dotenv()


async def translate_query_for_arxiv(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Turn a plain-English topic into arXiv API search syntax.

    Runs just before the live-fetch tool call. The router leaves the subject in
    `target_topic`; this node converts it into something the arXiv API understands,
    e.g. "diffusion models" -> "all:diffusion models AND cat:cs.CV".

    Also sets `live_fetch_attempted` — this node is the commit point, so the flag
    goes up even if the fetch itself later returns nothing. That stops the graph
    looping back and fetching a second time in the same run.
    """
    logger.info("NODE: translate_query_for_arxiv")
    start_time = time.time()

    graph_config = runtime.context.graph_config
    topic = state.get("target_topic") or state.get("original_query") or get_latest_query(state["messages"])

    # Fallbacks set up front so an LLM failure degrades instead of crashing.
    # "all:<topic>" is always valid arXiv syntax — a broader search, not a broken one.
    arxiv_search_query = f"all:{topic}"
    topic_label = topic

    try:
        model = ChatOpenAI(model=graph_config.model, temperature=graph_config.temperature)
        translator = model.with_structured_output(ArxivQueryTranslation)
        prompt = ARXIV_QUERY_TRANSLATION_PROMPT.format(topic=topic)

        response: ArxivQueryTranslation = await translator.ainvoke(prompt)

        if not response or not response.arxiv_search_query:
            raise ValueError("LLM returned no arXiv search query")

        arxiv_search_query = response.arxiv_search_query.strip()
        topic_label = (response.topic_label or topic).strip()

        logger.info(f"translated: '{topic}' -> '{arxiv_search_query}'")

    except Exception as e:
        logger.error(f"arXiv query translation failed: {e} — falling back to '{arxiv_search_query}'")

    logger.info(f"took={time.time() - start_time:.2f}s")
    return {
        "arxiv_search_query": arxiv_search_query,
        "live_fetch_topic_label": topic_label,
        "live_fetch_attempted": True,
        "retrieval_attempts": 0,
    }
