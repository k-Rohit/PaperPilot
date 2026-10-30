import logging
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from ..context import Context
from ..models import QueryRewriteOutput
from ..prompts import REWRITE_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)
load_dotenv()


async def rewrite_query(
    state: AgentState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Rewrite the original query for better retrieval using the LLM."""
    logger.info("NODE: rewrite_query")
    start_time = time.time()

    graph_config = runtime.context.graph_config
    original_query = state.get("original_query") or get_latest_query(state["messages"])
    current_attempts = state.get("retrieval_attempts", 0)

    rewritten_query = original_query
    reasoning = ""

    if current_attempts >= graph_config.max_retrieval_attempts:
        # Loop cap already hit — skip the LLM call, just pass through
        return {
            "messages": [HumanMessage(content=rewritten_query)],
            "rewritten_query": rewritten_query,
        }

    try:
        model = ChatOpenAI(model=graph_config.model, temperature=0.4)
        query_rewriter = model.with_structured_output(QueryRewriteOutput)
        prompt = REWRITE_PROMPT.format(question=original_query)

        response: QueryRewriteOutput = await query_rewriter.ainvoke(prompt)

        if not response or not response.rewritten_query:
            raise ValueError("LLM returned no rewritten query")

        rewritten_query = response.rewritten_query.strip()
        reasoning = response.reasoning

        logger.info(f"rewrite: '{original_query[:50]}...' -> '{rewritten_query[:50]}...'")
        logger.debug(f"reasoning: {reasoning}")

    except Exception as e:
        logger.error(f"LLM rewrite failed: {e} — falling back to keyword expansion")
        rewritten_query = f"{original_query} research paper arxiv machine learning"
        reasoning = "Fallback: simple keyword expansion"

    logger.info(f"took={time.time() - start_time:.2f}s")
    return {
        "messages": [HumanMessage(content=rewritten_query)],
        "rewritten_query": rewritten_query,
    }
