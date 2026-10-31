"""Agentic RAG Q&A endpoint — LangGraph workflow with self-correction loop.

Flow: guardrail → retrieve → grade → (rewrite loop OR generate) → END.
Every LangGraph node + LLM call auto-traced in LangSmith when
LANGCHAIN_TRACING_V2=true is set.
"""

import logging

from fastapi import APIRouter, HTTPException
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.dependencies import AgenticRagDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic-ask"])


def _trace_inputs(inputs: dict) -> dict:
    """Record only the request payload in LangSmith — the injected agent
    isn't serializable and pollutes the trace inputs."""
    req = inputs.get("request")
    if req is not None and hasattr(req, "model_dump"):
        return req.model_dump(exclude_none=True)
    return {}


@router.post("/agentic_ask", response_model=AgenticAskResponse)
@traceable(name="agentic_ask_request", run_type="chain", process_inputs=_trace_inputs)
async def agentic_ask(
    request: AskRequest,
    agent: AgenticRagDep,
) -> AgenticAskResponse:
    """Answer a question via the agentic RAG loop.

    Unlike `/ask` (one-shot retrieve → LLM), this endpoint:
      1. Guardrails the query — refuses off-topic questions
      2. Retrieves chunks via the LangChain tool
      3. Grades relevance with an LLM
      4. Optionally rewrites the query and retries (up to N attempts)
      5. Generates the final answer from relevant chunks
    """
    logger.info(f"agentic_ask | query={request.query[:80]!r}")

    run_tree = get_current_run_tree()
    if run_tree and request.thread_id:
        # session_id metadata is what LangSmith's Threads view groups runs by
        run_tree.add_metadata({"session_id": request.thread_id})

    try:
        result = await agent.ask(query=request.query, thread_id=request.thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}")

    sources_detailed = result.get("sources", [])
    sources_urls = [s["url"] for s in sources_detailed if isinstance(s, dict) and s.get("url")]

    return AgenticAskResponse(
        query=result["query"],
        answer=result["answer"],
        sources=sources_urls,  # flat URLs (schema compat)
        sources_detailed=sources_detailed,  # rich objects (new)
        reasoning_steps=result.get("reasoning_steps", []),
        chunks_used=result.get("retrieval_attempts", 0),
        search_mode="agentic",
    )
