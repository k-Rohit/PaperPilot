"""RAG Q&A endpoint — retrieve relevant chunks, ground an LLM, return an answer."""

import logging

from fastapi import APIRouter, HTTPException
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.dependencies import CacheDep, EmbeddingsDep, LLMDep, PineconeDep
from src.schemas.api.ask import AskRequest, AskResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


def _trace_inputs(inputs: dict) -> dict:
    """Record only the request payload in LangSmith — the injected DI clients
    (pinecone, embedder, llm, cache) aren't serializable and pollute the trace."""
    req = inputs.get("request")
    if req is not None and hasattr(req, "model_dump"):
        return req.model_dump(exclude_none=True)
    return {}


@router.post("/ask", response_model=AskResponse)
@traceable(name="ask_request", run_type="chain", process_inputs=_trace_inputs)
async def ask(
    request: AskRequest,
    pinecone: PineconeDep,
    embedder: EmbeddingsDep,
    llm: LLMDep,
    cache: CacheDep,
) -> AskResponse:
    """Answer a question grounded in retrieved arXiv paper chunks.

    0. Check exact-match cache; return cached answer if present.
    1. Embed the user query.
    2. Retrieve top-K relevant chunks from Pinecone.
    3. Build a prompt with chunks as context, call the LLM.
    4. Store the response in cache and return.
    """
    run_tree = get_current_run_tree()
    if run_tree and request.thread_id:
        # session_id metadata is what LangSmith's Threads view groups runs by
        run_tree.add_metadata({"session_id": request.thread_id})

    # 0. Cache lookup (exact match on request params)
    if cache is not None:
        cached = await cache.find_cached_response(request)
        if cached is not None:
            return cached

    try:
        query_embedding = await embedder.embed_text(request.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {e}") from e

    # 2. Retrieve top-K chunks
    logger.info(f"Retrieving top {request.top_k} chunks for query: '{request.query[:60]}'")
    hits = pinecone.search(query=request.query, query_embedding=query_embedding, top_k=request.top_k)

    if not hits:
        raise HTTPException(status_code=404, detail="No relevant papers found for this query.")

    # 3. Build chunks for the LLM (minimal payload — text + arxiv_id only)
    chunks_for_llm = [
        {
            "arxiv_id": hit.get("arxiv_id", ""),
            "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
        }
        for hit in hits
    ]

    # 4. Call the LLM
    try:
        rag_result = await llm.generate_rag_response(query=request.query, chunks=chunks_for_llm)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}") from e

    # 5. Build the response — extract unique source URLs from hits
    sources = list({hit.get("pdf_url", "") for hit in hits if hit.get("pdf_url")})

    response = AskResponse(
        query=request.query,
        answer=rag_result.get("answer", ""),
        sources=sources,
        chunks_used=len(hits),
        search_mode="hybrid",
    )

    # 6. Store in cache for next time (fire-and-forget; cache failures don't break the response)
    if cache is not None:
        await cache.store_response(request, response)

    return response
