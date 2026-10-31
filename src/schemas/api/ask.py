from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings


class AskRequest(BaseModel):
    """Request model for RAG question answering."""

    query: str = Field(
        ...,
        description="The user's question or query.",
        min_length=1,
        max_length=1000,
    )
    top_k: int = Field(
        default_factory=lambda: get_settings().retrieval.top_k,
        description="Number of top relevant documents to retrieve for answering the query.",
        ge=1,
        le=10,
    )
    thread_id: str | None = Field(
        None,
        description="Client-generated conversation id. Stamped as session_id on the "
        "LangSmith trace so all turns of one chat group into a single Thread.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 5,
                "categories": ["cs.AI", "cs.LG"],
            }
        }
    )


class AskResponse(BaseModel):
    """Response model for RAG question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from LLM")
    sources: list[str] = Field(..., description="PDF URLs of source papers")
    chunks_used: int = Field(..., description="Number of chunks used for generation")
    search_mode: str = Field(..., description="Search mode used: bm25 or hybrid")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture...",
                "sources": [
                    "https://arxiv.org/pdf/1706.03762.pdf",
                    "https://arxiv.org/pdf/1810.04805.pdf",
                ],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }
    )


class AgenticAskResponse(AskResponse):
    """Response model for the agentic RAG endpoint.

    Extends AskResponse with agent-specific fields (rich sources + reasoning steps).
    Existing `sources: List[str]` field stays for backward compatibility.
    """

    sources_detailed: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rich source citations (arxiv_id, title, authors, url, relevance_score).",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of what the agent did (guardrail, retrieve, grade, generate).",
    )
