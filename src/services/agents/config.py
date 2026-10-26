from typing import Any

from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    This is the configuration used by AgenticRAGService for controlling
    graph behavior, retrieval settings, and execution parameters.

    :param max_retrieval_attempts: Maximum number of retrieval attempts before fallback
    :param guardrail_threshold: Threshold score for guardrail validation (0-100)
    :param model: Default model to use for LLM calls (e.g., "llama3.2:1b")
    :param temperature: Temperature for LLM generation (0.0 = deterministic)
    :param top_k: Number of documents to retrieve from search
    :param enable_tracing: Whether to enable Langfuse tracing
    :param enable_live_fetch: Whether the agent may fetch new papers from arXiv on demand
    :param live_fetch_max_results: Max papers to pull per live fetch (each costs a PDF download + parse)
    :param metadata: Additional runtime metadata for tracking and analytics
    :param settings: Application settings instance for environment and service config
    """

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = Field(default_factory=lambda: get_settings().retrieval.top_k)
    enable_live_fetch: bool = True
    live_fetch_max_results: int = 3
    metadata: dict[str, Any] = {}
    settings: Settings = Field(default_factory=get_settings)
