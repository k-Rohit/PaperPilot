from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(TypedDict):
    """
    State class for the Agentic RAG workflow.

    Tracks all the data that needs to be passed between nodes.

    :cvar messages:
        List of messages in the convo. Uses add_messages reducer
        to append new messages instead of overwrite.

    :type messages: Annotated[list[AnyMessage], add_messages]

    :cvar original_query:
        The original user query before any rewrites/
    :type original_query : Optional[str]

    :cvar rewritten_query:
        The rewritten query after optimization for better retrieval.
    :type rewritten_query: Optional[str]

    :cvar retrieval_attempts:
        Number of retrieval attempts made (for max attempt tracking).
    :type retrieval_attempts: int

    :cvar guardrail_result:
        Result from guardrail validation with score and reasoning.
    :type guardrail_result: Optional[GuardrailScoring]

    :cvar routing_decision:
        The routing decision determining the next node in the graph.
    :type routing_decision: Optional[RoutingDecision]

    :cvar sources:
        Dictionary mapping tool_call_id to their output sources.
    :type sources: Optional[Dict[str, Any]]

    :cvar relevant_sources:
        List of relevant sources to display to the user.
    :type relevant_sources: List[SourceItem]

    :cvar relevant_tool_artefacts:
        List of tool artifacts with metadata from tool executions.
    :type relevant_tool_artefacts: Optional[list[ToolArtefact]]

    :cvar grading_results:
        List of grading results for each retrieved document.
    :type grading_results: List[GradingResult]

    :cvar metadata:
        Runtime metadata for tracing and analytics.
    :type metadata: Dict[str, Any]

    :cvar tool_selection:
        The selected tool for the next action (e.g., "retrieve" or "fetch_live_papers").
    :type tool_selection: Optional[str]

    :cvar target_topic:
        The target topic for live fetching, if applicable.
    :type target_topic: Optional[str]

    :cvar arxiv_search_query:
        The search query for Arxiv, if applicable.
    :type arxiv_search_query: Optional[str]

    :cvar live_fetch_topic_label:
        Human-readable label for the live fetch topic.
    :type live_fetch_topic_label: Optional[str]

    :cvar live_fetch_attempted:
        Flag indicating if a live fetch has been attempted.
    :type live_fetch_attempted: bool

    :cvar live_fetch_log:
        Log of progress lines for live fetch reasoning steps.
    :type live_fetch_log: List[str]

    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: str | None
    rewritten_query: str | None
    retrieval_attempts: int
    guardrail_result: GuardrailScoring | None
    routing_decision: RoutingDecision | None
    sources: dict[str, Any] | None
    relevant_sources: list[SourceItem]
    relevant_tool_artefacts: list[ToolArtefact] | None
    grading_results: list[GradingResult]
    metadata: dict[str, Any]

    # ─── Phase 2: tool routing + live fetch ───
    tool_selection: str | None  # "retrieve" | "fetch_live_papers"  (router writes)
    target_topic: str | None  # subject to fetch                  (router writes)
    arxiv_search_query: str | None  # "all:BERT AND cat:cs.CL"          (translate writes)
    live_fetch_topic_label: str | None  # human label for logs              (translate writes)
    live_fetch_attempted: bool  # guard: fetch at most once per run
    live_fetch_log: list[str]  # progress lines -> reasoning_steps
