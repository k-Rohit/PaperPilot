"""Pydantic models the agent uses as structured outputs.

Each class below is a SHAPE the LLM must return at one decision point in
the graph. Forcing the LLM into a Pydantic schema means we never parse free-form English text
— we parse validated JSON. Same models also act as type-safe graph state.

Mental model:
    LLM call  ──>  returns JSON in one of these shapes  ──>  graph routes / stores / displays
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """Used by the FIRST node: "Is this query even about arXiv papers?"

    The LLM scores the user's question 0-100 for on-topic-ness. The graph
    compares this score against `guardrail_threshold` (e.g. 60) to decide
    whether to retrieve at all or short-circuit with "I can only answer
    about research papers."

    Example:
        Query: "what's the weather?"  ->  GuardrailScoring(score=15, reason="not research")
        Query: "explain attention"    ->  GuardrailScoring(score=92, reason="ML topic")
    """

    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """Used PER CHUNK by the grading node: "Is THIS chunk relevant?"

    After retrieval brings back N chunks, the grader asks the LLM about
    each one. `Literal["yes", "no"]` forces a binary answer — no "maybe"
    or "kind of." Irrelevant chunks are dropped before the answer is
    generated, so the LLM only sees high-quality context.

    Example:
        Query: "what is multi-head attention?"
        Chunk: "Encoder-decoder seq2seq architecture..."
        LLM returns: GradeDocuments(binary_score="no", reasoning="about seq2seq, not attention")
    """

    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class SourceItem(BaseModel):
    """A rich CITATION returned to the user alongside the final answer.

    The /ask endpoint currently returns a flat list of URLs. SourceItem is
    the structured version the agent returns so the UI can render proper
    citation cards (title + authors + score + link).
    """

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")
    authors: list[str] = Field(default_factory=list, description="list of authors")
    url: str = Field(description="Link to paper")
    relevance_score: float = Field(default=0.0, description="Relevance score from search")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class ToolArtefact(BaseModel):
    """Standard ENVELOPE for any tool call's result.

    LangGraph agents call "tools".
    Every tool wraps its output in ToolArtefact so the graph stores
    consistent shapes regardless of which tool ran. `metadata` carries
    extras like latency, token counts, or the search mode used.

    Example:
        ToolArtefact(
            tool_name="retrieve_chunks",
            tool_call_id="call_abc123",
            content=[<5 chunks>],
            metadata={"latency_ms": 142, "search_mode": "hybrid"},
        )
    """

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")
    content: Any = Field(description="Tool result content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RoutingDecision(BaseModel):
    """The agent's WHERE-DO-I-GO-NEXT decision. Used at every branching node.

    `route` is `Literal[...]` — a white-list of the graph's actual nodes.
    Pydantic rejects any other string, so the LLM physically cannot route
    to a node that doesn't exist (no hallucinated routes).

    Example flow:
        After grading: 0/5 chunks relevant
        LLM returns: RoutingDecision(route="rewrite_query", reason="no good context")
        Graph jumps to the rewrite_query node, retries retrieval.
    """

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(description="Next node to route to")
    reason: str = Field(default="", description="Reason for routing decision")


class GradingResult(BaseModel):
    """The STORED version of a chunk grade (with id + numeric score).

    GradeDocuments = what the LLM RETURNS per chunk (yes/no + why).
    GradingResult  = what the graph STORES per chunk (chunk_id + bool + score + why).

    The graph builds one GradingResult per chunk after each call to the
    grader, so downstream nodes can filter, rank, and trace decisions.
    """

    document_id: str = Field(description="Document identifier")
    is_relevant: bool = Field(description="Relevance flag")
    reasoning: str = Field(default="", description="Grading reasoning")


class QueryRewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(description="The improved query optimized for document retrieval")
    reasoning: str = Field(description="Brief explanation of how the query was improved")


class CondensedQuery(BaseModel):
    standalone_query: str = Field(description="The follow-up rewritten as a standalone question")


class ToolSelection(BaseModel):
    """Which tool should handle this query, chosen by the routing node's LLM.

    Example:
        "how does attention work?"        -> tool="retrieve"
        "find new papers on diffusion"    -> tool="fetch_live_papers", topic="diffusion"
    """

    tool: Literal["retrieve", "fetch_live_papers"]
    topic: str | None = Field(default=None, description="Subject matter, set only for fetch_live_papers")
    reason: str = Field(default="", description="Brief reason for the choice")


class ArxivQueryTranslation(BaseModel):
    """Structured output for translating a user query into an arXiv search query.

    The LLM translates the user's natural language query into a structured
    search query that can be used to fetch relevant papers from arXiv.
    """

    arxiv_search_query: str = Field(description="arXiv API query, e.g. 'all:diffusion AND cat:cs.CV'")
    topic_label: str = Field(description="Short human-readable topic name")
