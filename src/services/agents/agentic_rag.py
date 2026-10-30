import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.embeddings.openai_client import OpenAIEmbeddingsClient
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.metadata_fetcher import MetadataFetcher
from src.services.openai_ import OpenAIClient
from src.services.pinecone_.client import PineconeClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_condense_followup,
    ainvoke_generate_answer,
    ainvoke_grade_retrieved_chunks,
    ainvoke_out_of_scope_step,
    finalize_live_fetch,
    initiate_live_fetch,
    initiate_retrieve,
    rewrite_query,
    route,
    route_after_grading,
    route_after_tool,
    route_after_tool_selection,
    score_user_query,
    select_tool,
    translate_query_for_arxiv,
)
from .state import AgentState
from .tools import create_live_fetch_tool, create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRag:
    """Agentic RAG service.

    Wires the LangGraph workflow: guardrail → retrieve → grade → (rewrite or generate) → END.
    """

    def __init__(
        self,
        pinecone: PineconeClient,
        openaiembeddings: OpenAIEmbeddingsClient,
        openai_: OpenAIClient,
        graph_config: GraphConfig,
        checkpointer=None,
        metadata_fetcher: MetadataFetcher | None = None,
        hybrid_indexing_service: HybridIndexingService | None = None,
        db_session_factory=None,
    ):
        """Initialize agentic RAG service.

        :param pinecone:              Client for document search
        :param openaiembeddings:        Client for embeddings
        :param openai_:                 OpenAI client for generation
        :param graph_config:            Configuration for graph execution
        :param checkpointer:            LangGraph checkpointer for conversational memory
        :param metadata_fetcher:        Ingestion pipeline, used by the live-fetch tool
        :param hybrid_indexing_service: Chunk/embed/index service, used by the live-fetch tool
        :param db_session_factory:      Zero-arg callable returning a DB session context manager
        """
        self.pinecone = pinecone
        self.openaiembeddings = openaiembeddings
        self.openai_ = openai_
        self.graph_config = graph_config
        self.checkpointer = checkpointer
        self.metadata_fetcher = metadata_fetcher
        self.hybrid_indexing_service = hybrid_indexing_service
        self.db_session_factory = db_session_factory
        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model:                  {self.graph_config.model}")
        logger.info(f"  Top-k:                  {self.graph_config.top_k}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold:    {self.graph_config.guardrail_threshold}")

        self.graph = self._build_graph()
        logger.info("✓ AgenticRAGService initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""
        logger.info(f"Building LangGraph workflow with context_schema, {Context}")
        workflow = StateGraph(AgentState, context_schema=Context)

        # Tool the LLM calls to retrieve chunks
        retriever_tool = create_retriever_tool(
            pinecone_client=self.pinecone,
            embeddings_client=self.openaiembeddings,
            top_k=self.graph_config.top_k,
        )
        live_fetch_tool = create_live_fetch_tool(
            metadata_fetcher=self.metadata_fetcher,
            hybrid_indexing_service=self.hybrid_indexing_service,
            db_session_factory=self.db_session_factory,
            default_max_results=self.graph_config.live_fetch_max_results,
        )
        tools = [retriever_tool, live_fetch_tool]

        logger.info("Adding nodes to workflow graph")
        workflow.add_node("condense_followup_node", ainvoke_condense_followup)
        workflow.add_node("guardrail_node", score_user_query)
        workflow.add_node("retrieve_node", initiate_retrieve)
        workflow.add_node("out_of_scope_node", ainvoke_out_of_scope_step)
        workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_document_node", ainvoke_grade_retrieved_chunks)
        workflow.add_node("rewrite_query_node", rewrite_query)
        workflow.add_node("generate_answer_node", ainvoke_generate_answer)
        workflow.add_node("translate_query_node", translate_query_for_arxiv)
        workflow.add_node("tool_router_node", select_tool)
        workflow.add_node("live_fetch_call_node", initiate_live_fetch)
        workflow.add_node("live_fetch_summary_node", finalize_live_fetch)

        logger.info("Configuring graph edges and routing logic")

        ## Adding edges
        # START → condense_followup if there are multiple human messages, otherwise skip to guardrail
        workflow.add_edge(START, "condense_followup_node")
        workflow.add_edge("condense_followup_node", "guardrail_node")

        # guardrail → retrieve OR out_of_scope
        workflow.add_conditional_edges(
            "guardrail_node",
            route,
            {
                "continue": "tool_router_node",
                "out_of_scope": "out_of_scope_node",
            },
        )

        # out_of_scope → END
        workflow.add_edge("out_of_scope_node", END)

        workflow.add_conditional_edges(
            "tool_router_node",
            route_after_tool_selection,
            {
                "retrieve_node": "retrieve_node",
                "translate_query_node": "translate_query_node",
            },
        )
        # live fetch path: translate the topic → build the tool call → run it
        workflow.add_edge("translate_query_node", "live_fetch_call_node")
        workflow.add_edge("live_fetch_call_node", "tool_retrieve")

        # after the fetch → search the papers we just indexed, then answer from their
        # actual content. Going straight to generate_answer_node would leave the LLM
        # with only the fetch summary (titles + counts) and no paper text or sources.
        workflow.add_edge("live_fetch_summary_node", "retrieve_node")

        # retrieve → tool_retrieve (via tool_calls) OR END (max attempts hit)
        workflow.add_conditional_edges(
            "retrieve_node",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )

        # tool_retrieve now runs 2 tools — send the result to the right handler
        workflow.add_conditional_edges(
            "tool_retrieve",
            route_after_tool,
            {
                "grade_document_node": "grade_document_node",
                "live_fetch_summary_node": "live_fetch_summary_node",
            },
        )

        # grade → generate, rewrite or do a live fetch (based on state["routing_decision"])
        workflow.add_conditional_edges(
            "grade_document_node",
            route_after_grading,
            {
                "generate_answer": "generate_answer_node",
                "rewrite_query": "rewrite_query_node",
                "live_fetch": "translate_query_node",
            },
        )

        # rewrite → retrieve (loop back)
        workflow.add_edge("rewrite_query_node", "retrieve_node")

        # generate → END
        workflow.add_edge("generate_answer_node", END)

        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        logger.info("✓ Graph compilation successful")

        return compiled_graph

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        thread_id: str | None = None,
    ) -> dict:
        """Run the agentic graph end-to-end for one user query.

        :param query:     The user's question
        :param user_id:   Optional user identifier (for tracing / logs)
        :param thread_id: Conversation id — same id across calls lets the
                           checkpointer restore prior turns. The frontend
                           generates and sends one on every request; a missing
                           thread_id here means a caller integration forgot to
                           wire it through, so this raises rather than masking
                           it with a throwaway random id.
        :returns:         Dict with query, answer, sources, retrieval_attempts
        :raises ValueError: if query is empty, or thread_id is missing
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not thread_id:
            raise ValueError(
                "thread_id is required — the graph is checkpointed and needs a "
                "conversation id to load/save state. The frontend already sends "
                "one on every request; a missing thread_id means the caller "
                "isn't passing it through."
            )

        # Initial state — every AgentState field explicitly set
        state_input: AgentState = {
            "messages": [HumanMessage(content=query)],
            "retrieval_attempts": 0,
            "guardrail_result": None,
            "routing_decision": None,
            "grading_results": [],
            "original_query": None,
            "rewritten_query": None,
            "sources": None,
            "relevant_sources": [],
            "relevant_tool_artefacts": None,
            "metadata": {"user_id": user_id},
            "tool_selection": None,
            "target_topic": None,
            "arxiv_search_query": None,
            "live_fetch_topic_label": None,
            "live_fetch_attempted": False,
            "live_fetch_log": [],
        }

        # Runtime dependencies bundled for every node
        runtime_context = Context(
            pinecone_client=self.pinecone,
            embeddings_client=self.openaiembeddings,
            openai_client=self.openai_,
            graph_config=self.graph_config,
        )

        logger.info(f"Invoking graph for user_id={user_id} thread_id={thread_id} query={query[:80]!r}")
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.graph.ainvoke(state_input, context=runtime_context, config=config)

        return {
            "query": query,
            "answer": self._extract_answer(result),
            "sources": self._extract_sources(result),
            "reasoning_steps": self._extract_reasoning_steps(result),
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "grading_results": result.get("grading_results", []),
        }

    @staticmethod
    def _extract_answer(result: dict) -> str:
        """Walk messages backwards to find the final AIMessage from the agent."""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "No answer generated."

    @staticmethod
    def _extract_sources(result: dict) -> list:
        """
        This method is used for extracting the sources from the generated answer
        """
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    @staticmethod
    def _extract_reasoning_steps(result: dict) -> list[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

        # Live-fetch progress, written by finalize_live_fetch. Listed before retrieval
        # because the fetch happens first when it happens at all.
        steps.extend(result.get("live_fetch_log", []))

        if retrieval_attempts > 0:
            steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        relevant_count = sum(1 for g in grading_results if g.is_relevant)
        if grading_results:
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        # Name the step that actually ran. Appending "Generated answer from context"
        # unconditionally claimed a generation even when the graph ended on the
        # out-of-scope refusal or the retrieval-exhausted fallback.
        if retrieval_attempts == 0 and not grading_results:
            steps.append("Declined — outside the scope of the paper corpus")
        elif grading_results and relevant_count == 0:
            steps.append("No relevant papers found — returned a fallback message")
        else:
            steps.append("Generated answer from context")

        return steps

    def visualize(self, format: str = "mermaid", save_to: str | None = None) -> str | bytes:
        """Visualize the compiled graph in mermaid, png, or ascii form.

        :param format:  One of "mermaid", "png", "ascii"
        :param save_to: Optional path to save the output (e.g. "graph.png")
        :returns:       str for "mermaid"/"ascii", bytes for "png"
        :raises ValueError: for unknown formats
        """
        drawers = {
            "mermaid": lambda g: g.draw_mermaid(),
            "png": lambda g: g.draw_mermaid_png(),
            "ascii": lambda g: g.draw_ascii(),
        }
        if format not in drawers:
            raise ValueError(f"Unknown format {format!r}. Use one of: {list(drawers)}")

        logger.info("Generating graph visualization:")
        try:
            output = drawers[format](self.graph.get_graph())
            if save_to:
                mode = "wb" if isinstance(output, bytes) else "w"
                with open(save_to, mode) as f:
                    f.write(output)
                logger.info(f"✓ Saved {format} visualization to {save_to}")
            else:
                logger.info(f"✓ Generated {format} visualization")
            return output
        except Exception as e:
            logger.error(f"Failed to generate {format} visualization: {e}")
            raise
