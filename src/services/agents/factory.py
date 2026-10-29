"""Factory for the AgenticRag service (singleton)."""

from functools import lru_cache

from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.embeddings.factory import make_openai_embeddings_client
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.indexing.text_chunker import TextChunker
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.openai_.factory import make_openai_client
from src.services.pdf_parser.factory import make_light_pdf_parser_service
from src.services.pinecone_.factory import make_pinecone_client

from .agentic_rag import AgenticRag
from .config import GraphConfig


@lru_cache(maxsize=1)
def make_agentic_rag(checkpointer=None) -> AgenticRag:
    """Build and return the singleton AgenticRag service.

    Wires all client dependencies (pinecone, embeddings, LLM) via their own
    factories, plus a default GraphConfig. Called once at app startup.
    """
    pinecone = make_pinecone_client()
    embeddings = make_openai_embeddings_client()

    # Built here rather than via make_hybrid_indexing_service(): that factory is
    # uncached and opens a fresh pinecone connection pool on every call, which is
    # wasteful for a service constructed once and reused for the app's lifetime.
    hybrid_indexing_service = HybridIndexingService(
        chunker=TextChunker(),
        embeddings_client=embeddings,
        pinecone_client=pinecone,
    )

    return AgenticRag(
        pinecone=pinecone,
        openaiembeddings=embeddings,
        openai_=make_openai_client(),
        graph_config=GraphConfig(),
        checkpointer=checkpointer,
        metadata_fetcher=make_metadata_fetcher(
            arxiv_client=make_arxiv_client(),
            pdf_parser=make_light_pdf_parser_service(),
        ),
        hybrid_indexing_service=hybrid_indexing_service,
        db_session_factory=make_database().get_session,
    )
