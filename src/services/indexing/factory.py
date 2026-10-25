from src.services.embeddings.factory import make_openai_embeddings_client
from src.services.pinecone_.factory import make_pinecone_client

from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker


def make_hybrid_indexing_service() -> HybridIndexingService:
    """Factory function to create hybrid indexing service.

    Creates a new service instance each time (not cached — scripts/ingest_papers.py
    calls it once per run). The clients inside are cached singletons.

    :returns: HybridIndexingService instance
    """
    return HybridIndexingService(
        chunker=TextChunker(),  # defaults: 600 words, 100 overlap
        embeddings_client=make_openai_embeddings_client(),
        pinecone_client=make_pinecone_client(),
    )
