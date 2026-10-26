import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.repositories.paper import PaperRepository
from src.services.embeddings.openai_client import OpenAIEmbeddingsClient
from src.services.indexing.hybrid_indexer import HybridIndexingService
from src.services.metadata_fetcher import MetadataFetcher
from src.services.pinecone_.client import PineconeClient

logger = logging.getLogger(__name__)


def create_retriever_tool(pinecone_client: PineconeClient, embeddings_client: OpenAIEmbeddingsClient, top_k: int):
    """Create a retriever tool that wraps Pinecone hybrid (sparse + dense) search.

    :param pinecone_client: Existing Pinecone service
    :param embeddings_client: Existing OpenAI embeddings service
    :param top_k: Number of chunks to retrieve
    :returns: LangChain tool for retrieving papers
    """

    @tool(response_format="content_and_artifact")
    async def retrieve_papers(query: str) -> tuple[str, list[Document]]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
            - Machine learning concepts or techniques
            - Deep learning architectures
            - Natural language processing
            - Computer vision methods
            - AI research topics
            - Specific algorithms or models

            :param query: The search query describing what papers to find
            :returns: List of relevant paper excerpts with metadata
        """
        logger.info(f"Retrieving papers for the query: {query[:100]}")
        logger.debug(f"top_k: {top_k}")

        # Generate query embeddings
        logger.debug("Generating query embedding")
        query_embeddings = await embeddings_client.embed_text(text=query)
        logger.debug(f"Generated embedding with {len(query_embeddings)} dimensions")

        # Search using Pinecone
        logger.debug("Searching Pinecone")
        hits = pinecone_client.search(query=query, query_embedding=query_embeddings, top_k=top_k)

        # Convert Pinecone hits to LangChain Documents
        documents = []
        logger.info(f"Found {len(hits)} documents from Pinecone")

        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": hit.get("pdf_url") or f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_title", ""),
                    "search_mode": "hybrid",
                    "top_k": top_k,
                },
            )
            documents.append(doc)
        logger.debug(f"Converted {len(documents)} hits to Langchain Documents.")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        # content = what the grading/generation LLMs read. Previously the tool
        # returned raw `documents` and LangChain stringified the whole list via
        # str(), which happened to dump metadata (arxiv_id, title) as text too
        # (that's how citations like "arXiv:2607.28623v1" ended up in answers).
        # Keep that same citation info here explicitly, instead of relying on
        # an incidental repr.
        # documents = kept as the ToolMessage artifact so downstream nodes can
        # access structured metadata directly instead of re-parsing text.
        content = "\n\n".join(
            f"[{doc.metadata.get('title', '')} (arXiv:{doc.metadata.get('arxiv_id', '')})]\n{doc.page_content}"
            for doc in documents
        )
        return content, documents

    return retrieve_papers


def create_live_fetch_tool(
    metadata_fetcher: MetadataFetcher,
    hybrid_indexing_service: HybridIndexingService,
    db_session_factory,
    default_max_results: int = 5,
):
    """Create a tool that fetches NEW papers from arXiv on demand and indexes them.

    Fills corpus gaps: the daily ingestion job only ingests a fixed window, so a question
    about a topic outside that window has nothing to retrieve. This runs the same
    fetch -> parse -> store pipeline synchronously, then indexes into Pinecone so the
    new papers are immediately searchable.

    :param metadata_fetcher:        Ingestion pipeline (arXiv fetch + PDF parse + Postgres)
    :param hybrid_indexing_service: Long-lived indexer (chunk + embed + Pinecone)
    :param db_session_factory:      Zero-arg callable returning a Session context manager
    :param default_max_results:     Papers to fetch per call when caller doesn't specify
    :returns: LangChain tool for live-fetching papers
    """

    @tool(response_format="content_and_artifact")
    async def fetch_live_papers(
        arxiv_search_query: str,
        topic_label: str,
        max_results: int = default_max_results,
    ) -> tuple[str, dict[str, Any]]:
        """Fetch brand-new papers from arXiv on a topic and add them to the corpus.

        Use this tool ONLY when the user wants papers the corpus doesn't have yet:
            - Asking to find/get/fetch new or recent papers on a topic
            - Asking about a paper or topic that isn't in the local corpus

        Do NOT use this for ordinary questions answerable from existing papers —
        this is slow (downloads and parses PDFs) and hits the arXiv API.

        :param arxiv_search_query: arXiv API query syntax, e.g. "all:BERT AND cat:cs.CL"
        :param topic_label:        Short human-readable topic name, for logs and progress
        :param max_results:        Maximum papers to fetch
        :returns: Summary text plus stats including the newly indexed arxiv_ids
        """
        logger.info(f"Live fetch | topic={topic_label!r} query={arxiv_search_query!r} max={max_results}")

        with db_session_factory() as session:
            stats = await metadata_fetcher.fetch_and_process_by_topic(
                search_query=arxiv_search_query,
                max_results=max_results,
                process_pdfs=True,
                store_to_db=True,
                db_session=session,
            )

            new_arxiv_ids = stats.get("new_arxiv_ids", [])
            artifact: dict[str, Any] = {
                "topic_label": topic_label,
                "arxiv_search_query": arxiv_search_query,
                "papers_fetched": stats.get("papers_fetched", 0),
                "papers_skipped_existing": stats.get("papers_skipped_existing", 0),
                "papers_stored": stats.get("papers_stored", 0),
                "papers_indexed": 0,
                "new_arxiv_ids": new_arxiv_ids,
                "errors": stats.get("errors", []),
            }

            if not new_arxiv_ids:
                content = (
                    f"No new papers were added for '{topic_label}'. "
                    f"Found {artifact['papers_fetched']} on arXiv, "
                    f"{artifact['papers_skipped_existing']} of which are already in the corpus."
                )
                logger.info(content)
                return content, artifact

            # Re-read the rows we just stored — they now carry the parsed raw_text and
            # the DB-assigned id that the indexer needs.
            #
            # A paper can be stored with raw_text=NULL when Docling refused the PDF
            # (over 30 pages or 20MB — see PDFParserSettings). Those papers produce zero
            # chunks, so they are NOT searchable no matter what the indexer reports.
            # Split them out here so the summary can't claim a paper is available when
            # retrieval will never find it.
            paper_repo = PaperRepository(session)
            indexable, unparseable = [], []
            for arxiv_id in new_arxiv_ids:
                paper = paper_repo.get_by_arxiv_id(arxiv_id)
                if paper is None:
                    logger.warning(f"Stored paper {arxiv_id} not found on re-read — skipping indexing")
                    continue
                if not paper.raw_text:
                    unparseable.append(paper.title)
                    continue
                indexable.append(
                    {
                        "id": str(paper.id),
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "categories": paper.categories,
                        "published_date": paper.published_date,
                        "raw_text": paper.raw_text,
                        "sections": paper.sections,
                    }
                )

            if indexable:
                index_stats = await hybrid_indexing_service.index_papers_batch(
                    papers=indexable,
                    replace_existing=False,
                )
                artifact["chunks_indexed"] = index_stats.get("total_chunks_indexed", 0)
            else:
                artifact["chunks_indexed"] = 0

            # Only count a paper as indexed if it actually produced searchable chunks.
            artifact["papers_indexed"] = len(indexable) if artifact["chunks_indexed"] else 0
            artifact["papers_unparseable"] = len(unparseable)

        if not artifact["papers_indexed"]:
            content = (
                f"Found {artifact['papers_fetched']} paper(s) on '{topic_label}' but could not make "
                f"any of them searchable — their PDFs could not be parsed (too large or too long). "
                f"Answer from existing knowledge instead; do not claim these papers are available."
            )
            logger.warning(f"Live fetch produced no searchable papers | topic={topic_label!r}")
            return content, artifact

        titles = ", ".join(p["title"] for p in indexable)
        content = (
            f"Fetched and indexed {artifact['papers_indexed']} new paper(s) on '{topic_label}' "
            f"({artifact['chunks_indexed']} chunks, now searchable). "
            f"{artifact['papers_skipped_existing']} were already in the corpus. "
            f"New papers: {titles}"
        )
        if unparseable:
            content += f" (Skipped {len(unparseable)} unparseable PDF(s).)"
        logger.info(f"✓ Live fetch complete | {content}")
        return content, artifact

    return fetch_live_papers
