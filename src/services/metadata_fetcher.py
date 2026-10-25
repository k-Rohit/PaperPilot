import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from src.config import Settings, get_settings
from src.exceptions import MetadataFetchingException, PipelineException
from src.repositories.paper import PaperRepository
from src.schemas.arxiv.paper import ArxivPaper, PaperCreate
from src.schemas.pdf_parser.models import ArxivMetadata, ParsedPaper
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """Service for fetching arXiv papers with PDF processing and database storage."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        pdf_cache_dir: Path | None = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        settings: Settings | None = None,
    ):
        """Initialize metadata fetcher with services and settings.

        :param arxiv_client: Client for arXiv API operations
        :param pdf_parser: Service for parsing PDF documents
        :param pdf_cache_dir: Directory for caching downloaded PDFs
        :param max_concurrent_downloads: Maximum concurrent PDF downloads
        :param max_concurrent_parsing: Maximum concurrent PDF parsing operations
        :param settings: Application settings instance
        """
        from src.config import get_settings

        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()

    async def fetch_and_process_papers(
        self,
        max_results: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Session | None = None,
    ) -> dict[str, Any]:
        """
        Fetch papers from arXiv, process PDFs, and store to database.

        :param max_results: Maximum papers to fetch
        :param from_date: Filter papers from this date (YYYYMMDD)
        :param to_date: Filter papers to this date (YYYYMMDD)
        :param process_pdfs: Whether to download and parse PDFs
        :param store_to_db: Whether to store results in database
        :param db_session: Database session (required if store_to_db=True)
        :returns: Dictionary with processing results and statistics
        """
        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "papers_indexed": 0,
            "errors": [],
            "processing_time": 0,
            "arxiv_ids": [],
        }

        start_time = datetime.now()

        try:
            # Step 1: fetch metadata from arXiv
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results,
                from_date=from_date,
                to_date=to_date,
                sort_by="submittedDate",
                sort_order="descending",
            )
            for paper in papers:
                results["arxiv_ids"].append(paper.arxiv_id)
            results["papers_fetched"] = len(papers)
            if not papers:
                logger.warning("No papers found")
                return results

            # Step 2: process PDFs if requested
            pdf_results = {}
            if process_pdfs:
                pdf_results = await self._process_pdfs_batch(papers)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            # Step 3: store to database if requested
            if store_to_db and db_session:
                logger.info("Storing papers to database...")
                stored_count = self._store_papers_to_db(
                    papers,
                    pdf_results.get("parsed_papers", {}),
                    db_session,
                )
                results["papers_stored"] = stored_count
            elif store_to_db:
                logger.warning("Database storage requested but no session provided")
                results["errors"].append("Database session not provided for storage")

            # Step 4: bookkeeping
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time

            logger.info(
                f"Pipeline completed in {processing_time:.1f}s: "
                f"{results['papers_fetched']} papers, "
                f"{results['pdfs_downloaded']} PDFs, "
                f"{len(results['errors'])} errors"
            )

            if results["errors"]:
                logger.warning("Errors summary:")
                for i, error in enumerate(results["errors"][:5], 1):
                    logger.warning(f"  {i}. {error}")
                if len(results["errors"]) > 5:
                    logger.warning(f"  ... and {len(results['errors']) - 5} more errors")

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            results["errors"].append(f"Pipeline error: {str(e)}")
            raise PipelineException(f"Pipeline execution failed: {e}") from e

    async def fetch_and_process_by_topic(
        self,
        search_query: str,
        max_results: int = 5,
        process_pdfs: bool = True,
        store_to_db: bool = True,
        db_session: Session | None = None,
    ) -> dict[str, Any]:
        """
        Fetch papers from arXiv by topic (not date range), process PDFs, and store to database.

        Same pipeline as fetch_and_process_papers, with two differences: it searches by
        an arXiv query string instead of a date window, and it drops papers already in
        the corpus BEFORE downloading/parsing them (that work costs ~10-30s per paper).

        :param search_query: arXiv API query syntax, e.g. "all:BERT AND cat:cs.CL"
        :param max_results: Maximum papers to fetch
        :param process_pdfs: Whether to download and parse PDFs
        :param store_to_db: Whether to store results in database
        :param db_session: Database session (required if store_to_db=True)
        :returns: Dictionary with processing results, statistics, and new_arxiv_ids
        """
        results = {
            "papers_fetched": 0,
            "papers_skipped_existing": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "new_arxiv_ids": [],
            "errors": [],
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            # Step 1: fetch metadata from arXiv by topic.
            # sort_by="relevance", NOT "submittedDate" (which the daily-ingestion path
            # uses). A topic search wants the best matches; sorting by date returns the
            # newest papers that loosely match the query, which in practice means
            # unrelated papers that happen to share a keyword.
            papers = await self.arxiv_client.fetch_papers_with_query(
                search_query=search_query,
                max_results=max_results,
                sort_by="relevance",
                sort_order="descending",
            )

            results["papers_fetched"] = len(papers)
            if not papers:
                logger.warning(f"No papers found for query: {search_query}")
                return results

            # Step 1.5: drop papers already in the corpus, before any expensive work.
            # upsert() alone would avoid duplicate rows but NOT the download/parse cost.
            if store_to_db and db_session:
                paper_repo = PaperRepository(db_session)
                new_papers = [p for p in papers if not paper_repo.get_by_arxiv_id(p.arxiv_id)]
                results["papers_skipped_existing"] = len(papers) - len(new_papers)
                papers = new_papers

                if not papers:
                    results["processing_time"] = (datetime.now() - start_time).total_seconds()
                    logger.info(f"All {results['papers_skipped_existing']} fetched papers already in corpus — nothing to do")
                    return results

            pdf_results = {}
            if process_pdfs:
                pdf_results = await self._process_pdfs_batch(papers)
                results["pdfs_downloaded"] = pdf_results["downloaded"]
                results["pdfs_parsed"] = pdf_results["parsed"]
                results["errors"].extend(pdf_results["errors"])

            if store_to_db and db_session:
                logger.info("Storing papers to database...")
                stored_count = self._store_papers_to_db(
                    papers,
                    pdf_results.get("parsed_papers", {}),
                    db_session,
                )
                results["papers_stored"] = stored_count
                results["new_arxiv_ids"] = [p.arxiv_id for p in papers]
            elif store_to_db:
                logger.warning("Database storage requested but no session provided")
                results["errors"].append("Database session not provided for storage")

            results["processing_time"] = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Topic pipeline completed in {results['processing_time']:.1f}s: "
                f"query={search_query!r}, "
                f"{results['papers_fetched']} fetched, "
                f"{results['papers_skipped_existing']} already known, "
                f"{results['papers_stored']} stored, "
                f"{len(results['errors'])} errors"
            )

            return results

        except Exception as e:
            logger.error(f"Topic pipeline error: {e}")
            results["errors"].append(f"Topic pipeline error: {str(e)}")
            raise PipelineException(f"Topic pipeline execution failed: {e}") from e

    async def _process_pdfs_batch(self, papers: list[ArxivPaper]) -> dict[str, Any]:
        """
        Process PDFs for a batch of papers with async concurrency.

        Uses overlapping download+parse pipeline:
        - Downloads happen concurrently (up to max_concurrent_downloads)
        - As each download completes, parsing starts immediately
        - Multiple PDFs can be parsing while others are still downloading

        This is optimal for production workloads like 100 papers/day.

        Args:
            papers: List of ArxivPaper objects

        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "downloaded": 0,
            "parsed": 0,
            "parsed_papers": {},
            "errors": [],
            "download_failures": [],
            "parse_failures": [],
        }

        logger.info(f"Starting async pipeline for {len(papers)} PDFs...")
        logger.info(f"Concurrent downloads: {self.max_concurrent_downloads}")
        logger.info(f"Concurrent parsing: {self.max_concurrent_parsing}")

        # Create semaphores for controlled concurrency
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        parse_semaphore = asyncio.Semaphore(self.max_concurrent_parsing)

        # Start downloading and parsing concurrently
        pipeline_tasks = [self._download_and_parse_pipeline(paper, download_semaphore, parse_semaphore) for paper in papers]
        pipeline_results = await asyncio.gather(*pipeline_tasks, return_exceptions=True)

        # Process results with detailed error tracking
        for paper, result in zip(papers, pipeline_results):
            if isinstance(result, Exception):
                error_msg = f"Pipeline error for {paper.arxiv_id}: {str(result)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            elif result:
                # Result should be a (download_success, parsed_paper) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    download_success, parsed_paper = result
                else:
                    error_msg = f"Pipeline error for {paper.arxiv_id}: Unexpected result type {type(result).__name__}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue

                if download_success:
                    results["downloaded"] += 1

                    if parsed_paper:
                        results["parsed"] += 1
                        results["parsed_papers"][paper.arxiv_id] = parsed_paper
                    else:
                        # Download succeeded but parsing failed
                        results["parse_failures"].append(paper.arxiv_id)
                else:
                    # Download failed
                    results["download_failures"].append(paper.arxiv_id)
            else:
                # No results returned
                results["download_failures"].append(paper.arxiv_id)

        # Simple processing summary
        logger.info(f"PDF processing: {results['downloaded']}/{len(papers)} downloaded, {results['parsed']} parsed")

        if results["download_failures"]:
            logger.warning(f"Download failures: {len(results['download_failures'])}")

        if results["parse_failures"]:
            logger.warning(f"Parse failures: {len(results['parse_failures'])}")

        # Add specific failure info to general errors list for backward compatibility
        if results["download_failures"]:
            results["errors"].extend([f"Download failed: {arxiv_id}" for arxiv_id in results["download_failures"]])
        if results["parse_failures"]:
            results["errors"].extend([f"PDF parse failed: {arxiv_id}" for arxiv_id in results["parse_failures"]])

        return results

    async def _download_and_parse_pipeline(
        self,
        paper: ArxivPaper,
        download_semaphore: asyncio.Semaphore,
        parse_semaphore: asyncio.Semaphore,
    ) -> tuple:
        """
        Complete download+parse pipeline for a single paper with true parallelism.
        Downloads PDF, then immediately starts parsing while other downloads continue.

        Returns:
            Tuple of (download_success: bool, parsed_paper: Optional[ParsedPaper])
        """
        download_success = False
        parsed_paper = None

        try:
            # Download PDF with download concurrency control
            async with download_semaphore:
                logger.debug(f"Starting download: {paper.arxiv_id}")
                pdf_path = await self.arxiv_client.download_pdf(paper, False)

                if pdf_path:
                    download_success = True
                    logger.debug(f"Download complete: {paper.arxiv_id}")
                else:
                    logger.error(f"Download failed: {paper.arxiv_id}")
                    return (False, None)

            # Parse PDF with parse concurrency control (happens AFTER download completes)
            async with parse_semaphore:
                logger.debug(f"Starting parse: {paper.arxiv_id}")
                pdf_content = await self.pdf_parser.parse_pdf(pdf_path=pdf_path)

                if pdf_content:
                    # Create ArxivMetadata from the paper
                    arxiv_metadata = ArxivMetadata(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        arxiv_id=paper.arxiv_id,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url,
                    )
                    # Combine into ParsedPaper
                    parsed_paper = ParsedPaper(arxiv_metadata=arxiv_metadata, pdf_content=pdf_content)
                    logger.debug(f"Parse complete: {paper.arxiv_id} - {len(pdf_content.raw_text)} chars extracted")
                else:
                    logger.warning(f"PDF parsing failed for {paper.arxiv_id}, continuing with metadata only")

        except Exception as e:
            logger.error(f"Pipeline error for {paper.arxiv_id}: {e}")
            raise MetadataFetchingException(f"Pipeline error for {paper.arxiv_id}: {e}") from e

        return (download_success, parsed_paper)

    def _serialize_parsed_content(self, parsed_paper: ParsedPaper) -> dict[str, Any]:
        """
        Serialize ParsedPaper content for database storage.

        :param parsed_paper: ParsedPaper object with PDF content
        :type parsed_paper: ParsedPaper
        :returns: Dictionary with serialized content for database storage
        :rtype: Dict[str, Any]
        """
        try:
            pdf_content = parsed_paper.pdf_content

            # Serialize sections
            sections = [{"title": section.title, "content": section.content} for section in pdf_content.sections]

            # Serialize references
            references = list(pdf_content.references)

            return {
                "raw_text": pdf_content.raw_text,
                "sections": sections,
                "references": references,
                "parser_used": pdf_content.parser_used.value if pdf_content.parser_used else None,
                "parser_metadata": pdf_content.metadata or {},
                "pdf_processed": True,
                "pdf_processing_date": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Failed to serialize parsed content: {e}")
            return {"pdf_processed": False, "parser_metadata": {"error": str(e)}}

    def _store_papers_to_db(
        self,
        papers: list[ArxivPaper],
        parsed_papers: dict[str, ParsedPaper],
        db_session: Session,
    ) -> int:
        """
        Store papers and parsed content into the database with comprehensive content storage.

        Args:
            papers: List of ArxivPaper metadata
            parsed_papers: Dictionary of parsed PDF content keyed by arxiv_id
            db_session: Active database session

        Returns:
            Number of papers that got stored into the database
        """
        paper_repo = PaperRepository(db_session)
        stored_count = 0

        for paper in papers:
            # Base paper data
            published_date = (
                date_parser.parse(paper.published_date) if isinstance(paper.published_date, str) else paper.published_date
            )
            try:
                # Store the metadata first
                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": published_date,
                    "pdf_url": paper.pdf_url,
                }

                # Get parsed content if available
                parsed_paper = parsed_papers.get(paper.arxiv_id)
                if parsed_paper:
                    parsed_content = self._serialize_parsed_content(parsed_paper)
                    paper_data.update(parsed_content)
                    raw_text_len = len(parsed_content.get("raw_text", "")) if parsed_content.get("raw_text") else 0
                    logger.debug(f"Storing paper {paper.arxiv_id} with parsed content ({raw_text_len} chars)")
                else:
                    # No parsed content - just store metadata
                    paper_data.update(
                        {
                            "pdf_processed": False,
                            "parser_metadata": {"note": "PDF processing not available or failed"},
                        }
                    )
                    logger.debug(f"Storing paper {paper.arxiv_id} with metadata only")

                paper_create = PaperCreate(**paper_data)
                stored_paper = paper_repo.upsert(paper_create)

                if stored_paper:
                    stored_count += 1
                    content_info = "with parsed content" if parsed_paper else "metadata only"
                    logger.debug(f"Stored paper {paper.arxiv_id} to database ({content_info})")
            except Exception as e:
                logger.error(f"Failed to store paper {paper.arxiv_id}: {e}")

        # Commit all changes
        try:
            db_session.commit()
            logger.info(f"Committed {stored_count} papers to database with full content storage")
        except Exception as e:
            logger.error(f"Failed to commit papers to database: {e}")
            db_session.rollback()
            stored_count = 0

        return stored_count


def make_metadata_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    pdf_cache_dir: Path | None = None,
    settings: Settings | None = None,
) -> MetadataFetcher:
    """Create MetadataFetcher instance with configuration settings.

    :param arxiv_client: Client for arXiv API operations
    :param pdf_parser: Service for parsing PDF documents
    :param pdf_cache_dir: Directory for caching downloaded PDFs
    :param settings: Application settings instance (uses default if None)
    :type arxiv_client: ArxivClient
    :type pdf_parser: PDFParserService
    :type pdf_cache_dir: Optional[Path]
    :type settings: Optional[Settings]
    :returns: Configured MetadataFetcher instance
    :rtype: MetadataFetcher
    """

    if settings is None:
        settings = get_settings()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        settings=settings,
    )
