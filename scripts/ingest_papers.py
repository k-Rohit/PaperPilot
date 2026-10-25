"""Fetch the latest arXiv papers and make them searchable (replaces the Airflow DAG).

Pipeline:
    1. arXiv API  → fetch the newest papers (count: ArxivSettings.max_results, default 15)
    2. Docling    → download + parse each PDF
    3. Postgres   → store metadata + parsed text (upsert, so re-runs don't duplicate)
    4. Pinecone   → chunk + embed + upsert the papers that parsed successfully

"""

import asyncio

from src.db.factory import make_database
from src.repositories.paper import PaperRepository
from src.services.arxiv.factory import make_arxiv_client
from src.services.indexing.factory import make_hybrid_indexing_service
from src.services.metadata_fetcher import make_metadata_fetcher
from src.services.pdf_parser.factory import make_pdf_parser_service


async def main() -> None:
    fetcher = make_metadata_fetcher(arxiv_client=make_arxiv_client(), pdf_parser=make_pdf_parser_service())
    db = make_database()

    with db.get_session() as session:
        # Steps 1-3: fetch, parse, store
        results = await fetcher.fetch_and_process_papers(
            process_pdfs=True,
            store_to_db=True,
            db_session=session,
        )
        print(
            f"Fetched {results['papers_fetched']} | parsed {results['pdfs_parsed']} | "
            f"stored {results['papers_stored']} | errors {len(results['errors'])}"
        )

        # Re-read the stored rows: they carry the parsed raw_text and DB id the indexer needs
        repo = PaperRepository(session)
        paper_dicts = []
        for arxiv_id in results["arxiv_ids"]:
            paper = repo.get_by_arxiv_id(arxiv_id=arxiv_id)
            if paper is None or not paper.raw_text:  # not stored, or PDF couldn't be parsed
                continue
            paper_dicts.append(
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

    # Step 4: index into Pinecone — once, after the loop
    if not paper_dicts:
        print("No parsed papers to index.")
        return

    stats = await make_hybrid_indexing_service().index_papers_batch(paper_dicts, replace_existing=True)
    print(f"Indexed {stats['papers_processed']} papers → {stats['total_chunks_indexed']} chunks ({stats['total_errors']} errors)")


if __name__ == "__main__":
    asyncio.run(main())
