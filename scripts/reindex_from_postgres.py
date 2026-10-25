# ruff: noqa: I001
from src.models.paper import Paper  # noqa: F401
import asyncio
from src.repositories.paper import PaperRepository
from src.services.indexing.factory import make_hybrid_indexing_service
from src.db.factory import make_database


async def main() -> None:
    db = make_database()
    with db.get_session() as session:
        papers = PaperRepository(session).get_processed_papers(limit=10_000)

        print(f"Found {len(papers)} papers")
        paper_dicts = []
        for paper in papers:
            if not paper.raw_text:
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
        print(f"{len(paper_dicts)} papers ready to index")

    stats = await make_hybrid_indexing_service().index_papers_batch(paper_dicts, replace_existing=True)
    print(stats)


if __name__ == "__main__":
    asyncio.run(main())
