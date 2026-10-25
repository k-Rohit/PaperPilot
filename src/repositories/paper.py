from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        # Convert the Pydantic schema (PaperCreate) into a SQLAlchemy model (Paper).
        # model_dump() turns the schema into a plain dict; **dict unpacks it as kwargs.
        db_paper = Paper(**paper.model_dump())

        # Stage the new row in the session. Nothing is sent to Postgres yet —
        # this just adds it to the session's "pending changes" list.
        self.session.add(db_paper)

        # Flush the staged changes and commit the transaction. This is when
        # the actual INSERT SQL hits Postgres.
        self.session.commit()

        # Re-read the row from Postgres so server-generated fields (id UUID,
        # created_at, updated_at) get populated on the Python object.
        self.session.refresh(db_paper)

        # Return the fully-populated row to the caller.
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        # scalar() is a SQLAlchemy shortcut that returns a single value from a query, not a row of multiple values.
        return self.session.scalar(stmt)

    def get_by_id(self, paper_id: UUID) -> Paper | None:
        stmt = select(Paper).where(Paper.id == paper_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        stmt = select(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        stmt = select(func.count(Paper.id))
        return self.session.scalar(stmt) or 0

    def get_processed_papers(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        """Get papers that have been successfully processed with PDF content."""
        stmt = select(Paper).where(Paper.pdf_processed).order_by(Paper.pdf_processing_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_unprocessed_papers(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        """Get papers that haven't been processed for PDF content yet."""
        stmt = select(Paper).where(not Paper.pdf_processed).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_papers_with_raw_text(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        """Get papers that have raw text content stored."""
        stmt = (
            select(Paper).where(Paper.raw_text is not None).order_by(Paper.pdf_processing_date.desc()).limit(limit).offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_processing_stats(self) -> dict:
        """Get statistics about PDF processing status."""
        total_papers = self.get_count()

        # count processed papers
        processed_stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        processed_papers = self.session.scalar(processed_stmt) or 0

        # Count papers with text
        text_ext_stmt = select(func.count(Paper.id)).where(Paper.raw_text != None)
        papers_with_text = self.session.scalar(text_ext_stmt) or 0

        return {
            "total_papers": total_papers,
            "processed_papers": processed_papers,
            "papers_with_text": papers_with_text,
            "processing_rate": (processed_papers / total_papers * 100) if total_papers > 0 else 0,
            "text_extraction_rate": (papers_with_text / processed_papers * 100) if processed_papers > 0 else 0,
        }

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_create: PaperCreate) -> Paper:
        # "UPSERT" = UPDATE if the row exists, INSERT if it doesn't.
        # This makes the write path IDEMPOTENT: calling upsert twice with the
        # same arxiv_id results in one row, not a duplicate or a unique-constraint
        # crash. Required so ingestion can safely re-run / retry without breaking.

        # Step 1: look up by the unique key (arxiv_id) to see if it's already in DB.
        # get_by_arxiv_id returns Optional[Paper] — Paper if found, None if not.
        existing_paper = self.get_by_arxiv_id(paper_create.arxiv_id)

        if existing_paper:
            # UPDATE path: a row with this arxiv_id already exists.
            # Copy each field from the incoming schema onto the existing model.
            # exclude_unset=True filters out fields the caller didn't pass.
            # So if you build a PaperCreate # with just arxiv_id and title, only those two columns get updated.
            # raw_text, sections, etc. # stay untouched on the existing row.
            # Without that flag, un-passed fields would come through as None and silently wipe out existing data.
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing_paper, key, value)
            # update() flushes the mutations to Postgres (UPDATE statement) and
            # refreshes the in-memory object with new server-side values
            # (e.g. the updated_at timestamp).
            return self.update(existing_paper)
        # INSERT path: no existing row — fall through to create() to insert.
        return self.create(paper_create)
