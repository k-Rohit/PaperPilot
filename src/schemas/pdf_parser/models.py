from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ParserType(str, Enum):
    """PDF parser types options"""

    DOCLING = "docling"
    PYPDFIUM = "pypdfium"


class PaperSection(BaseModel):
    """Represents a section of a paper."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    level: int = Field(default=1, description="Section hierarchy level")


class PaperFigure(BaseModel):
    """Represents a figure in a paper."""

    caption: str = Field(..., description="Figure caption")
    id: str = Field(..., description="Figure identifier")


class PaperTable(BaseModel):
    """Represents a table in a paper."""

    caption: str = Field(..., description="Table caption")
    id: str = Field(..., description="Table identifier")


class PdfContent(BaseModel):
    """PDF-specific content extracted by parsers like Docling."""

    sections: list[PaperSection] = Field(default_factory=list, description="Paper sections")
    figures: list[PaperFigure] = Field(default_factory=list, description="Figures")
    tables: list[PaperTable] = Field(default_factory=list, description="Tables")
    raw_text: str = Field(..., description="Full extracted text")
    references: list[str] = Field(default_factory=list, description="References")
    parser_used: ParserType = Field(..., description="Parser used for extraction")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Parser metadata")


class ArxivMetadata(BaseModel):
    """Paper metadata from arXiv API."""

    title: str = Field(..., description="Paper title from arXiv")
    authors: list[str] = Field(..., description="Authors from arXiv")
    abstract: str = Field(..., description="Abstract from arXiv")
    arxiv_id: str = Field(..., description="arXiv identifier")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published_date: str = Field(..., description="Publication date")
    pdf_url: str = Field(..., description="PDF download URL")


class ParsedPaper(BaseModel):
    """Complete paper data combining arXiv metadata and PDF content."""

    arxiv_metadata: ArxivMetadata = Field(..., description="Metadata from arXiv API")
    pdf_content: PdfContent | None = Field(None, description="Content extracted from PDF")
