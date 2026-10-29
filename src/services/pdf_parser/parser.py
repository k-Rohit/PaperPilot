import logging
from pathlib import Path
from typing import Literal

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import PdfContent

from .docling_parser import DoclingParser
from .pypdfium_parser import PyPdfiumParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service. Wraps either Docling (structured, slow) or pypdfium (text-only, fast)."""

    def __init__(
        self,
        parser: Literal["docling", "pypdfium"],
        max_pages: int,
        max_file_size_mb: int,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ):
        """Initialize PDF parser service with configurable limits.

        :param parser: Which backend to use — "docling" or "pypdfium"
        :param do_ocr: Docling only
        :param do_table_structure: Docling only
        """
        if parser == "docling":
            self.parser = DoclingParser(
                max_pages=max_pages, max_file_size_mb=max_file_size_mb, do_ocr=do_ocr, do_table_structure=do_table_structure
            )
        elif parser == "pypdfium":
            self.parser = PyPdfiumParser(max_pages=max_pages, max_file_size_mb=max_file_size_mb)
        else:
            raise ValueError(f"Unknown parser {parser!r} — use 'docling' or 'pypdfium'")

    async def parse_pdf(self, pdf_path: Path) -> PdfContent | None:
        """Parse PDF using the configured backend.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = self.parser.parse_pdf(pdf_path)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result
            logger.error(f"PDF parsing returned no result for {pdf_path.name}")
            raise PDFParsingException(f"PDF parsing returned no result for {pdf_path.name}")

        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"PDF parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"PDF parsing error for {pdf_path.name}: {e}") from e
