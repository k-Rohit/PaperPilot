# Lightweight PDF parser used by the API's live-fetch tool. pypdfium2 reads the
# text layer page by page — no ML models, so it runs in well under a second per
# paper (Docling loads ~1GB of weights and takes tens of seconds).


import logging
from pathlib import Path

import pypdfium2 as pdfium

from src.exceptions import PDFParsingException, PDFValidationError
from src.schemas.pdf_parser.models import ParserType, PdfContent

logger = logging.getLogger(__name__)


class PyPdfiumParser:
    """Fast text-only PDF parser built on pypdfium2."""

    def __init__(self, max_pages: int, max_file_size_mb: int):
        """
        :param max_pages: Maximum number of pages to process
        :param max_file_size_mb: Maximum file size in MB
        """
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Comprehensive PDF validation including size and page limits.

        :param pdf_path: Path to PDF file
        :returns: True if PDF appears valid and within limits
        :raises PDFValidationError: if the file is empty, too large, not a PDF, or has too many pages
        """
        try:
            # Check file exists and is not empty
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            actual_pages = len(pdf_doc)
            pdf_doc.close()

            if actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}") from e

    def parse_pdf(self, pdf_path: Path) -> PdfContent | None:
        """Extract the full text layer of a PDF.

        :param pdf_path: Path to PDF file
        :returns: PdfContent, or None if the PDF exceeds the size/page limits
        :raises PDFValidationError: for empty or non-PDF files
        :raises PDFParsingException: if pypdfium2 fails to read the text
        """
        try:
            self._validate_pdf(pdf_path)

            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                pages = []
                for page in pdf:
                    textpage = page.get_textpage()
                    pages.append(textpage.get_text_range())
                    textpage.close()
                    page.close()
            finally:
                pdf.close()

            raw_text = "\n".join(pages).strip()
            if not raw_text:
                # Scanned PDFs have no text layer — only OCR (Docling with do_ocr) can read them
                raise PDFParsingException(f"No extractable text in {pdf_path.name} (scanned PDF?)")

            return PdfContent(
                sections=[],
                raw_text=raw_text,
                parser_used=ParserType.PYPDFIUM,
                metadata={"source": "pypdfium", "pages": len(pages)},
            )

        except PDFValidationError as e:
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            raise
        except PDFParsingException:
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with pypdfium: {pdf_path}: {e}")
            raise PDFParsingException(f"pypdfium parsing error for {pdf_path.name}: {e}") from e
