"""
Document Parser — Extracts text from raw files.

This is the FIRST stage of the RAG ingestion pipeline:

    Raw Files (.pdf, .txt, .md)
          │
          ▼
    [ Document Parser ]  ◄── YOU ARE HERE
          │
          ▼
    Document(text=..., metadata={source, file_type, pages, ...})

WHAT THIS MODULE DOES:
    Reads files from disk and extracts their textual content into a
    standardized Document dataclass. Different file formats require
    different extraction logic, but the output is always the same
    structure — a clean text string plus metadata about where it came from.

WHY WE NEED THIS:
    Embedding models and LLMs only understand plain text strings.
    A PDF file is binary (PostScript commands, font tables, graphics).
    A parser bridges the gap between "file on disk" and "text we can embed."

HOW IT FITS IN THE PIPELINE:
    The Document objects produced here are passed to the Text Cleaner
    (cleaner.py), then to the Chunker (chunker.py), which splits them
    into smaller pieces for embedding.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ── Data Model ──────────────────────────────────────────────────

@dataclass
class Document:
    """
    The core data structure for our entire RAG pipeline.

    Every stage downstream works with Document objects:
    - Parser creates them (this module)
    - Cleaner modifies the text field
    - Chunker splits one Document into many smaller Documents
    - Embedder generates vectors from the text field
    - Retriever returns Documents as search results

    Attributes:
        text:     The extracted textual content.
        metadata: A dictionary of key-value pairs describing the source.
                  Examples: {"source": "manual.pdf", "page": 3, "file_type": "pdf"}

    WHY a dataclass?
        - Less boilerplate than writing __init__ manually
        - Built-in __repr__ for easy debugging
        - Type hints for IDE autocomplete
        - Can be easily converted to dict with dataclasses.asdict()
    """

    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate that the document has actual content."""
        if not isinstance(self.text, str):
            raise TypeError(f"Document text must be a string, got {type(self.text)}")

    def __len__(self) -> int:
        """Returns the character count of the document text."""
        return len(self.text)

    def is_empty(self) -> bool:
        """Check if the document has no meaningful text content."""
        return len(self.text.strip()) == 0


# ── Parser ──────────────────────────────────────────────────────

# Supported file extensions — we explicitly list what we can handle.
# If someone tries to parse a .docx or .html, we fail clearly rather
# than silently producing garbage output.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class DocumentParser:
    """
    Extracts text content from files and returns Document objects.

    Usage:
        parser = DocumentParser()
        doc = parser.parse("data/documents/manual.pdf")
        print(doc.text)        # "Chapter 1: Introduction..."
        print(doc.metadata)    # {"source": "manual.pdf", "file_type": "pdf", ...}

    For PDFs with multiple pages, you can choose to get:
    - A single Document with all pages concatenated (default)
    - A list of Documents, one per page (useful for page-level citations)
    """

    def parse(self, file_path: str | Path) -> Document:
        """
        Parse a single file and return a Document with concatenated text.

        Args:
            file_path: Path to the file to parse.

        Returns:
            A Document containing the extracted text and source metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file type is not supported.
        """
        path = Path(file_path).resolve()

        # --- Validation ---
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{path.suffix}'. "
                f"Supported types: {SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Parsing file: {path.name} (type: {path.suffix})")

        # --- Dispatch to the correct extraction method ---
        # Each file type has its own extraction logic. We use the file
        # extension to decide which method to call. This is a simple
        # "strategy pattern" — easy to extend by adding new elif branches.
        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._parse_pdf(path)
        elif extension in {".txt", ".md"}:
            return self._parse_text(path)
        else:
            # This should never happen due to the check above,
            # but defensive programming is good practice.
            raise ValueError(f"No parser available for: {extension}")

    def parse_directory(self, dir_path: str | Path) -> list[Document]:
        """
        Parse all supported files in a directory (non-recursive).

        Args:
            dir_path: Path to the directory containing documents.

        Returns:
            A list of Document objects, one per successfully parsed file.
            Files that fail to parse are logged and skipped (not raised).
        """
        dir_path = Path(dir_path).resolve()

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        documents = []
        # Sort files for deterministic ordering (useful for testing)
        supported_files = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not supported_files:
            logger.warning(f"No supported files found in: {dir_path}")
            return documents

        logger.info(f"Found {len(supported_files)} supported file(s) in: {dir_path}")

        for file_path in supported_files:
            try:
                doc = self.parse(file_path)
                if not doc.is_empty():
                    documents.append(doc)
                else:
                    logger.warning(f"Skipping empty document: {file_path.name}")
            except Exception as e:
                # Log the error but don't stop processing other files.
                # In production, you might want to collect these errors
                # and report them to the user.
                logger.error(f"Failed to parse {file_path.name}: {e}")

        logger.info(f"Successfully parsed {len(documents)} document(s)")
        return documents

    # ── Private extraction methods ──────────────────────────────

    def _parse_text(self, path: Path) -> Document:
        """
        Extract text from .txt and .md files.

        This is straightforward — these files are already plain text.
        We just read the file content and attach metadata.

        For Markdown files, we intentionally keep the Markdown syntax
        (headers, bullets, etc.) because:
        1. Markdown headers (##) indicate section boundaries, which
           help the chunker split at meaningful points later.
        2. The embedding model can still understand text with Markdown
           formatting — it doesn't hurt similarity.
        """
        text = path.read_text(encoding="utf-8")

        metadata = {
            "source": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "file_path": str(path),
            "char_count": len(text),
        }

        logger.debug(f"Extracted {len(text)} chars from {path.name}")
        return Document(text=text, metadata=metadata)

    def _parse_pdf(self, path: Path) -> Document:
        """
        Extract text from PDF files using pypdf.

        HOW PDF TEXT EXTRACTION WORKS:
        A PDF stores text as drawing instructions — "render glyph 'H' at
        position (72, 700) using font Helvetica-12pt". The pypdf library:
        1. Opens the PDF binary and parses its object tree
        2. For each page, finds text stream objects
        3. Maps character codes to Unicode using the font's encoding table
        4. Reconstructs lines of text based on glyph positions

        LIMITATIONS:
        - Scanned PDFs (images of text) will return empty strings because
          there are no text streams — only image objects. Handling scanned
          PDFs requires OCR (Optical Character Recognition), which is out
          of scope for Milestone 1.
        - Complex layouts (multi-column, tables) may produce garbled output.
          Production systems use specialized libraries like pdfplumber or
          unstructured.io for these cases.
        """
        reader = PdfReader(path)
        pages_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
            else:
                logger.debug(
                    f"Page {page_num} of {path.name} yielded no text "
                    "(possibly scanned/image-based)"
                )

        # Concatenate all pages with double newline as separator.
        # Double newline creates a clear "paragraph break" between pages,
        # which helps the recursive chunker (Milestone 2) split at
        # page boundaries when possible.
        full_text = "\n\n".join(pages_text)

        metadata = {
            "source": path.name,
            "file_type": "pdf",
            "file_path": str(path),
            "total_pages": len(reader.pages),
            "pages_with_text": len(pages_text),
            "char_count": len(full_text),
        }

        logger.debug(
            f"Extracted {len(full_text)} chars from {path.name} "
            f"({len(pages_text)}/{len(reader.pages)} pages had text)"
        )
        return Document(text=full_text, metadata=metadata)

    def parse_pdf_by_page(self, file_path: str | Path) -> list[Document]:
        """
        Parse a PDF and return one Document PER PAGE.

        WHY would you want this?
        - Page-level granularity for citations ("Answer found on page 7")
        - Some chunking strategies work better with page-sized inputs
        - Easier to debug which page contains problematic text

        This is an alternative to parse() which concatenates all pages.
        You choose based on your use case.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"parse_pdf_by_page only accepts .pdf files, got: {path.suffix}")

        reader = PdfReader(path)
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                doc = Document(
                    text=page_text,
                    metadata={
                        "source": path.name,
                        "file_type": "pdf",
                        "file_path": str(path),
                        "page": page_num,
                        "total_pages": len(reader.pages),
                        "char_count": len(page_text),
                    },
                )
                documents.append(doc)

        logger.info(
            f"Parsed {path.name} into {len(documents)} page-level documents"
        )
        return documents
