"""
Text Cleaner — Normalizes and cleans extracted text.

This is the SECOND stage of the RAG ingestion pipeline:

    Document(raw text from parser)
          │
          ▼
    [ Text Cleaner ]  ◄── YOU ARE HERE
          │
          ▼
    Document(clean, normalized text)

WHAT THIS MODULE DOES:
    Takes the raw text extracted by the parser and normalizes it:
    - Strips excessive whitespace and blank lines
    - Removes non-printable / control characters
    - Fixes hyphenated line breaks from PDF extraction
    - Normalizes Unicode characters (e.g., smart quotes → straight quotes)

WHY WE NEED THIS:
    Raw text (especially from PDFs) contains layout artifacts that are
    meaningless noise. If we embed dirty text:
    - The embedding vector captures noise alongside meaning
    - BM25 may fail to match terms broken by hyphens ("connec-tion")
    - The LLM wastes context window tokens on whitespace
    Cleaning ensures both retrievers and the LLM work with high-quality text.

DESIGN DECISION:
    The cleaner is a separate module from the parser because:
    1. Single Responsibility — parser extracts, cleaner normalizes
    2. Testability — we can test cleaning logic independently
    3. Flexibility — different document types may need different cleaning
       (e.g., code files should preserve indentation; prose should not)
"""

import re
import logging
import unicodedata

from src.ingestion.parser import Document

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes document text for downstream processing.

    Usage:
        cleaner = TextCleaner()
        clean_doc = cleaner.clean(raw_doc)

    The cleaner applies a pipeline of transformations in a specific order.
    Order matters — for example, we fix hyphenation BEFORE collapsing
    whitespace, because hyphenation fix joins lines first.
    """

    def clean(self, document: Document) -> Document:
        """
        Apply all cleaning steps to a document's text.

        Returns a NEW Document with cleaned text and updated metadata.
        The original document is not modified (immutability principle).

        Args:
            document: A Document object with raw extracted text.

        Returns:
            A new Document with cleaned text. The metadata is preserved
            and augmented with 'char_count_after_cleaning'.
        """
        if document.is_empty():
            logger.warning(
                f"Received empty document for cleaning: "
                f"{document.metadata.get('source', 'unknown')}"
            )
            return document

        text = document.text

        # --- Cleaning Pipeline (order matters!) ---

        # Step 1: Fix hyphenated line breaks FIRST
        # "connec-\ntion" → "connection"
        # Must happen before whitespace normalization
        text = self._fix_hyphenated_breaks(text)

        # Step 2: Normalize Unicode
        # Smart quotes → straight quotes, etc.
        text = self._normalize_unicode(text)

        # Step 3: Remove non-printable characters
        # Control characters like \x00, \x01, etc.
        text = self._remove_control_characters(text)

        # Step 4: Normalize whitespace
        # Multiple spaces → single space, excessive blank lines → max 2
        text = self._normalize_whitespace(text)

        # Step 5: Strip leading/trailing whitespace
        text = text.strip()

        # --- Build new Document with cleaned text ---
        # We create a new Document rather than modifying the original.
        # This makes debugging easier — you can compare raw vs cleaned.
        cleaned_metadata = {
            **document.metadata,
            "char_count_original": len(document.text),
            "char_count_cleaned": len(text),
            "cleaning_applied": True,
        }

        logger.debug(
            f"Cleaned {document.metadata.get('source', 'unknown')}: "
            f"{len(document.text)} → {len(text)} chars "
            f"({len(document.text) - len(text)} removed)"
        )

        return Document(text=text, metadata=cleaned_metadata)

    def clean_batch(self, documents: list[Document]) -> list[Document]:
        """
        Clean a list of documents, skipping any that become empty.

        Args:
            documents: List of raw Document objects.

        Returns:
            List of cleaned Document objects (empty results filtered out).
        """
        cleaned = []
        for doc in documents:
            result = self.clean(doc)
            if not result.is_empty():
                cleaned.append(result)
            else:
                logger.warning(
                    f"Document became empty after cleaning: "
                    f"{doc.metadata.get('source', 'unknown')}"
                )
        return cleaned

    # ── Private cleaning methods ────────────────────────────────

    def _fix_hyphenated_breaks(self, text: str) -> str:
        """
        Rejoin words broken by hyphens at line breaks.

        PDF extractors often produce text like:
            "The applica-\ntion crashed"

        This should become:
            "The application crashed"

        The regex looks for:
            (lowercase letter)(hyphen)(newline)(lowercase letter)
        and joins them. We only match lowercase-to-lowercase to avoid
        breaking intentional hyphens like "well-known" at line ends.
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to their standard forms.

        WHAT IS UNICODE NORMALIZATION?
        Unicode has multiple ways to represent the same character:
        - 'é' can be a single codepoint (U+00E9) or 'e' + combining accent (U+0065 + U+0301)
        - NFC normalization picks the composed (single codepoint) form

        We also replace common typographic variants:
        - Smart/curly quotes → straight quotes (consistency for search)
        - Em/en dashes → regular hyphens
        - Non-breaking spaces → regular spaces
        """
        # NFC = Canonical Decomposition, followed by Canonical Composition
        # This ensures é is always one character, not e + accent
        text = unicodedata.normalize("NFC", text)

        # Replace typographic characters with standard equivalents
        replacements = {
            "\u2018": "'",   # Left single quote → apostrophe
            "\u2019": "'",   # Right single quote → apostrophe
            "\u201c": '"',   # Left double quote → straight quote
            "\u201d": '"',   # Right double quote → straight quote
            "\u2013": "-",   # En dash → hyphen
            "\u2014": "-",   # Em dash → hyphen
            "\u00a0": " ",   # Non-breaking space → regular space
            "\u2026": "...", # Ellipsis character → three dots
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _remove_control_characters(self, text: str) -> str:
        """
        Remove non-printable control characters.

        Control characters (ASCII 0-31, except tab/newline/carriage-return)
        are invisible and meaningless in text. They can appear in PDFs
        due to encoding issues and would pollute embeddings.

        We keep:
        - \\t (tab, 0x09) — sometimes meaningful for formatting
        - \\n (newline, 0x0A) — paragraph/line structure
        - \\r (carriage return, 0x0D) — will be normalized with whitespace
        """
        # Remove all control chars except \t, \n, \r
        return re.sub(r"[^\S \t\n\r]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace without destroying document structure.

        Rules:
        1. Multiple spaces/tabs on a single line → single space
        2. More than 2 consecutive newlines → exactly 2 newlines
           (preserves paragraph breaks but removes excessive gaps)
        3. Trailing whitespace on each line → removed
        """
        # Replace multiple spaces/tabs (not newlines) with single space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove trailing spaces on each line
        text = re.sub(r" +\n", "\n", text)

        # Collapse 3+ newlines into 2 (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text
