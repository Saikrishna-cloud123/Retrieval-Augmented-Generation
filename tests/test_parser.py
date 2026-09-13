"""
Tests for the Document Parser and Text Cleaner.

RUN WITH:
    pytest tests/test_parser.py -v

WHAT WE'RE TESTING:
    1. Parser can extract text from .txt files
    2. Parser can extract text from .md files (preserving Markdown)
    3. Parser attaches correct metadata
    4. Parser raises clear errors for missing files / unsupported types
    5. parse_directory works for batch processing
    6. Cleaner normalizes whitespace, Unicode, and hyphenated breaks
    7. Cleaner preserves paragraph structure
"""

import pytest
from pathlib import Path

from src.ingestion.parser import Document, DocumentParser, SUPPORTED_EXTENSIONS
from src.ingestion.cleaner import TextCleaner


# ── Fixtures ────────────────────────────────────────────────────
# Fixtures are pytest's way of providing reusable test setup.
# Any test function that has a parameter name matching a fixture
# will automatically receive the fixture's return value.

@pytest.fixture
def parser():
    """Provide a fresh DocumentParser instance for each test."""
    return DocumentParser()


@pytest.fixture
def cleaner():
    """Provide a fresh TextCleaner instance for each test."""
    return TextCleaner()


@pytest.fixture
def sample_txt_path():
    """Path to the sample .txt test document."""
    return Path(__file__).parent.parent / "data" / "documents" / "sample.txt"


@pytest.fixture
def sample_md_path():
    """Path to the sample .md test document."""
    return Path(__file__).parent.parent / "data" / "documents" / "sample.md"


@pytest.fixture
def data_dir():
    """Path to the test documents directory."""
    return Path(__file__).parent.parent / "data" / "documents"


# ── Document Dataclass Tests ────────────────────────────────────

class TestDocument:
    """Tests for the Document data model itself."""

    def test_create_document(self):
        doc = Document(text="Hello world", metadata={"source": "test.txt"})
        assert doc.text == "Hello world"
        assert doc.metadata["source"] == "test.txt"

    def test_document_length(self):
        doc = Document(text="Hello")
        assert len(doc) == 5

    def test_empty_document(self):
        doc = Document(text="")
        assert doc.is_empty()

    def test_whitespace_only_is_empty(self):
        doc = Document(text="   \n\t  ")
        assert doc.is_empty()

    def test_non_empty_document(self):
        doc = Document(text="content")
        assert not doc.is_empty()

    def test_invalid_text_type(self):
        with pytest.raises(TypeError):
            Document(text=123)

    def test_default_metadata_is_empty_dict(self):
        doc = Document(text="test")
        assert doc.metadata == {}


# ── Parser Tests ────────────────────────────────────────────────

class TestDocumentParser:
    """Tests for file parsing functionality."""

    def test_parse_txt_file(self, parser, sample_txt_path):
        """Parser should extract text from .txt files."""
        doc = parser.parse(sample_txt_path)

        assert not doc.is_empty()
        assert "RAG" in doc.text
        assert "Retrieval-Augmented Generation" in doc.text
        assert doc.metadata["file_type"] == "txt"
        assert doc.metadata["source"] == "sample.txt"

    def test_parse_md_file(self, parser, sample_md_path):
        """Parser should extract text from .md files, preserving Markdown."""
        doc = parser.parse(sample_md_path)

        assert not doc.is_empty()
        assert "# Hybrid Search" in doc.text  # Markdown headers preserved
        assert "**BM25**" in doc.text          # Bold syntax preserved
        assert doc.metadata["file_type"] == "md"

    def test_parse_nonexistent_file(self, parser):
        """Parser should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent_file.txt")

    def test_parse_unsupported_extension(self, parser, tmp_path):
        """Parser should raise ValueError for unsupported file types."""
        # Create a temporary .docx file
        fake_file = tmp_path / "document.docx"
        fake_file.write_text("fake content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse(fake_file)

    def test_parse_directory(self, parser, data_dir):
        """parse_directory should return Documents for all supported files."""
        docs = parser.parse_directory(data_dir)

        assert len(docs) >= 2  # At least sample.txt and sample.md
        sources = [d.metadata["source"] for d in docs]
        assert "sample.txt" in sources
        assert "sample.md" in sources

    def test_parse_directory_nonexistent(self, parser):
        """parse_directory should raise for non-existent directory."""
        with pytest.raises(NotADirectoryError):
            parser.parse_directory("fake_directory")

    def test_metadata_has_char_count(self, parser, sample_txt_path):
        """Parsed documents should include character count in metadata."""
        doc = parser.parse(sample_txt_path)
        assert "char_count" in doc.metadata
        assert doc.metadata["char_count"] > 0


# ── Cleaner Tests ───────────────────────────────────────────────

class TestTextCleaner:
    """Tests for text cleaning and normalization."""

    def test_clean_preserves_normal_text(self, cleaner):
        """Clean text should pass through unchanged."""
        doc = Document(text="This is normal clean text.", metadata={"source": "test"})
        result = cleaner.clean(doc)
        assert result.text == "This is normal clean text."

    def test_fix_hyphenated_breaks(self, cleaner):
        """Hyphenated line breaks should be rejoined."""
        doc = Document(
            text="The applica-\ntion crashed during ini-\ntialization.",
            metadata={"source": "test"},
        )
        result = cleaner.clean(doc)
        assert "application" in result.text
        assert "initialization" in result.text

    def test_normalize_excessive_whitespace(self, cleaner):
        """Multiple spaces should collapse to one."""
        doc = Document(
            text="Too    many     spaces    here.",
            metadata={"source": "test"},
        )
        result = cleaner.clean(doc)
        assert "Too many spaces here." == result.text

    def test_normalize_excessive_newlines(self, cleaner):
        """More than 2 consecutive newlines should collapse to 2."""
        doc = Document(
            text="Paragraph one.\n\n\n\n\nParagraph two.",
            metadata={"source": "test"},
        )
        result = cleaner.clean(doc)
        assert result.text == "Paragraph one.\n\nParagraph two."

    def test_normalize_smart_quotes(self, cleaner):
        """Curly/smart quotes should become straight quotes."""
        doc = Document(
            text="\u201cHello,\u201d she said. \u2018World.\u2019",
            metadata={"source": "test"},
        )
        result = cleaner.clean(doc)
        assert '"Hello,"' in result.text
        assert "'World.'" in result.text

    def test_clean_adds_metadata(self, cleaner):
        """Cleaning should add cleaning-related metadata."""
        doc = Document(text="Test content.", metadata={"source": "test"})
        result = cleaner.clean(doc)
        assert result.metadata["cleaning_applied"] is True
        assert "char_count_cleaned" in result.metadata

    def test_clean_returns_new_document(self, cleaner):
        """Cleaning should return a NEW document, not modify the original."""
        original = Document(text="  extra spaces  ", metadata={"source": "test"})
        cleaned = cleaner.clean(original)
        # Original should be unchanged
        assert original.text == "  extra spaces  "
        assert cleaned.text == "extra spaces"

    def test_clean_batch(self, cleaner):
        """clean_batch should process multiple documents."""
        docs = [
            Document(text="Doc one.", metadata={"source": "a.txt"}),
            Document(text="Doc two.", metadata={"source": "b.txt"}),
        ]
        results = cleaner.clean_batch(docs)
        assert len(results) == 2

    def test_clean_empty_document(self, cleaner):
        """Cleaning an empty document should return it as-is."""
        doc = Document(text="", metadata={"source": "empty.txt"})
        result = cleaner.clean(doc)
        assert result.is_empty()
