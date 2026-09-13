"""
Tests for the Recursive Text Chunker.

RUN WITH:
    pytest tests/test_chunker.py -v

WHAT WE'RE TESTING:
    1. Short text (< chunk_size) returns a single chunk
    2. Long text is split into correctly sized chunks
    3. Overlap is applied between consecutive chunks
    4. Metadata propagation (chunk_index, chunk_total, source)
    5. Empty document returns empty list
    6. Paragraph boundaries are respected
    7. Hard split fallback for text with no separators
    8. Batch chunking across multiple documents
    9. Invalid configuration raises errors
   10. Integration: Parser → Cleaner → Chunker pipeline
"""

import pytest
from pathlib import Path

from src.ingestion.parser import Document, DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import RecursiveChunker


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def chunker():
    """Provide a chunker with small sizes for easy testing."""
    return RecursiveChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def no_overlap_chunker():
    """Provide a chunker with no overlap for simpler assertions."""
    return RecursiveChunker(chunk_size=100, chunk_overlap=0)


@pytest.fixture
def parser():
    return DocumentParser()


@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.fixture
def sample_txt_path():
    return Path(__file__).parent.parent / "data" / "documents" / "sample.txt"


@pytest.fixture
def sample_md_path():
    return Path(__file__).parent.parent / "data" / "documents" / "sample.md"


# ── Basic Chunking Tests ───────────────────────────────────────

class TestBasicChunking:
    """Tests for fundamental chunking behavior."""

    def test_short_text_returns_single_chunk(self, chunker):
        """Text shorter than chunk_size should come back as one chunk."""
        doc = Document(text="This is short.", metadata={"source": "test.txt"})
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "This is short."

    def test_empty_document_returns_empty_list(self, chunker):
        """Empty documents should produce no chunks."""
        doc = Document(text="", metadata={"source": "empty.txt"})
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_whitespace_only_returns_empty_list(self, chunker):
        """Whitespace-only documents should produce no chunks."""
        doc = Document(text="   \n\n\t  ", metadata={"source": "blank.txt"})
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_long_text_is_split(self, no_overlap_chunker):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create text that's about 300 chars — should produce 3-4 chunks at size 100
        long_text = "This is a sentence. " * 15  # ~300 chars
        doc = Document(text=long_text.strip(), metadata={"source": "long.txt"})
        chunks = no_overlap_chunker.chunk(doc)

        assert len(chunks) > 1
        # Every chunk should be at most chunk_size
        for chunk in chunks:
            assert len(chunk.text) <= no_overlap_chunker.chunk_size

    def test_all_text_is_preserved_without_overlap(self, no_overlap_chunker):
        """
        Without overlap, joining all chunks should approximately
        reconstruct the original text (minus separator differences).
        """
        text = "Paragraph one about RAG systems.\n\nParagraph two about embeddings.\n\nParagraph three about vector search."
        doc = Document(text=text, metadata={"source": "test.txt"})
        chunks = no_overlap_chunker.chunk(doc)

        # All original content should appear in at least one chunk
        for paragraph in ["Paragraph one", "Paragraph two", "Paragraph three"]:
            found = any(paragraph in c.text for c in chunks)
            assert found, f"'{paragraph}' not found in any chunk"


# ── Separator Behavior Tests ──────────────────────────────────

class TestSeparatorBehavior:
    """Tests for how the chunker respects document structure."""

    def test_splits_at_paragraph_boundaries(self):
        """The chunker should prefer splitting at paragraph breaks."""
        text = "First paragraph content here.\n\nSecond paragraph content here."
        # chunk_size big enough to hold each paragraph but not both
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text

    def test_falls_to_line_break_separator(self):
        """If paragraphs are too long, split at line breaks."""
        text = "Line one of the content.\nLine two of the content.\nLine three of the content."
        chunker = RecursiveChunker(chunk_size=55, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 55

    def test_falls_to_sentence_separator(self):
        """If lines are too long, split at sentence endings."""
        text = "First sentence is here. Second sentence is here. Third sentence is here. Fourth sentence is here."
        chunker = RecursiveChunker(chunk_size=55, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 55

    def test_hard_split_when_no_separators(self):
        """Text with no separators should be hard-split at chunk_size."""
        # 200 characters with no spaces, newlines, or periods
        text = "a" * 200
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert len(chunks) == 3  # 80 + 80 + 40
        assert len(chunks[0].text) == 80
        assert len(chunks[1].text) == 80
        assert len(chunks[2].text) == 40

    def test_merges_small_consecutive_pieces(self):
        """Small paragraphs should be merged together, not become tiny chunks."""
        text = "Hi.\n\nOk.\n\nYes.\n\nNo.\n\nEnd."
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        # All paragraphs combined are ~25 chars — should fit in one chunk
        assert len(chunks) == 1
        assert "Hi." in chunks[0].text
        assert "End." in chunks[0].text


# ── Overlap Tests ──────────────────────────────────────────────

class TestOverlap:
    """Tests for chunk overlap behavior."""

    def test_overlap_is_applied(self):
        """Second chunk should contain text from the end of the first chunk."""
        text = "First paragraph with enough content.\n\nSecond paragraph with different content."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=15)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        if len(chunks) >= 2:
            # The second chunk should start with some text from the first
            # (the overlap region), not just jump to "Second paragraph"
            assert len(chunks[1].text) > len("Second paragraph with different content.")

    def test_zero_overlap_produces_no_repetition(self):
        """With overlap=0, chunks should have no repeated content."""
        text = "Alpha section.\n\nBeta section.\n\nGamma section."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        # Each section should appear exactly once across all chunks
        full_text = " ".join(c.text for c in chunks)
        assert full_text.count("Alpha") == 1
        assert full_text.count("Beta") == 1
        assert full_text.count("Gamma") == 1

    def test_first_chunk_has_no_overlap(self):
        """The first chunk should NOT have any prepended overlap text."""
        text = "Paragraph one content here.\n\nParagraph two content here."
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert chunks[0].text.startswith("Paragraph one")


# ── Metadata Tests ─────────────────────────────────────────────

class TestMetadata:
    """Tests for chunk metadata propagation."""

    def test_chunk_index_and_total(self):
        """Each chunk should have correct index and total count."""
        text = "Part A.\n\nPart B.\n\nPart C."
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=0)
        chunks = chunker.chunk(Document(text=text, metadata={"source": "doc.txt"}))

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["chunk_total"] == len(chunks)

    def test_parent_metadata_is_inherited(self):
        """Chunks should carry forward all metadata from the parent document."""
        parent_meta = {
            "source": "report.pdf",
            "file_type": "pdf",
            "total_pages": 5,
            "cleaning_applied": True,
        }
        doc = Document(text="Some content here.", metadata=parent_meta)
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        # All parent metadata should be present
        assert chunks[0].metadata["source"] == "report.pdf"
        assert chunks[0].metadata["file_type"] == "pdf"
        assert chunks[0].metadata["total_pages"] == 5
        assert chunks[0].metadata["cleaning_applied"] is True
        # Plus chunk-specific metadata
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["chunk_total"] == 1

    def test_chunk_size_in_metadata(self):
        """Each chunk's metadata should record its own character count."""
        doc = Document(text="Hello world.", metadata={"source": "t"})
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        assert chunks[0].metadata["chunk_size"] == len("Hello world.")

    def test_chunk_overlap_in_metadata(self):
        """Each chunk's metadata should record the overlap setting used."""
        doc = Document(text="Content.", metadata={"source": "t"})
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)

        assert chunks[0].metadata["chunk_overlap"] == 50


# ── Configuration Validation Tests ─────────────────────────────

class TestConfiguration:
    """Tests for chunker parameter validation."""

    def test_overlap_must_be_less_than_size(self):
        """chunk_overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        """chunk_overlap > chunk_size should also raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            RecursiveChunker(chunk_size=100, chunk_overlap=200)

    def test_zero_chunk_size_raises(self):
        """chunk_size=0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            RecursiveChunker(chunk_size=0, chunk_overlap=0)

    def test_negative_overlap_raises(self):
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            RecursiveChunker(chunk_size=100, chunk_overlap=-1)

    def test_custom_separators(self):
        """Custom separators should be respected."""
        chunker = RecursiveChunker(
            chunk_size=50,
            chunk_overlap=0,
            separators=["---"],  # Only split on "---"
        )
        # Each section must be long enough that the total exceeds chunk_size
        text = "Section one has enough content here.---Section two also has enough content here.---Section three is present."
        chunks = chunker.chunk(Document(text=text, metadata={"source": "t"}))

        assert len(chunks) >= 2


# ── Batch Chunking Tests ──────────────────────────────────────

class TestBatchChunking:
    """Tests for processing multiple documents."""

    def test_batch_chunks_multiple_documents(self):
        """chunk_batch should return chunks from all documents."""
        docs = [
            Document(text="Doc one content.", metadata={"source": "a.txt"}),
            Document(text="Doc two content.", metadata={"source": "b.txt"}),
        ]
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_batch(docs)

        assert len(chunks) == 2
        sources = [c.metadata["source"] for c in chunks]
        assert "a.txt" in sources
        assert "b.txt" in sources

    def test_batch_skips_empty_documents(self):
        """Empty documents in a batch should produce no chunks."""
        docs = [
            Document(text="Has content.", metadata={"source": "a.txt"}),
            Document(text="", metadata={"source": "empty.txt"}),
            Document(text="Also has content.", metadata={"source": "b.txt"}),
        ]
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk_batch(docs)

        assert len(chunks) == 2


# ── Integration Test ───────────────────────────────────────────

class TestPipelineIntegration:
    """End-to-end test: Parser → Cleaner → Chunker."""

    def test_full_pipeline_txt(self, parser, cleaner, sample_txt_path):
        """Parse, clean, and chunk a real .txt file."""
        # Step 1: Parse
        doc = parser.parse(sample_txt_path)
        assert not doc.is_empty()

        # Step 2: Clean
        cleaned = cleaner.clean(doc)
        assert not cleaned.is_empty()

        # Step 3: Chunk
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(cleaned)

        assert len(chunks) >= 1
        # All chunks should be within size limit
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 40  # Allow overlap margin
            assert chunk.metadata["source"] == "sample.txt"
            assert "chunk_index" in chunk.metadata

        # Content should be preserved
        all_text = " ".join(c.text for c in chunks)
        assert "RAG" in all_text

    def test_full_pipeline_md(self, parser, cleaner, sample_md_path):
        """Parse, clean, and chunk a real .md file."""
        doc = parser.parse(sample_md_path)
        cleaned = cleaner.clean(doc)
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(cleaned)

        assert len(chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Hybrid Search" in all_text

    def test_full_pipeline_directory(self, parser, cleaner):
        """Parse, clean, and chunk all documents in the data directory."""
        data_dir = Path(__file__).parent.parent / "data" / "documents"

        # Parse all files
        docs = parser.parse_directory(data_dir)
        assert len(docs) >= 2

        # Clean all
        cleaned = cleaner.clean_batch(docs)
        assert len(cleaned) >= 2

        # Chunk all
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk_batch(cleaned)

        assert len(chunks) >= 2
        # Every chunk should have proper metadata
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata
