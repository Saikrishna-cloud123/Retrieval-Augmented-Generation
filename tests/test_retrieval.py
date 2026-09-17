"""
Tests for the BM25 Retriever and Hybrid Retriever (RRF Fusion).

These tests verify:
1. BM25 index builds correctly
2. BM25 returns relevant keyword matches
3. BM25 tokenization works properly
4. RRF fusion correctly combines two ranked lists
5. RRF handles documents appearing in only one list
"""

import pytest
from src.ingestion.parser import Document
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


# ── Test Data ────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    """A set of Document chunks simulating our pipeline output."""
    return [
        Document(
            text="BM25 is a ranking function used by search engines to estimate "
                 "the relevance of documents to a given search query.",
            metadata={"source": "search.md", "chunk_index": 0}
        ),
        Document(
            text="Semantic search uses embeddings to find documents that are "
                 "conceptually similar, even if they don't share exact keywords.",
            metadata={"source": "search.md", "chunk_index": 1}
        ),
        Document(
            text="Hybrid search combines keyword-based retrieval with semantic "
                 "retrieval using score fusion techniques like RRF.",
            metadata={"source": "search.md", "chunk_index": 2}
        ),
        Document(
            text="Vector databases like Qdrant store embeddings and support "
                 "approximate nearest neighbor search using HNSW graphs.",
            metadata={"source": "vector_db.md", "chunk_index": 0}
        ),
        Document(
            text="RAG (Retrieval-Augmented Generation) reduces hallucinations by "
                 "grounding LLM responses in retrieved factual documents.",
            metadata={"source": "rag.md", "chunk_index": 0}
        ),
    ]


@pytest.fixture
def bm25_retriever(sample_chunks):
    """Build a BM25 index from the sample chunks."""
    return BM25Retriever(sample_chunks)


# ── BM25 Retriever Tests ────────────────────────────────────────

class TestBM25Retriever:

    def test_index_builds_successfully(self, bm25_retriever):
        """Test that the BM25 index builds without errors."""
        assert bm25_retriever.bm25 is not None
        assert len(bm25_retriever.documents) == 5

    def test_empty_documents_raises(self):
        """Test that building from empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty document list"):
            BM25Retriever([])

    def test_search_returns_relevant_results(self, bm25_retriever):
        """Test that searching for 'BM25' returns the BM25 chunk first."""
        results = bm25_retriever.search("BM25 ranking", top_k=3)

        assert len(results) > 0
        # The first result should be about BM25
        assert "BM25" in results[0].text
        # Each result should have a bm25_score in metadata
        assert "bm25_score" in results[0].metadata
        assert results[0].metadata["bm25_score"] > 0

    def test_search_for_qdrant(self, bm25_retriever):
        """Test that searching for 'Qdrant' returns the vector DB chunk."""
        results = bm25_retriever.search("Qdrant", top_k=3)

        assert len(results) > 0
        assert "Qdrant" in results[0].text

    def test_search_respects_top_k(self, bm25_retriever):
        """Test that at most top_k results are returned."""
        results = bm25_retriever.search("search", top_k=2)
        assert len(results) <= 2

    def test_search_no_match_returns_empty(self, bm25_retriever):
        """Test that a query with no matching tokens returns few/no results."""
        results = bm25_retriever.search("xyznonexistent", top_k=5)
        # BM25 might return 0 results if the term doesn't exist
        for r in results:
            assert r.metadata["bm25_score"] > 0  # Only positive scores returned

    def test_results_sorted_by_score(self, bm25_retriever):
        """Test that results are sorted in descending score order."""
        results = bm25_retriever.search("search retrieval", top_k=5)

        scores = [r.metadata["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_tokenization(self, bm25_retriever):
        """Test that tokenization lowercases and splits correctly."""
        tokens = bm25_retriever._tokenize("Hello, World! BM25-Score")
        assert "hello" in tokens
        assert "world" in tokens
        assert "bm25" in tokens
        assert "score" in tokens

    def test_tokenization_empty_string(self, bm25_retriever):
        """Test that tokenizing empty string returns empty list."""
        tokens = bm25_retriever._tokenize("")
        assert tokens == []


# ── RRF Fusion Tests ────────────────────────────────────────────

class TestRRFFusion:
    """Test the RRF fusion logic in isolation."""

    def test_rrf_boosts_documents_in_both_lists(self):
        """Documents found by BOTH retrievers should rank higher."""
        # Simulate: doc_A is rank 1 in semantic, rank 2 in BM25
        # doc_B is rank 2 in semantic only
        # doc_C is rank 1 in BM25 only
        doc_a = Document(text="doc A - in both", metadata={
            "source": "test.md", "chunk_index": 0, "similarity_score": 0.9
        })
        doc_b = Document(text="doc B - semantic only", metadata={
            "source": "test.md", "chunk_index": 1, "similarity_score": 0.8
        })
        doc_c_bm25 = Document(text="doc C - bm25 only", metadata={
            "source": "test.md", "chunk_index": 2, "bm25_score": 5.0
        })
        doc_a_bm25 = Document(text="doc A - in both", metadata={
            "source": "test.md", "chunk_index": 0, "bm25_score": 4.0
        })

        semantic_results = [doc_a, doc_b]
        bm25_results = [doc_c_bm25, doc_a_bm25]

        # Create a minimal HybridRetriever just to test _rrf_fuse
        # We'll use a mock-like approach — create the instance and call _rrf_fuse directly
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fuse(semantic_results, bm25_results)

        # doc_a should be first because it appears in BOTH lists
        assert fused[0].metadata["source"] == "test.md"
        assert fused[0].metadata["chunk_index"] == 0
        assert len(fused[0].metadata["found_by"]) == 2
        assert "semantic" in fused[0].metadata["found_by"]
        assert "bm25" in fused[0].metadata["found_by"]

        # doc_a's RRF score should be higher than doc_b or doc_c
        assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]

    def test_rrf_preserves_individual_scores(self):
        """RRF results should contain both similarity_score and bm25_score."""
        doc = Document(text="test doc", metadata={
            "source": "t.md", "chunk_index": 0, "similarity_score": 0.85
        })
        doc_bm25 = Document(text="test doc", metadata={
            "source": "t.md", "chunk_index": 0, "bm25_score": 3.2
        })

        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fuse([doc], [doc_bm25])

        assert len(fused) == 1
        assert fused[0].metadata["similarity_score"] == 0.85
        assert fused[0].metadata["bm25_score"] == 3.2
        assert fused[0].metadata["rrf_score"] > 0

    def test_rrf_empty_lists(self):
        """RRF with empty lists should return empty results."""
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fuse([], [])
        assert fused == []

    def test_rrf_single_list(self):
        """RRF with results from only one retriever should still work."""
        doc = Document(text="only semantic", metadata={
            "source": "t.md", "chunk_index": 0, "similarity_score": 0.7
        })

        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fuse([doc], [])

        assert len(fused) == 1
        assert fused[0].metadata["found_by"] == ["semantic"]
        assert fused[0].metadata["bm25_score"] is None
