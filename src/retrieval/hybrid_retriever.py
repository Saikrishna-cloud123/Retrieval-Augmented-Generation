"""
Hybrid Retriever — Combines semantic and keyword search using RRF.

This module is the ORCHESTRATOR of our retrieval pipeline:

    User Query
        │
        ├──▶ [ Semantic Search ]  → ranked list A
        │         (Qdrant)
        │
        ├──▶ [ BM25 Search ]      → ranked list B
        │
        └──▶ [ RRF Fusion ]       → merged ranked list
                  │
                  ▼
             Top-K Hybrid Results

WHAT IS RECIPROCAL RANK FUSION (RRF)?

    RRF is a method for combining multiple ranked lists into one. Instead
    of trying to normalize and compare raw scores (which have different
    scales), RRF uses only the RANK POSITION of each document.

    Formula:
        RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                         i∈retrievers

    Where:
        - rank_i(doc) = the position of doc in retriever i's results (1-based)
        - k           = a constant (default 60) that controls how much
                        weight top-ranked results get vs. lower-ranked ones

    Example with k=60:
        Doc appears at rank 1 in semantic:  1/(60+1) = 0.01639
        Doc appears at rank 3 in BM25:      1/(60+3) = 0.01587
        RRF score = 0.01639 + 0.01587 = 0.03226

        Doc appears at rank 5 in semantic only:  1/(60+5) = 0.01538
        RRF score = 0.01538 (only one signal)

    The first doc ranks higher because it was found by BOTH retrievers.

WHY RRF INSTEAD OF SCORE NORMALIZATION?
    - BM25 scores: unbounded (0 to ∞), depend on corpus size
    - Cosine similarity: bounded (-1 to 1)
    - Min-max normalization is fragile and dataset-dependent
    - RRF is simple, robust, and parameter-free (k=60 works well universally)
    - RRF has been shown to match or outperform learned fusion methods
      in many benchmarks

DESIGN DECISIONS:
    - HybridRetriever takes its dependencies via constructor injection.
      This makes it easy to test with mocks and swap implementations.
    - We retrieve MORE than top_k from each retriever (2× top_k) to
      give RRF a larger pool of candidates for fusion.
    - Documents appearing in both lists get boosted (that's the whole point
      of hybrid search — agreement between retrievers is a strong signal).
"""

import logging
from typing import List, Dict, Optional

from src.ingestion.parser import Document
from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines semantic vector search and BM25 keyword search using RRF.

    Usage:
        hybrid = HybridRetriever(embedder, vector_store, bm25_retriever)
        results = hybrid.search("What is hybrid search?", top_k=5)
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: Optional[int] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            embedder:        For converting query text → vector.
            vector_store:    For semantic similarity search in Qdrant.
            bm25_retriever:  For keyword-based BM25 search.
            rrf_k:           The k parameter for RRF. Default from settings (60).
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

        logger.info(f"HybridRetriever initialized (RRF k={self.rrf_k})")

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Run hybrid search: semantic + BM25, fused with RRF.

        Args:
            query:  The user's search query.
            top_k:  Number of final results to return.

        Returns:
            List of Documents ranked by RRF score (highest first).
            Each Document's metadata contains:
                - rrf_score:          The fused RRF score
                - similarity_score:   Cosine similarity (if found by semantic)
                - bm25_score:         BM25 score (if found by BM25)
                - found_by:           List of retriever names that found it
        """
        # Retrieve MORE candidates from each retriever to give RRF
        # a bigger pool. 2× top_k is a common heuristic.
        candidate_count = top_k * 2

        # --- Run both retrievers in sequence ---
        # (In production, you'd run these in parallel with asyncio/threads)

        logger.debug(f"Running semantic search for: '{query}'")
        query_vector = self.embedder.embed_query(query)
        semantic_results = self.vector_store.search(query_vector, top_k=candidate_count)

        logger.debug(f"Running BM25 search for: '{query}'")
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_count)

        logger.info(
            f"Retrieval complete — Semantic: {len(semantic_results)} results, "
            f"BM25: {len(bm25_results)} results"
        )

        # --- Fuse with RRF ---
        fused_results = self._rrf_fuse(semantic_results, bm25_results)

        # Return only the top_k
        return fused_results[:top_k]

    def _rrf_fuse(
        self,
        semantic_results: List[Document],
        bm25_results: List[Document],
    ) -> List[Document]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.

        The algorithm:
        1. For each document in either list, calculate:
           RRF_score = Σ 1/(k + rank)  for each list it appears in
        2. Sort all documents by their combined RRF score (descending).

        Documents found by BOTH retrievers naturally score higher because
        they accumulate RRF contributions from two sources.

        Args:
            semantic_results: Ranked results from Qdrant vector search.
            bm25_results:     Ranked results from BM25 keyword search.

        Returns:
            A single list of Documents sorted by fused RRF score.
        """
        k = self.rrf_k

        # We need a way to identify "the same document" across both lists.
        # We use a composite key: source filename + chunk_index.
        # This works because both retrievers operate on the same chunk set.

        # Dict to accumulate RRF scores and track metadata
        # Key: (source, chunk_index) → value: {score, doc, found_by, ...}
        doc_scores: Dict[str, dict] = {}

        def _doc_key(doc: Document) -> str:
            """Create a unique key for a document chunk."""
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            return f"{source}::{chunk_idx}"

        # Process semantic results
        for rank, doc in enumerate(semantic_results, start=1):
            key = _doc_key(doc)
            rrf_contribution = 1.0 / (k + rank)

            if key not in doc_scores:
                doc_scores[key] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "similarity_score": doc.metadata.get("similarity_score"),
                    "bm25_score": None,
                    "found_by": [],
                }

            doc_scores[key]["rrf_score"] += rrf_contribution
            doc_scores[key]["similarity_score"] = doc.metadata.get("similarity_score")
            doc_scores[key]["found_by"].append("semantic")

        # Process BM25 results
        for rank, doc in enumerate(bm25_results, start=1):
            key = _doc_key(doc)
            rrf_contribution = 1.0 / (k + rank)

            if key not in doc_scores:
                doc_scores[key] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "similarity_score": None,
                    "bm25_score": None,
                    "found_by": [],
                }

            doc_scores[key]["rrf_score"] += rrf_contribution
            doc_scores[key]["bm25_score"] = doc.metadata.get("bm25_score")
            doc_scores[key]["found_by"].append("bm25")

        # Sort by RRF score (descending)
        sorted_entries = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Build the final Document list with enriched metadata
        fused_results = []
        for entry in sorted_entries:
            doc = entry["doc"]
            # Build clean metadata without leftover retriever-specific scores
            result_metadata = {
                k_meta: v for k_meta, v in doc.metadata.items()
                if k_meta not in ("similarity_score", "bm25_score", "embedding")
            }
            result_metadata["rrf_score"] = entry["rrf_score"]
            result_metadata["similarity_score"] = entry["similarity_score"]
            result_metadata["bm25_score"] = entry["bm25_score"]
            result_metadata["found_by"] = entry["found_by"]

            fused_results.append(Document(text=doc.text, metadata=result_metadata))

        logger.info(
            f"RRF fusion complete: {len(fused_results)} unique documents "
            f"(from {len(semantic_results)} semantic + {len(bm25_results)} BM25)"
        )

        return fused_results
