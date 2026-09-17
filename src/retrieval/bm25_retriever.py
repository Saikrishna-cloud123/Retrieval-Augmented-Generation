"""
BM25 Keyword Retriever — Finds documents by exact term matching.

This module implements the KEYWORD SEARCH side of our hybrid retrieval:

    User Query
        │
        ├──▶ [ BM25 Retriever ]  ◄── YOU ARE HERE
        │         │
        │         ▼
        │    Keyword-ranked results
        │
        └──▶ [ Semantic Search (Qdrant) ]
                  │
                  ▼
             Vector-ranked results
                  │
        ┌─────────┘
        ▼
    [ RRF Fusion → hybrid_retriever.py ]

WHAT IS BM25?
    BM25 (Best Matching 25) is a bag-of-words ranking function that scores
    documents based on the query terms appearing in each document.

    It improves on simple TF-IDF by adding two key refinements:
    1. TERM FREQUENCY SATURATION — The first occurrence of a word matters
       most. Repeating it 100 times doesn't give 100× the score. This is
       controlled by parameter k1 (default 1.5).
    2. DOCUMENT LENGTH NORMALIZATION — Short documents that contain the
       query term are scored higher than long documents that happen to
       mention it once. Controlled by parameter b (default 0.75).

    The formula:
        BM25(q, d) = Σ IDF(qi) × f(qi,d) × (k1+1) / (f(qi,d) + k1 × (1 - b + b × |d|/avgdl))

    Where:
        - f(qi, d) = frequency of query term qi in document d
        - |d|      = length of document d (in tokens)
        - avgdl    = average document length across the corpus
        - IDF(qi)  = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
                     N = total documents, n(qi) = documents containing qi

WHY WE NEED THIS:
    Embedding-based semantic search is great for meaning, but poor for:
    - Exact term matches ("error ERR_CONN_REFUSED")
    - Rare technical terms ("BM25", "HNSW")
    - IDs and codes ("order #12345")

    BM25 catches these cases perfectly because it looks for the actual
    tokens, not their semantic meaning.

DESIGN DECISIONS:
    - We use the `rank_bm25` library (pure Python, no external services).
    - The BM25 index is built IN-MEMORY from the same Document chunks that
      were embedded and stored in Qdrant. This ensures both retrievers
      operate on identical data.
    - Tokenization is intentionally simple (lowercase + regex split).
      Production systems use stemming, stopword removal, etc., but for
      our learning project, simplicity is more valuable than marginal
      accuracy gains.
"""

import re
import logging
from typing import List

from rank_bm25 import BM25Okapi

from src.ingestion.parser import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Keyword-based retriever using BM25 ranking.

    Usage:
        # Build the index from your chunks
        retriever = BM25Retriever(chunks)

        # Search by keyword
        results = retriever.search("BM25 algorithm", top_k=5)
    """

    def __init__(self, documents: List[Document]):
        """
        Build a BM25 index from a list of Document chunks.

        This tokenizes every document and creates an inverted index
        (mapping from terms → documents containing them). The index
        lives in memory and is fast for small-to-medium corpora
        (< 100K documents).

        Args:
            documents: The same chunk Documents that were embedded.
        """
        if not documents:
            raise ValueError("Cannot build BM25 index from empty document list.")

        self.documents = documents

        # Tokenize all documents
        self.tokenized_corpus = [self._tokenize(doc.text) for doc in documents]

        # Build the BM25 index
        # BM25Okapi is the most common variant (the "Okapi" version from
        # the City University of London's Okapi information retrieval system).
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(
            f"BM25 index built: {len(documents)} documents, "
            f"avg {sum(len(t) for t in self.tokenized_corpus) / len(self.tokenized_corpus):.0f} tokens/doc"
        )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Find the top-K documents most relevant to the query by keywords.

        Args:
            query:  The user's search query string.
            top_k:  Number of results to return.

        Returns:
            List of Documents ranked by BM25 score (highest first),
            with 'bm25_score' added to each Document's metadata.
        """
        # Tokenize the query the same way we tokenized documents
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning(f"Query '{query}' produced no tokens after tokenization.")
            return []

        # Get BM25 scores for ALL documents
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each document with its score, then sort descending
        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Take only the top-K results that have a positive score
        results = []
        for doc, score in scored_docs[:top_k]:
            if score <= 0:
                break  # No point returning documents with zero relevance

            # Create a new Document with the BM25 score in metadata
            result_metadata = {**doc.metadata, "bm25_score": float(score)}
            results.append(Document(text=doc.text, metadata=result_metadata))

        logger.debug(
            f"BM25 search for '{query}': "
            f"returned {len(results)} results (top score: {scores.max():.4f})"
        )
        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on non-alphanumeric characters.

        Examples:
            "Hello, World!"     → ["hello", "world"]
            "BM25 algorithm"    → ["bm25", "algorithm"]
            "error_code: 404"   → ["error_code", "404"]

        WHY simple tokenization?
            For a learning project, this is sufficient and transparent.
            Production systems would add:
            - Stemming (running → run)
            - Stopword removal (the, is, at → removed)
            - Subword tokenization
            But each of these adds complexity without helping us understand
            the core BM25 concept.
        """
        # Convert to lowercase and split on any non-alphanumeric/underscore character
        tokens = re.findall(r'\w+', text.lower())
        return tokens
