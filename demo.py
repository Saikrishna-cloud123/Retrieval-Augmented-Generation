"""
Interactive Demo: RAG Pipeline (Milestones 1–4)

Run with:
    python demo.py

This script demonstrates the end-to-end pipeline:
    1. Parsing raw files (.txt, .md, .pdf) -> Document objects
    2. Text Cleaning & Normalization -> Clean Document objects
    3. Recursive Text Chunking -> Embeddable Chunks with Metadata
    4. Embedding Generation -> Vectors
    5. Vector Storage -> Qdrant
    6. Interactive Hybrid Search (Semantic / BM25 / Hybrid RRF)
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on all platforms (including Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.config import settings


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_results(results, mode_name):
    """Pretty-print search results for any mode."""
    if not results:
        print("    (No results found)\n")
        return

    for i, res in enumerate(results, 1):
        score_parts = []

        if mode_name == "hybrid":
            rrf = res.metadata.get("rrf_score")
            sim = res.metadata.get("similarity_score")
            bm25 = res.metadata.get("bm25_score")
            found_by = res.metadata.get("found_by", [])
            if rrf is not None:
                score_parts.append(f"RRF: {rrf:.5f}")
            if sim is not None:
                score_parts.append(f"Semantic: {sim:.4f}")
            if bm25 is not None:
                score_parts.append(f"BM25: {bm25:.2f}")
            score_parts.append(f"Found by: {', '.join(found_by)}")
        elif mode_name == "semantic":
            sim = res.metadata.get("similarity_score", 0.0)
            score_parts.append(f"Cosine Similarity: {sim:.4f}")
        elif mode_name == "bm25":
            bm25 = res.metadata.get("bm25_score", 0.0)
            score_parts.append(f"BM25 Score: {bm25:.4f}")

        source = res.metadata.get("source", "unknown")
        chunk_idx = res.metadata.get("chunk_index", "?")
        score_str = " | ".join(score_parts)

        print(f"    {i}. [{score_str}]")
        print(f"       Source: {source} | Chunk: {chunk_idx}")
        preview = res.text[:140].replace('\n', ' ')
        print(f"       \"{preview}...\"\n")


def main():
    data_dir = Path("data/documents")

    # -------------------------------------------------------------
    # 1. PARSING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 1: DOCUMENT PARSING")
    parser = DocumentParser()
    raw_docs = parser.parse_directory(data_dir)

    print(f"\n[+] Found and parsed {len(raw_docs)} document(s):\n")
    for idx, doc in enumerate(raw_docs, 1):
        print(f"  Document #{idx}:")
        print(f"    - Source:    {doc.metadata.get('source')}")
        print(f"    - File Type: {doc.metadata.get('file_type')}")
        print(f"    - Length:    {len(doc)} characters")
        preview = doc.text[:100].replace("\n", " ")
        print(f"    - Preview:   \"{preview}...\"\n")

    # -------------------------------------------------------------
    # 2. CLEANING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 2: TEXT CLEANING & NORMALIZATION")
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_batch(raw_docs)

    print(f"\n[+] Cleaned {len(cleaned_docs)} document(s):\n")
    for idx, doc in enumerate(cleaned_docs, 1):
        print(f"  Document #{idx} ({doc.metadata.get('source')}): "
              f"{doc.metadata.get('char_count_original')} -> "
              f"{doc.metadata.get('char_count_cleaned')} chars")

    # -------------------------------------------------------------
    # 3. CHUNKING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 3: RECURSIVE TEXT CHUNKING")
    chunker = RecursiveChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_batch(cleaned_docs)

    print(f"\n[+] Generated {len(chunks)} total chunk(s) across all documents.\n")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        print(f"  Chunk #{i} | Source: {meta.get('source')} | "
              f"Chunk {meta.get('chunk_index') + 1} of {meta.get('chunk_total')} | "
              f"{len(chunk.text)} chars")

    # -------------------------------------------------------------
    # 4. EMBEDDING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 4: EMBEDDING GENERATION")
    print("Loading embedding model (this may take a few seconds on first run)...")
    embedder = Embedder()

    embedded_chunks = embedder.embed_batch(chunks)

    print(f"\n[+] Generated {embedder.embedding_dimension}-dimensional embeddings "
          f"for {len(embedded_chunks)} chunks.\n")

    # -------------------------------------------------------------
    # 5. VECTOR STORAGE STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 5: VECTOR STORAGE (QDRANT)")
    print("Connecting to Qdrant (ensure Docker is running)...")
    try:
        vector_store = VectorStore()
        vector_store.create_collection(
            vector_size=embedder.embedding_dimension, recreate=True
        )
        vector_store.upsert(embedded_chunks)
        count = vector_store.count()
        print(f"\n[+] Qdrant collection '{vector_store.collection_name}' "
              f"now contains {count} points.\n")
    except Exception as e:
        print(f"\n[!] Qdrant error: {e}")
        print("    Is the Docker container running? (docker ps)")
        return

    # -------------------------------------------------------------
    # 6. BUILD BM25 INDEX
    # -------------------------------------------------------------
    print_separator("STAGE 6: BM25 INDEX")
    bm25_retriever = BM25Retriever(chunks)
    print(f"\n[+] BM25 index built over {len(chunks)} chunks.\n")

    # -------------------------------------------------------------
    # 7. HYBRID RETRIEVER
    # -------------------------------------------------------------
    hybrid_retriever = HybridRetriever(embedder, vector_store, bm25_retriever)

    # -------------------------------------------------------------
    # INTERACTIVE SEARCH
    # -------------------------------------------------------------
    print_separator("INTERACTIVE SEARCH")
    print("""
  Search Modes:
    [1] Hybrid  (Semantic + BM25 + RRF)  (default)
    [2] Semantic only  (Qdrant vector search)
    [3] BM25 only  (keyword search)

  Commands:
    Type a query to search, or:
    'mode 1/2/3'  -- switch search mode
    'exit'        -- quit
""")

    current_mode = "hybrid"
    mode_names = {"1": "hybrid", "2": "semantic", "3": "bm25"}
    top_k = settings.top_k

    while True:
        mode_label = {"hybrid": "Hybrid", "semantic": "Semantic", "bm25": "BM25"}
        user_input = input(f"  [{mode_label[current_mode]}] Enter query: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("  Goodbye!")
            break

        # Mode switching
        if user_input.lower().startswith("mode "):
            mode_key = user_input.split()[-1]
            if mode_key in mode_names:
                current_mode = mode_names[mode_key]
                print(f"  -> Switched to {mode_label[current_mode]} mode.\n")
            else:
                print("  -> Invalid mode. Use 'mode 1', 'mode 2', or 'mode 3'.\n")
            continue

        # Execute search based on current mode
        print(f"\n  Searching ({mode_label[current_mode]}) for: '{user_input}'\n")

        if current_mode == "hybrid":
            results = hybrid_retriever.search(user_input, top_k=top_k)
        elif current_mode == "semantic":
            query_vec = embedder.embed_query(user_input)
            results = vector_store.search(query_vec, top_k=top_k)
        elif current_mode == "bm25":
            results = bm25_retriever.search(user_input, top_k=top_k)

        print(f"  [ Top {len(results)} Results ]\n")
        print_results(results, current_mode)

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print_separator("SUMMARY")
    print(f"  Input Files:    {len(raw_docs)}")
    print(f"  Cleaned Docs:   {len(cleaned_docs)}")
    print(f"  Final Chunks:   {len(chunks)}")
    print(f"  Embedded:       {len(embedded_chunks)}")
    print(f"  BM25 Indexed:   {len(chunks)}")
    print(f"  Search Modes:   Semantic | BM25 | Hybrid (RRF)")
    print("  Ready for Milestone 5 (Reranking & Context Construction)!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
