"""
Interactive Demo: Document Ingestion Pipeline (Milestones 1 & 2)

Run with:
    python demo.py

This script demonstrates the end-to-end ingestion process:
    1. Parsing raw files (.txt, .md, .pdf) -> Document objects
    2. Text Cleaning & Normalization -> Clean Document objects
    3. Recursive Text Chunking -> Embeddable Chunks with Metadata
"""

from pathlib import Path
from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import VectorStore


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


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
        print(f"    - Preview (first 100 chars):")
        preview = doc.text[:100].replace("\n", " ")
        print(f"      \"{preview}...\"\n")

    # -------------------------------------------------------------
    # 2. CLEANING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 2: TEXT CLEANING & NORMALIZATION")
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_batch(raw_docs)

    print(f"\n[+] Cleaned {len(cleaned_docs)} document(s):\n")
    for idx, doc in enumerate(cleaned_docs, 1):
        print(f"  Document #{idx} ({doc.metadata.get('source')}):")
        print(f"    - Original Length: {doc.metadata.get('char_count_original')} chars")
        print(f"    - Cleaned Length:  {doc.metadata.get('char_count_cleaned')} chars")
        print(f"    - Cleaned Applied: {doc.metadata.get('cleaning_applied')}")

    # -------------------------------------------------------------
    # 3. CHUNKING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 3: RECURSIVE TEXT CHUNKING")
    # For demonstration with short sample texts, we use smaller chunk size (150 chars, 30 overlap)
    # to show the chunking & overlap in action clearly!
    chunker = RecursiveChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_batch(cleaned_docs)

    print(f"\n[+] Generated {len(chunks)} total chunk(s) across all documents:\n")

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        print("-" * 70)
        print(f"  Chunk #{i} | Source: {meta.get('source')} | Chunk {meta.get('chunk_index') + 1} of {meta.get('chunk_total')}")
        print(f"  Metadata: {meta}")
        print("-" * 70)
        print(f"  Content ({len(chunk.text)} chars):")
        print(f"  \"\"\"\n{chunk.text}\n  \"\"\"\n")

    # -------------------------------------------------------------
    # 4. EMBEDDING STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 4: EMBEDDING GENERATION")
    print("Loading embedding model (this may take a few seconds on first run)...")
    embedder = Embedder()
    
    embedded_chunks = embedder.embed_batch(chunks)
    
    print(f"\n[+] Generated embeddings for {len(embedded_chunks)} chunks:\n")
    if embedded_chunks:
        sample = embedded_chunks[0]
        vec = sample.metadata["embedding"]
        print(f"  Sample Chunk ID: 1")
        print(f"  Vector Dimension: {len(vec)}")
        print(f"  Vector Preview: [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, ...]\n")

    # -------------------------------------------------------------
    # 5. VECTOR STORAGE STAGE
    # -------------------------------------------------------------
    print_separator("STAGE 5: VECTOR STORAGE (QDRANT)")
    print("Connecting to Qdrant (ensure Docker is running)...")
    try:
        vector_store = VectorStore()
        
        # Ensure collection exists and has correct dimensions
        vector_store.create_collection(vector_size=embedder.embedding_dimension, recreate=True)
        
        # Upsert the embedded chunks
        vector_store.upsert(embedded_chunks)
        
        count = vector_store.count()
        print(f"\n[+] Qdrant Collection '{vector_store.collection_name}' now contains {count} points.\n")
        
        # Interactive Search Test
        from src.config import settings
        
        print("\n" + "-" * 70)
        print("  INTERACTIVE SEARCH (Type 'exit' or 'quit' to stop)")
        print("-" * 70)
        
        while True:
            user_query = input("\n  Enter search query: ").strip()
            
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("  Exiting search loop...")
                break
                
            print(f"  Searching for: '{user_query}'")
            query_vec = embedder.embed_query(user_query)
            
            # Using settings.top_k which defaults to 5
            top_k_value = settings.top_k
            results = vector_store.search(query_vec, top_k=top_k_value)
            
            print(f"\n  [ Top {len(results)} Results for '{user_query}' ]\n")
            for i, res in enumerate(results, 1):
                score = res.metadata.get("similarity_score", 0.0)
                source = res.metadata.get("source", "unknown")
                chunk_idx = res.metadata.get("chunk_index", "unknown")
                print(f"    {i}. [Score: {score:.4f}] (Source: {source} | Chunk: {chunk_idx})")
                print(f"       \"{res.text[:120].replace(chr(10), ' ')}...\"\n")

    except Exception as e:
        print(f"\n[!] Failed to connect or write to Qdrant: {e}")
        print("    Did you start the Qdrant Docker container?")

    print_separator("SUMMARY")
    print(f"  Input Files:   {len(raw_docs)}")
    print(f"  Cleaned Docs:  {len(cleaned_docs)}")
    print(f"  Final Chunks:  {len(chunks)}")
    print(f"  Embedded:      {len(embedded_chunks)}")
    print("  Ready for Milestone 4 (Hybrid Search)!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
