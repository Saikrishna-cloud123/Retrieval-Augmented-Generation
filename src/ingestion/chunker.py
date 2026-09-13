"""
Recursive Text Chunker — Splits documents into embeddable pieces.

This is the THIRD stage of the RAG ingestion pipeline:

    Document(clean text from cleaner)
          │
          ▼
    [ Recursive Chunker ]  ◄── YOU ARE HERE
          │
          ▼
    List[Document(chunk_text, metadata={..., chunk_index, ...})]

WHAT THIS MODULE DOES:
    Takes a cleaned Document and splits its text into smaller chunks
    that are suitable for embedding. Each chunk becomes its own Document
    with inherited metadata plus chunk-specific fields.

WHY WE NEED THIS:
    1. Embedding models have token limits (256 tokens for all-MiniLM-L6-v2).
       A full document would be truncated or produce a diluted vector.
    2. Smaller chunks produce more focused embeddings — a chunk about
       "BM25 scoring" will match a BM25 query much better than a
       10-page document that mentions BM25 once.
    3. The LLM context window is finite. Passing 5 focused chunks as
       context is far more useful than 1 enormous document.

HOW IT WORKS (Recursive Character Text Splitting):
    The algorithm tries to split at natural text boundaries, in order
    of preference:
        1. "\n\n" — Paragraph breaks (most semantic)
        2. "\n"   — Line breaks
        3. ". "   — Sentence endings
        4. " "    — Word boundaries (last resort)

    If splitting by paragraphs produces a piece that's still too large,
    we recurse with the next separator. This preserves document structure
    as much as possible while guaranteeing every chunk fits within the
    size limit.

    After splitting, we apply OVERLAP between consecutive chunks so that
    concepts spanning a split boundary aren't lost.

DESIGN DECISIONS:
    - We measure size in CHARACTERS, not tokens. Character count is
      instant (len()), while tokenization requires loading a model.
      For our embedding model, 500 chars ≈ 100-130 tokens — safely
      within the 256-token limit. If you switch to a tokenizer-based
      approach later, only _len() needs to change.
    - Separators are configurable but have sensible defaults.
    - Chunk overlap is applied as a post-processing step, not during
      the recursive split, to keep the logic clean and debuggable.
"""

import logging
from src.ingestion.parser import Document
from src.config import settings

logger = logging.getLogger(__name__)

# Default separators ordered from most semantic to least semantic.
# The algorithm tries each one in order, falling to the next only
# when the current separator produces pieces that are still too large.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker:
    """
    Splits documents into smaller chunks using recursive character splitting.

    Usage:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk(cleaned_document)

        # Or process multiple documents at once:
        all_chunks = chunker.chunk_batch(cleaned_documents)

    Each returned chunk is a Document with:
        - text: The chunk content
        - metadata: All parent metadata + chunk_index, chunk_total, etc.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        """
        Initialize the chunker with size constraints.

        Args:
            chunk_size:    Max characters per chunk. Defaults to settings.chunk_size (500).
            chunk_overlap: Characters to repeat between consecutive chunks.
                           Defaults to settings.chunk_overlap (100).
            separators:    Ordered list of split points to try.
                           Defaults to ["\n\n", "\n", ". ", " "].

        Raises:
            ValueError: If chunk_overlap >= chunk_size (would cause infinite loops).
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

        # --- Validation ---
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # If overlap >= chunk_size, each chunk would need to start with
        # more overlap text than its total size allows — infinite loop.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size}). Otherwise chunks would "
                f"never make forward progress through the text."
            )

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")

        logger.info(
            f"RecursiveChunker initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"separators={self.separators}"
        )

    def chunk(self, document: Document) -> list[Document]:
        """
        Split a single document into chunks.

        Args:
            document: A cleaned Document to split.

        Returns:
            A list of Document objects, each representing one chunk.
            If the document text is already within chunk_size, returns
            a single-element list (the document itself, with chunk metadata).
            If the document is empty, returns an empty list.
        """
        if document.is_empty():
            logger.warning(
                f"Skipping empty document: "
                f"{document.metadata.get('source', 'unknown')}"
            )
            return []

        text = document.text

        # --- Step 1: Recursively split the text ---
        raw_splits = self._recursive_split(text, self.separators)

        # --- Step 2: Apply overlap between consecutive splits ---
        chunks_text = self._merge_with_overlap(raw_splits)

        # --- Step 3: Wrap each chunk text in a Document with metadata ---
        chunk_documents = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_metadata = {
                **document.metadata,                 # Inherit all parent metadata
                "chunk_index": i,                    # 0-based position
                "chunk_total": len(chunks_text),     # How many chunks from this doc
                "chunk_size": len(chunk_text),        # Char count of this chunk
                "chunk_overlap": self.chunk_overlap,  # Overlap setting used
            }
            chunk_documents.append(Document(text=chunk_text, metadata=chunk_metadata))

        logger.debug(
            f"Chunked '{document.metadata.get('source', 'unknown')}': "
            f"{len(text)} chars → {len(chunk_documents)} chunk(s)"
        )

        return chunk_documents

    def chunk_batch(self, documents: list[Document]) -> list[Document]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of cleaned Document objects.

        Returns:
            A flat list of all chunk Documents from all input documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Batch chunking complete: {len(documents)} document(s) → "
            f"{len(all_chunks)} chunk(s)"
        )
        return all_chunks

    # ── Core Algorithm ──────────────────────────────────────────

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using separators in order of preference.

        This is the heart of the chunker. The algorithm:

        1. Pick the first separator in the list (e.g., "\n\n").
        2. Split the text by that separator.
        3. Walk through the pieces, merging small consecutive pieces
           together until adding another would exceed chunk_size.
        4. If any merged piece is STILL larger than chunk_size,
           recursively split it using the NEXT separator.
        5. If we run out of separators and a piece is still too large,
           hard-split it at chunk_size boundaries.

        Args:
            text:       The text to split.
            separators: Remaining separators to try (shrinks with recursion).

        Returns:
            A list of text pieces, each <= chunk_size.
        """
        # Base case: text already fits
        if len(text) <= self.chunk_size:
            # Only return non-empty strings
            return [text] if text.strip() else []

        # Base case: no separators left → hard split by character count
        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split the text by the current separator
        pieces = text.split(separator)

        # Now merge small consecutive pieces back together until
        # they approach chunk_size. This avoids creating tiny chunks
        # when paragraphs are short.
        #
        # Example with separator="\n\n" and chunk_size=500:
        #   pieces = ["Short para (80 chars)", "Another short (90 chars)",
        #             "Long para (600 chars)", "Final (50 chars)"]
        #
        #   Merge pass:
        #     current = "Short para" + "\n\n" + "Another short" = 174 chars → keep merging
        #     Adding "Long para" would make 774 chars → too big!
        #     → Flush current (174 chars) as a chunk
        #     → "Long para" (600 chars) > chunk_size → will be recursed later
        #     → "Final" (50 chars) becomes its own group

        merged_pieces = []
        current_piece = ""

        for piece in pieces:
            # Would adding this piece (plus separator) exceed the limit?
            if current_piece:
                combined = current_piece + separator + piece
            else:
                combined = piece

            if len(combined) <= self.chunk_size:
                # Still fits — keep accumulating
                current_piece = combined
            else:
                # Flush what we have so far
                if current_piece:
                    merged_pieces.append(current_piece)
                # Start a new accumulation with the current piece
                current_piece = piece

        # Don't forget the last accumulated piece
        if current_piece:
            merged_pieces.append(current_piece)

        # Now recursively handle any pieces that are still too large
        final_splits = []
        for piece in merged_pieces:
            if len(piece) <= self.chunk_size:
                if piece.strip():  # Skip whitespace-only pieces
                    final_splits.append(piece)
            else:
                # This piece is still too large — recurse with next separator
                sub_splits = self._recursive_split(piece, remaining_separators)
                final_splits.extend(sub_splits)

        return final_splits

    def _hard_split(self, text: str) -> list[str]:
        """
        Last-resort split: break text at exactly chunk_size boundaries.

        This only triggers when text has no spaces, newlines, or sentence
        endings (e.g., a very long URL or encoded string). In natural
        language text, the recursive separator logic almost always handles
        splitting before we reach this point.

        Args:
            text: A text string with no usable separators.

        Returns:
            A list of text pieces, each <= chunk_size.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            piece = text[i : i + self.chunk_size]
            if piece.strip():
                chunks.append(piece)
        return chunks

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """
        Apply overlap between consecutive chunks.

        For each chunk after the first, prepend the last `chunk_overlap`
        characters from the previous chunk. This ensures that concepts
        spanning a chunk boundary appear in at least one chunk intact.

        Example with chunk_overlap=10:
            splits = ["Hello world, this is chunk one.",
                       "This is chunk two, goodbye."]

            Result:
                Chunk 0: "Hello world, this is chunk one."
                Chunk 1: "chunk one. This is chunk two, goodbye."
                          ^^^^^^^^^^  ← overlap from previous chunk

        Args:
            splits: List of text pieces from _recursive_split().

        Returns:
            List of text pieces with overlap applied.
        """
        if not splits or self.chunk_overlap == 0:
            return splits

        result = [splits[0]]  # First chunk has no predecessor to overlap with

        for i in range(1, len(splits)):
            previous = splits[i - 1]

            # Take the last `chunk_overlap` characters from the previous chunk
            overlap_text = previous[-self.chunk_overlap:]

            # Find a clean word boundary in the overlap text to avoid
            # starting a chunk mid-word. Look for the first space.
            space_index = overlap_text.find(" ")
            if space_index != -1:
                # Start from the first complete word in the overlap
                overlap_text = overlap_text[space_index + 1:]

            # Prepend the overlap to the current chunk
            merged = overlap_text + " " + splits[i] if overlap_text else splits[i]
            result.append(merged)

        return result
