"""
Embedding Module — Converts text into semantic vectors.

This is the FOURTH stage of the RAG ingestion pipeline:

    List[Document(chunk_text, metadata)]
          │
          ▼
    [ Embedder ]  ◄── YOU ARE HERE
          │
          ▼
    List[Document(chunk_text, metadata={..., embedding: [0.1, ...]})]

WHAT THIS MODULE DOES:
    Uses a pre-trained embedding model (e.g., all-MiniLM-L6-v2) to
    convert chunk text into dense numerical vectors (embeddings).
    These vectors capture the semantic meaning of the text.

WHY WE NEED THIS:
    Vector databases require vectors to perform semantic search.
    By converting text into vectors, we can find chunks that are
    conceptually similar to a query, even if they don't share exact words.
"""

import logging
from typing import List, Optional

# sentence-transformers is a popular library for computing embeddings
# based on HuggingFace transformers.
from sentence_transformers import SentenceTransformer

from src.ingestion.parser import Document
from src.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps an embedding model to convert Document text into vectors.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder and load the model.

        Args:
            model_name: The name of the HuggingFace model to load.
                        Defaults to settings.embedding_model_name.
        """
        self.model_name = model_name or settings.embedding_model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        
        # This will download the model to a local cache (~80MB for all-MiniLM)
        # on the very first run. Subsequent runs load from cache.
        self.model = SentenceTransformer(self.model_name)
        
        # We can inspect the model to see what size vectors it produces
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Vector dimension: {self.embedding_dimension}")

    def embed(self, document: Document) -> Document:
        """
        Embed a single Document.

        Args:
            document: The Document to embed.

        Returns:
            A new Document with the 'embedding' vector added to its metadata.
        """
        if document.is_empty():
            logger.warning("Attempted to embed an empty document.")
            return document

        # model.encode() returns a numpy array. We convert it to a python list
        # of floats because most vector DB clients (like Qdrant) expect lists.
        vector = self.model.encode(document.text).tolist()

        # Create a new metadata dict to avoid modifying the original Document
        new_metadata = {**document.metadata, "embedding": vector}
        
        return Document(text=document.text, metadata=new_metadata)

    def embed_batch(self, documents: List[Document], batch_size: int = 32) -> List[Document]:
        """
        Embed multiple Documents efficiently using batching.

        Batching is much faster than a loop because it fully utilizes CPU/GPU
        vectorization capabilities.

        Args:
            documents: List of Documents to embed.
            batch_size: How many texts to process at once.

        Returns:
            List of new Documents with embeddings.
        """
        if not documents:
            return []

        logger.info(f"Embedding {len(documents)} documents in batches of {batch_size}...")
        
        # Extract all texts
        texts = [doc.text for doc in documents]
        
        # Encode all texts at once (this handles batching internally)
        # show_progress_bar=True gives a nice UI for large datasets
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)

        embedded_docs = []
        for doc, vector in zip(documents, embeddings):
            new_metadata = {**doc.metadata, "embedding": vector.tolist()}
            embedded_docs.append(Document(text=doc.text, metadata=new_metadata))

        logger.info("Batch embedding complete.")
        return embedded_docs

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a raw search query string.

        Used during the retrieval phase to convert the user's question
        into a vector so we can search the database.

        Args:
            query: The user's search query.

        Returns:
            The query vector as a list of floats.
        """
        return self.model.encode(query).tolist()
