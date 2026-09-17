"""
Vector Store Module — Connects to Qdrant for storing and searching vectors.

This is the FIFTH and final stage of the RAG ingestion pipeline:

    List[Document(..., metadata={embedding: [...]})]
          │
          ▼
    [ Vector Store ]  ◄── YOU ARE HERE
          │
          ▼
    Qdrant Database (Searchable!)

WHAT THIS MODULE DOES:
    Provides a clean interface over the Qdrant client to:
    1. Create/manage collections (tables for vectors)
    2. Upsert (insert/update) Documents with their embeddings
    3. Perform semantic similarity search

WHY WE NEED THIS:
    To find relevant chunks for a user's query, we need to quickly find
    the vectors that are closest to the query vector. Vector databases
    use algorithms like HNSW (Hierarchical Navigable Small World) to do
    this across millions of vectors in milliseconds.
"""

import hashlib
import logging
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.parser import Document
from src.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages connections and operations with the Qdrant vector database.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the Qdrant client.

        Args:
            host: Qdrant server host. Defaults to settings.
            port: Qdrant server port. Defaults to settings.
            collection_name: The collection to use. Defaults to settings.
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection_name

        try:
            logger.info(f"Connecting to Qdrant server at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
            self.client.get_collections()
            logger.info("Connected to Qdrant server successfully.")
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant server at {self.host}:{self.port} ({e}). "
                "Falling back to local embedded Qdrant storage ('./data/qdrant_db')."
            )
            self.client = QdrantClient(path="./data/qdrant_db")

    def create_collection(self, vector_size: int, recreate: bool = False):
        """
        Ensure the collection exists with the correct vector dimensions.

        Args:
            vector_size: The dimension of the embeddings (e.g., 384 for MiniLM).
            recreate: If True, delete the existing collection and start fresh.
        """
        # Check if collection already exists
        collections_response = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections_response.collections)

        if exists:
            if recreate:
                logger.warning(f"Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(collection_name=self.collection_name, timeout=60)
                import time
                time.sleep(1)  # Give Qdrant a second to clean up files
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                return

        logger.info(f"Creating collection '{self.collection_name}' (vector size: {vector_size})")
        
        # Distance.COSINE is the standard metric for text embeddings.
        # It measures the angle between vectors (direction), ignoring magnitude.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            timeout=60
        )
        logger.info("Collection created.")

    def _generate_id(self, document: Document) -> str:
        """
        Generate a deterministic UUID for a Document.

        By hashing the text and source metadata, the same chunk of text
        will always generate the same ID. This allows 'upserting' — if we
        run ingestion twice, Qdrant will overwrite the old points instead
        of creating duplicates.
        """
        # Combine source file and text to make a unique signature
        source = document.metadata.get("source", "unknown")
        chunk_index = str(document.metadata.get("chunk_index", "0"))
        signature = f"{source}-{chunk_index}-{document.text}"
        
        # Create an MD5 hash (32 hex characters)
        hash_hex = hashlib.md5(signature.encode("utf-8")).hexdigest()
        
        # Format it as a UUID (8-4-4-4-12 format) required by Qdrant
        return f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:]}"

    def upsert(self, documents: List[Document]):
        """
        Store Documents and their embeddings in Qdrant.

        Args:
            documents: List of Documents that MUST have an 'embedding' in metadata.
        """
        if not documents:
            return

        points = []
        for doc in documents:
            if "embedding" not in doc.metadata:
                logger.warning(f"Skipping document with no embedding: {doc.metadata.get('source')}")
                continue

            # We store the text itself inside the payload along with other metadata
            # so we can retrieve the text later during search.
            payload = {k: v for k, v in doc.metadata.items() if k != "embedding"}
            payload["text"] = doc.text

            points.append(
                models.PointStruct(
                    id=self._generate_id(doc),
                    vector=doc.metadata["embedding"],
                    payload=payload,
                )
            )

        logger.info(f"Upserting {len(points)} points to '{self.collection_name}'...")
        
        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Upsert complete.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Document]:
        """
        Perform a semantic similarity search.

        Args:
            query_vector: The embedded user query.
            top_k: How many results to return.

        Returns:
            List of Documents representing the closest matches.
        """
        logger.debug(f"Searching for top {top_k} matches...")
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            # with_payload=True tells Qdrant to return the metadata we stored
            with_payload=True
        )

        results = []
        for scored_point in search_result:
            # Reconstruct the Document from the payload
            payload = scored_point.payload or {}
            text = payload.pop("text", "")
            
            # Add the similarity score to metadata so the LLM/Reranker can use it
            payload["similarity_score"] = scored_point.score
            
            results.append(Document(text=text, metadata=payload))

        return results

    def count(self) -> int:
        """Return the number of points in the collection."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0
