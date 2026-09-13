import pytest
from src.ingestion.parser import Document
from src.ingestion.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    """
    Load the embedder once for all tests in this module.
    Loading the model is slow, so we don't want to do it per-test.
    """
    return Embedder()

def test_embed_single_document(embedder):
    """Test that a single document gets the correct embedding field."""
    doc = Document(text="This is a test document about Python.", metadata={"source": "test.txt"})
    embedded_doc = embedder.embed(doc)
    
    assert "embedding" in embedded_doc.metadata
    assert isinstance(embedded_doc.metadata["embedding"], list)
    assert len(embedded_doc.metadata["embedding"]) == embedder.embedding_dimension
    assert isinstance(embedded_doc.metadata["embedding"][0], float)

def test_embed_batch_documents(embedder):
    """Test batch processing produces correct number of embeddings."""
    docs = [
        Document(text="Document one", metadata={"id": 1}),
        Document(text="Document two", metadata={"id": 2}),
        Document(text="Document three", metadata={"id": 3}),
    ]
    
    embedded_docs = embedder.embed_batch(docs)
    
    assert len(embedded_docs) == 3
    for d in embedded_docs:
        assert "embedding" in d.metadata
        assert len(d.metadata["embedding"]) == embedder.embedding_dimension

def test_embed_empty_document(embedder):
    """Test handling of empty text."""
    doc = Document(text="", metadata={"source": "empty"})
    embedded_doc = embedder.embed(doc)
    
    # Depending on implementation, we either return original doc or add empty vector
    # Our implementation currently skips empty docs and just returns them.
    assert "embedding" not in embedded_doc.metadata

def test_embed_query(embedder):
    """Test that query string is converted to vector correctly."""
    query = "How does vector search work?"
    vector = embedder.embed_query(query)
    
    assert isinstance(vector, list)
    assert len(vector) == embedder.embedding_dimension
    assert isinstance(vector[0], float)
