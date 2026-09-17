# Hybrid Search in RAG Systems

## What is Hybrid Search?

Hybrid search combines **keyword-based retrieval** (like BM25) with
**semantic retrieval** (using embeddings and vector similarity).

## Why Use Hybrid Search?

Neither approach alone covers all query types:

- **BM25** excels at exact matches: error codes, product IDs, specific names
- **Semantic search** excels at meaning: synonyms, paraphrases, intent

## Score Fusion

Results from both retrievers are merged using **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(doc) = sum(1 / (k + rank)) for each retriever list
```

This produces a single ranked list that benefits from both retrieval signals.
