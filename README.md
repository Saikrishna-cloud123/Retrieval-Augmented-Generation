# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built to understand and implement the complete journey from raw documents to grounded LLM responses.

This project is currently in **Phase 1 — Ingestion and Semantic Retrieval**.

The current pipeline takes documents, processes and chunks their content, generates embeddings using `all-MiniLM-L6-v2`, stores those embeddings in Qdrant, and performs semantic similarity search to retrieve relevant chunks.

The next phase will introduce **Hybrid Search using BM25 + Semantic Search**, combine the rankings using **Reciprocal Rank Fusion (RRF)**, and pass the retrieved context to an LLM for answer generation with source citations.

---

## Project Status

### Phase 1 — Completed

* File ingestion
* Document parsing
* Text cleaning
* Text chunking
* Embedding generation
* Vector storage using Qdrant
* Metadata storage
* Semantic search using cosine similarity
* Top-K relevant chunk retrieval
* Source-file identification using metadata
* Qdrant running through Docker

### Phase 2 — Planned

* BM25 keyword search
* Hybrid search
* Reciprocal Rank Fusion (RRF)
* Improved retrieval quality
* LLM integration
* Context-aware answer generation
* Source citations and metadata
* Retrieval evaluation

---

## Architecture

```text
                Documents
                    |
                    v
            File Ingestion
                    |
                    v
          Parsing & Cleaning
                    |
                    v
               Chunking
                    |
                    v
      all-MiniLM-L6-v2 Embeddings
                    |
                    v
              Qdrant Vector DB
                    |
                    v
             Semantic Search
             (Cosine Similarity)
                    |
                    v
             Top-K Chunks
```

The planned Phase 2 architecture will extend this to:

```text
                  User Query
                      |
             +--------+--------+
             |                 |
             v                 v
      Semantic Search       BM25 Search
             |                 |
             +--------+--------+
                      |
                      v
            Reciprocal Rank Fusion
                     (RRF)
                      |
                      v
             Relevant Chunks
                      |
                      v
                     LLM
                      |
                      v
          Grounded Final Answer
                      |
                      v
            Citations / Metadata
```

---

# Phase 1 — Ingestion Pipeline

The ingestion pipeline follows these main steps.

## 1. File Ingestion

The pipeline accepts documents from the designated data/input directory.

The documents are loaded and passed to the processing pipeline.

Example:

```text
data/
├── document1.pdf
├── document2.pdf
└── document3.txt
```

Place your input files inside the directory specified by the project configuration.

---

## 2. Parsing and Cleaning

The loaded documents are parsed to extract their textual content.

The extracted text is then cleaned and prepared for further processing.

This step is important because raw documents can contain unnecessary formatting, whitespace, or other content that can negatively affect downstream retrieval.

---

## 3. Chunking

Large documents are divided into smaller chunks.

Instead of embedding an entire document as one vector, each chunk is converted into its own vector representation.

This allows the retrieval system to find specific sections of a document that are relevant to a query.

Conceptually:

```text
Document
   |
   +---- Chunk 1
   +---- Chunk 2
   +---- Chunk 3
   +---- Chunk 4
```

---

## 4. Embedding Generation

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

The model converts each text chunk into a numerical vector representation.

These vectors capture the semantic meaning of the text and allow similar pieces of information to be compared mathematically.

---

## 5. Vector Database — Qdrant

The generated embeddings are stored in **Qdrant**, which is used as the vector database.

Along with the vectors, relevant metadata is stored, such as the source file from which the chunk originated.

Example:

```text
Vector
   |
   +-- Text Chunk
   +-- Source File
   +-- Metadata
```

Qdrant is currently run using Docker.

---

## 6. Semantic Search

When a query is provided, the same embedding model converts the query into a vector.

The query vector is then compared against the vectors stored in Qdrant using **cosine similarity**.

The system retrieves the Top-K most relevant chunks.

For example:

```text
User Query
     |
     v
Query Embedding
     |
     v
Qdrant
     |
     v
Top 5 Relevant Chunks
     |
     +---- Chunk 1 → document1.pdf
     +---- Chunk 2 → document3.pdf
     +---- Chunk 3 → document1.pdf
     +---- Chunk 4 → document2.pdf
     +---- Chunk 5 → document3.pdf
```

---

# Current Limitation

During testing, the system was able to retrieve the Top 5 chunks and identify their source files correctly.

However, semantic search alone does not always return the most relevant results.

This is because semantic search focuses on the meaning of the query, while some queries also depend heavily on exact keywords, names, technical terms, or identifiers.

To improve this, Phase 2 will introduce **Hybrid Search**.

---

# Phase 2 — Hybrid Search and Generation

The planned retrieval system will combine two different search strategies.

### Semantic Search

Uses embeddings to understand the semantic meaning of the query.

### BM25

Uses keyword-based matching to identify documents or chunks containing important terms from the query.

Instead of relying on only one method, both search systems will produce rankings.

These rankings will then be combined using **Reciprocal Rank Fusion (RRF)**.

```text
                 User Query
                     |
          +----------+----------+
          |                     |
          v                     v
   Semantic Search          BM25 Search
          |                     |
          v                     v
   Ranked Results          Ranked Results
          |                     |
          +----------+----------+
                     |
                     v
                    RRF
                     |
                     v
             Fused Ranking
                     |
                     v
             Best Chunks
```

The final retrieved context will then be passed to an LLM to generate a grounded answer.

The goal is to produce answers that are supported by the retrieved documents and include citations or source metadata.

---

# Technologies Used

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Core implementation          |
| Sentence Transformers | Text embeddings              |
| `all-MiniLM-L6-v2`    | Embedding model              |
| Qdrant                | Vector database              |
| Docker                | Running Qdrant               |
| WSL 2                 | Linux environment on Windows |
| Cosine Similarity     | Semantic vector search       |
| BM25                  | Planned keyword search       |
| RRF                   | Planned ranking fusion       |
| LLM                   | Planned answer generation    |

---





# How to Run the Project

## Prerequisites

Before running the project, make sure you have:

* Git
* Python 3.x
* Docker Desktop
* WSL 2 on Windows
* A Linux distribution such as Ubuntu
* VS Code (recommended)

The Docker Desktop WSL 2 backend requires WSL 2 and is the recommended setup for this Windows workflow.

---

# 1. Fork the Repository

Start by forking this repository to your own GitHub account.

Then clone your fork:

```bash
git clone https://github.com/Saikrishna-cloud123/Retrieval-Augmented-Generation.git
```

Move into the project directory:

```bash
cd Retrieval-Augmented-Generation
```

---

# 2. Install WSL 2 on Windows

If you are using Windows, open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Restart your computer if Windows asks you to.

The command installs WSL and, by default, an Ubuntu distribution.

After restarting, verify the installation:

```powershell
wsl --version
```

You can also check the installed distributions and their WSL versions:

```powershell
wsl --list --verbose
```

You should see your Linux distribution running under **WSL 2**.

If necessary, set WSL 2 as the default:

```powershell
wsl --set-default-version 2
```

---

# 3. Install Docker Desktop

Download and install **Docker Desktop for Windows**.

During installation, use the **WSL 2 backend**.

After installation, open Docker Desktop and make sure it is running.

Go to:

```text
Docker Desktop
→ Settings
→ General
→ Use WSL 2 based engine
```

Then enable WSL integration under:

```text
Settings
→ Resources
→ WSL Integration
```

Select your Ubuntu distribution and apply the changes.

Docker's current Windows documentation recommends the WSL 2 backend for most Windows development workflows.

You can verify Docker from your terminal:

```bash
docker --version
```

And:

```bash
docker run hello-world
```

If the `hello-world` container runs successfully, Docker is ready.

---

# 4. Start Qdrant

The project uses Qdrant as the vector database.

Start Qdrant using Docker:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

Check whether the container is running:

```bash
docker ps
```

You should see a container named:

```text
qdrant
```

Qdrant will now be available locally.

The project can connect to the local Qdrant instance through:

```text
http://localhost:6333
```

To stop Qdrant:

```bash
docker stop qdrant
```

To start it again later:

```bash
docker start qdrant
```

To remove the container:

```bash
docker rm qdrant
```

---

# 5. Clone the Repository Inside WSL

Open Ubuntu/WSL and navigate to your project location.

For example:

```bash
cd ~
```

Then clone your fork:

```bash
git clone https://github.com/Saikrishna-cloud123/Retrieval-Augmented-Generation.git
```

Enter the project:

```bash
cd Retrieval-Augmented-Generation
```

If you use VS Code, you can open the project with:

```bash
code .
```

Docker recommends using WSL 2 integration for Linux-based development workflows on Windows.

---

# 6. Create a Python Virtual Environment

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

On Linux/WSL:

```bash
source venv/bin/activate
```

On Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

---

# 7. Install Python Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If the project uses a different dependency file, install the dependencies specified by that file.

---

# 8. Add Your Documents

Place the files you want to process inside the project's input/data directory.

For example:

```text
data/
├── document1.pdf
├── document2.pdf
├── notes.txt
└── document3.pdf
```

Make sure the directory matches the path configured in the project.

Do not upload private, confidential, or sensitive documents to a public repository.

---

# 9. Run the Ingestion Pipeline

Start the ingestion pipeline using the project's main script.

For example:

```bash
./venv/Scripts/python.exe demo.py
```

The pipeline should perform the following:

```text
Files
  ↓
Parsing
  ↓
Cleaning
  ↓
Chunking
  ↓
Embedding Generation
  ↓
Qdrant
  ↓
Vector Storage
```

---

# 10. Test Semantic Search

After the documents have been ingested, provide a query when prompted.

The system should return the most relevant chunks along with their source metadata.

Example:

```text
Query: <your query>

Top 5 Results:

1. Score: 0.82
   Source: document1.pdf
   Chunk: ...

2. Score: 0.79
   Source: document3.pdf
   Chunk: ...

...
```

The exact output depends on the documents used.

---

# Troubleshooting

### Docker is not recognized

Make sure Docker Desktop is installed and running.

Check:

```bash
docker --version
```

If you are using WSL, make sure your Linux distribution is enabled under:

```text
Docker Desktop
→ Settings
→ Resources
→ WSL Integration
```

---

### Qdrant connection refused

Make sure the Qdrant container is running:

```bash
docker ps
```

If it is stopped:

```bash
docker start qdrant
```

If the container does not exist, run:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

---

### Check Qdrant logs

If Qdrant is not starting correctly:

```bash
docker logs qdrant
```

---

### Check WSL version

Run:

```powershell
wsl --list --verbose
```

Make sure your distribution is using:

```text
VERSION
2
```

Microsoft documents `wsl --list --verbose` as the command for checking the WSL version of installed distributions.

---

# Learning Goals

This project is being developed as a hands-on exploration of how modern RAG systems work internally.

The main concepts being explored include:

* Document processing
* Text chunking
* Embeddings
* Vector databases
* Semantic search
* Similarity metrics
* Metadata-based retrieval
* Keyword search
* BM25
* Hybrid retrieval
* Reciprocal Rank Fusion
* LLM-based generation
* Grounded generation
* Citations and source attribution

---



# Current Limitations

This project is still under development.

The current version primarily focuses on **ingestion and semantic retrieval**.

Semantic search alone may not always retrieve the most relevant chunks, particularly for queries that depend on exact keywords, technical terms, names, or identifiers.

This limitation is one of the reasons Hybrid Search with BM25 and RRF is planned for the next phase.

---

# Contributing

Feel free to fork the repository, experiment with the pipeline, and suggest improvements.

If you find a bug or have an idea for improving the retrieval pipeline, feel free to open an issue or submit a pull request.

---

# Project Progress

```text
Phase 1
Ingestion + Embeddings + Vector Storage + Semantic Search
                         ✅

Phase 2
BM25 + Hybrid Search + RRF
                         🚧

Phase 3
LLM + Grounded Generation + Citations
                         🔜
```

---

## References

* Microsoft WSL Documentation: https://learn.microsoft.com/en-us/windows/wsl/
* Docker Desktop Documentation: https://docs.docker.com/desktop/
* Docker + WSL 2 Documentation: https://docs.docker.com/desktop/features/wsl/
* Qdrant Documentation: https://qdrant.tech/documentation/
* Sentence Transformers Documentation: https://www.sbert.net/
