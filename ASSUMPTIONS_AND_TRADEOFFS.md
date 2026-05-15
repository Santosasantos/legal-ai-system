# Assumptions and Tradeoffs

## Assumptions

### Input Documents
- All inputs are PDFs (scanned, digital, or mixed). Other formats (DOCX, images) are out of scope for this submission but the processing pipeline is designed to be extended.
- Documents are in English. The models used (Qwen2.5-VL, Qwen2.5) support multilingual input but the prompts and structured field extraction are English-optimised.
- A "messy" document means: low-resolution scans, skewed pages, handwritten annotations, inconsistent formatting, or partially illegible text — not corrupted binary files.

### Operator Edits
- Operator edits are submitted via the UI or API after reviewing a generated draft. The system does not auto-detect edits from a file system watcher.
- Simulated operator edits (used in evaluation) are representative of real editing patterns: adding headers, reformatting sections, adding disclaimers, removing unsupported claims.

### Deployment
- The system is designed to run on a single machine with Docker Compose. Horizontal scaling (multiple backend replicas, distributed ChromaDB) is architecturally possible but not implemented here.
- GPU is recommended for Ollama but not required. CPU inference works, just slower (~3–5x).

### Legal Correctness
- The system does not claim legal correctness. It generates first-pass drafts grounded in source documents. All outputs require human review before use.

---

## Tradeoffs

### Qwen2.5-VL for OCR vs. Dedicated OCR Services (AWS Textract, Google Document AI)

**Chose:** Qwen2.5-VL via Ollama

**Why:** Qwen2.5-VL is a vision-language model that understands document layout, tables, handwriting, and mixed content in a single pass. It outperforms Tesseract on degraded documents and requires no API key or cloud dependency. It's also fine-tunable for legal document formats.

**Tradeoff:** Slower than cloud OCR services (5–15s per page vs. <1s). Requires ~8GB VRAM for the 7B model. For production at scale, a hybrid approach (cloud OCR for speed + VLM for difficult pages) would be better.

---

### ChromaDB vs. Pinecone / Weaviate / pgvector

**Chose:** ChromaDB

**Why:** Zero-config, runs in Docker, supports rich metadata filtering, cosine similarity, and persistent storage. Perfect for a self-contained demo that reviewers can run locally.

**Tradeoff:** Not horizontally scalable. For production with millions of documents, Weaviate or pgvector (with PostgreSQL) would be more appropriate. The vector store interface is abstracted so swapping is straightforward.

---

### nomic-embed-text vs. OpenAI text-embedding-3-small

**Chose:** nomic-embed-text (local, via Ollama)

**Why:** Runs locally, no API cost, 768-dim vectors with strong performance on domain-specific text. Keeps the system fully offline-capable.

**Tradeoff:** Slightly lower quality than OpenAI's embeddings on general benchmarks. For legal domain text, the gap is small. Switching to OpenAI embeddings requires re-indexing all documents (one-time cost).

---

### JSON File Store vs. PostgreSQL for Document/Draft Persistence

**Chose:** JSON files in `/data/outputs/`

**Why:** Zero dependencies, human-readable, easy to inspect during review. Sufficient for the scale of this assessment.

**Tradeoff:** Not suitable for concurrent writes or large document volumes. In production, PostgreSQL with SQLAlchemy would replace this. The `document_store.py` utility is the only file that needs changing — all other services use it through the same interface.

---

### Sentence-Boundary Chunking vs. Fixed-Token Chunking

**Chose:** Sentence-boundary chunking with token-size enforcement

**Why:** Fixed-token chunking cuts mid-sentence, which fragments semantic meaning and degrades retrieval quality. Sentence-boundary chunking preserves complete thoughts, making retrieved passages more useful as evidence.

**Tradeoff:** Chunk sizes are variable (some sentences are long). Mitigated by the 512-token soft limit — if a section exceeds the limit, it's split at the nearest sentence boundary.

---

### Multi-Query Retrieval vs. Single-Query Retrieval

**Chose:** 6 targeted queries per draft type

**Why:** A single query like "key facts" misses passages about parties, dates, financial terms, and procedural history. Using 6 domain-specific queries and deduplicating results gives much better coverage of the document.

**Tradeoff:** 6x more embedding calls per draft generation. With local embeddings this adds ~2–3 seconds. Acceptable for a drafting workflow where quality matters more than latency.

---

### LLM-Based Rule Deduplication vs. String Similarity

**Chose:** LLM semantic deduplication

**Why:** Two rules like "Include the case number at the top" and "Always put case number in the header" are semantically identical but string-different. LLM deduplication catches these; string matching doesn't.

**Tradeoff:** One extra LLM call per new rule during deduplication. Since rule extraction happens asynchronously after an edit (not in the critical path), this is acceptable.

---

### Preference Rules as Prompt Injection vs. Fine-Tuning

**Chose:** Prompt injection (rules added to system prompt)

**Why:** Fine-tuning requires training data, compute, and model management. Prompt injection is immediate, inspectable, reversible, and works with any LLM provider. Rules can be deactivated via API if they produce bad results.

**Tradeoff:** System prompt length grows with more rules. Mitigated by only injecting `active=True` rules and capping at the most confident ones. For very large rule sets (100+), a retrieval step over rules would be needed.

---

## What I Would Do Differently in Production

1. **PostgreSQL** for document, draft, and rule persistence with proper migrations
2. **Async task queue** (Celery + Redis) for document processing — ingestion is slow and shouldn't block the API
3. **Re-ranking** (cross-encoder) after initial retrieval for higher precision
4. **Streaming** draft generation via Server-Sent Events for better UX
5. **Auth** (JWT) on all API endpoints
6. **Structured logging** to a log aggregator (Datadog, Loki)
7. **Model versioning** — track which model version generated each draft for reproducibility
8. **A/B testing** for preference rules — measure whether a rule actually improves draft quality before promoting it
