# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LEGAL AI SYSTEM                              │
│                     Pearson Specter Litt                            │
└─────────────────────────────────────────────────────────────────────┘

  PDF Upload
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. DOCUMENT PROCESSING                                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ PyMuPDF      │    │ Tesseract    │    │ Qwen2.5-VL           │  │
│  │ Direct text  │───▶│ OCR fallback │───▶│ Vision LLM fallback  │  │
│  │ extraction   │    │ (conf < 60%) │    │ (scanned/handwritten)│  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                │                                    │
│                                ▼                                    │
│                    ┌───────────────────────┐                        │
│                    │ Structured Field      │                        │
│                    │ Extraction (LLM)      │                        │
│                    │ case_number, parties, │                        │
│                    │ dates, claims, etc.   │                        │
│                    └───────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. SEMANTIC CHUNKING                                               │
│                                                                     │
│  • Section-aware splitting (headings, paragraphs)                   │
│  • Sentence-boundary chunking (never cuts mid-sentence)             │
│  • Overlapping windows (64 token overlap)                           │
│  • Semantic labelling: facts | claims | relief | definitions | ...  │
│  • Rich metadata per chunk:                                         │
│    - doc_id, filename, page_numbers                                 │
│    - document_type, case_number, parties, jurisdiction              │
│    - section_heading, semantic_label                                │
│    - is_table, is_definition, is_date_heavy                         │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. VECTOR STORE (ChromaDB)                                         │
│                                                                     │
│  • nomic-embed-text embeddings (768-dim, cosine similarity)         │
│  • Metadata filtering: doc_id, document_type, semantic_label        │
│  • Context expansion: retrieves neighbouring chunks                 │
│  • Score threshold: 0.35 minimum similarity                         │
│  • Chunk inspection by ID for evidence tracing                      │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. GROUNDED DRAFT GENERATION                                       │
│                                                                     │
│  Multi-query retrieval:                                             │
│  • 6 targeted queries per draft type                                │
│  • Deduplication across query results                               │
│  • Top 12 passages selected by score                                │
│                                                                     │
│  Prompt construction:                                               │
│  • Evidence block with source attribution                           │
│  • Structured metadata (case #, parties, dates)                     │
│  • Active preference rules injected                                 │
│                                                                     │
│  Output:                                                            │
│  • Draft with [Evidence N] inline citations                         │
│  • Evidence links: section → chunk_ids → source text               │
│  • Full traceability chain                                          │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. OPERATOR EDIT LEARNING LOOP                                     │
│                                                                     │
│  Operator edits draft                                               │
│       │                                                             │
│       ▼                                                             │
│  Unified diff computation                                           │
│       │                                                             │
│       ▼                                                             │
│  LLM analyses diff → extracts reusable rules                        │
│  e.g. "Always include case number in header"                        │
│       "Use bullet points for timeline events"                       │
│       │                                                             │
│       ▼                                                             │
│  Semantic deduplication (LLM-based)                                 │
│  Duplicate rules → confidence boost                                 │
│       │                                                             │
│       ▼                                                             │
│  Rules stored in preference_rules.json                              │
│       │                                                             │
│       ▼                                                             │
│  Injected into system prompt for ALL future drafts                  │
└─────────────────────────────────────────────────────────────────────┘
```

## LLM Provider Abstraction

```
BaseLLMProvider (ABC)
    ├── generate(prompt, system_prompt) → str
    ├── generate_with_image(prompt, image_path) → str  [vision/OCR]
    └── embed(texts) → List[List[float]]

Implementations:
    ├── OllamaProvider    (default — Qwen2.5-VL + Qwen2.5 + nomic-embed-text)
    ├── OpenAIProvider    (GPT-4o + text-embedding-3-small)
    ├── GeminiProvider    (Gemini 1.5 Pro + text-embedding-004)
    └── AnthropicProvider (Claude 3.5 Sonnet)

Switch via: LLM_PROVIDER env var
```

## Why Qwen2.5-VL for OCR?

- Natively understands document layouts, tables, and handwriting
- Outperforms Tesseract on degraded/scanned documents
- Open source, runs locally via Ollama — no API costs
- Fine-tunable for domain-specific legal document formats
- Handles mixed text/image pages in a single pass

## Chunking Strategy

The chunker uses a 3-layer approach:

1. **Structural split** — respects document sections (headings, page breaks)
2. **Sentence-boundary chunking** — never cuts mid-sentence using regex sentence detection
3. **Token-size enforcement** — 512 token target with 64 token overlap

Each chunk gets a **semantic label** (facts, claims, relief, definitions, etc.) assigned by the LLM, enabling targeted retrieval by content type.

## Retrieval Design

- **Multi-query**: 6 queries per draft type, results merged and deduplicated
- **Context expansion**: top-4 retrieved chunks get their neighbours fetched
- **Metadata filtering**: filter by doc_id, document_type, semantic_label
- **Score threshold**: 0.35 minimum to prevent irrelevant passages
- **Evidence tracing**: every passage has a chunk_id linkable back to source

## Improvement Loop Design

The improvement loop is not a diff display — it's a learning system:

1. Operator edits are diffed (unified diff)
2. LLM analyses additions/deletions to extract **generalizable rules**
3. Rules are semantically deduplicated (not string-matched)
4. Duplicate rules get confidence boosts (reinforcement)
5. All active rules are injected into the system prompt for future drafts
6. Rules can be deactivated via API if they produce bad results

## Assumptions and Tradeoffs

| Decision | Rationale |
|----------|-----------|
| Qwen2.5-VL for vision | Best open-source vision model for document OCR; avoids API costs |
| ChromaDB | Simple to deploy, good metadata filtering, cosine similarity |
| nomic-embed-text | High quality, runs locally, 768-dim vectors |
| JSON preference store | Simple, inspectable, no DB dependency for MVP |
| Sentence-boundary chunking | Prevents semantic fragmentation at chunk edges |
| Multi-query retrieval | Single query misses relevant passages; 6 queries covers the draft comprehensively |
| LLM deduplication | String matching misses semantically equivalent rules |
| 512 token chunks | Balances context richness vs. retrieval precision |
| Context expansion | Prevents cutting off important context at chunk boundaries |
