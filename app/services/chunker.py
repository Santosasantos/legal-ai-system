"""
Semantic Chunking Service
──────────────────────────
Strategy:
  1. Split document into natural sections (headings, paragraphs)
  2. Apply semantic boundary detection — don't cut mid-sentence or mid-clause
  3. Enforce token-size limits with overlap
  4. Attach rich metadata to every chunk for precise retrieval filtering
  5. Label each chunk with a semantic role (facts, claims, relief, definitions…)
"""

import re
import json
from typing import List, Optional, Tuple
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    ProcessedDocument, DocumentChunk, ChunkMetadata, StructuredFields,
)
from app.services.llm_provider import get_provider

logger = get_logger(__name__)

# ── Semantic section labels ───────────────────────────────────────────────────
SEMANTIC_LABELS = [
    "background_facts", "legal_claims", "relief_requested",
    "definitions", "procedural_history", "evidence",
    "parties_information", "dates_timeline", "financial_terms",
    "jurisdiction_venue", "signatures_attestation", "general",
]

HEADING_PATTERN = re.compile(
    r"^(?:SECTION|ARTICLE|CLAUSE|WHEREAS|NOW THEREFORE|IN WITNESS|"
    r"BACKGROUND|FACTS|CLAIMS|RELIEF|PARTIES|JURISDICTION|"
    r"\d+[\.\)]\s+[A-Z]|[IVX]+\.\s+[A-Z])",
    re.MULTILINE | re.IGNORECASE,
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ── Token counting (approximate — 1 token ≈ 4 chars) ─────────────────────────

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Section splitter ──────────────────────────────────────────────────────────

def _split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Split document text into (heading, body) tuples.
    Preserves natural document structure.
    """
    sections: List[Tuple[Optional[str], str]] = []
    parts = re.split(r"(--- Page \d+ ---)", text)

    current_heading: Optional[str] = None
    current_body: List[str] = []

    for part in parts:
        if re.match(r"--- Page \d+ ---", part):
            current_heading = part.strip("- ").strip()
            continue

        # Split on heading-like lines within the page
        lines = part.split("\n")
        for line in lines:
            stripped = line.strip()
            if HEADING_PATTERN.match(stripped) and len(stripped) < 120:
                if current_body:
                    sections.append((current_heading, "\n".join(current_body).strip()))
                    current_body = []
                current_heading = stripped
            else:
                current_body.append(line)

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return [(h, b) for h, b in sections if b.strip()]


# ── Sentence-aware chunker ────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    min_size: int = settings.MIN_CHUNK_SIZE,
) -> List[str]:
    """
    Split text into overlapping chunks respecting sentence boundaries.
    Never cuts mid-sentence.
    """
    sentences = SENTENCE_END.split(text)
    chunks: List[str] = []
    current_tokens = 0
    current_sentences: List[str] = []
    overlap_buffer: List[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_tokens = _approx_tokens(sentence)

        if current_tokens + s_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences)
            if _approx_tokens(chunk_text) >= min_size:
                chunks.append(chunk_text)

            # Build overlap: keep last N tokens worth of sentences
            overlap_buffer = []
            overlap_tokens = 0
            for prev in reversed(current_sentences):
                pt = _approx_tokens(prev)
                if overlap_tokens + pt <= overlap:
                    overlap_buffer.insert(0, prev)
                    overlap_tokens += pt
                else:
                    break

            current_sentences = overlap_buffer.copy()
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += s_tokens

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if _approx_tokens(chunk_text) >= min_size:
            chunks.append(chunk_text)

    return chunks


# ── Semantic labelling ────────────────────────────────────────────────────────

LABEL_SYSTEM = """You are a legal document analyst.
Classify the provided text chunk into exactly one semantic category.
Return ONLY the category name, nothing else."""

LABEL_PROMPT = """Classify this legal document chunk into one of these categories:
background_facts, legal_claims, relief_requested, definitions,
procedural_history, evidence, parties_information, dates_timeline,
financial_terms, jurisdiction_venue, signatures_attestation, general

Text:
{text}

Category:"""


async def _label_chunk_batch(chunks: List[str]) -> List[str]:
    """Label a batch of chunks with semantic roles."""
    provider = get_provider()
    labels = []
    for chunk in chunks:
        try:
            label = await provider.generate(
                prompt=LABEL_PROMPT.format(text=chunk[:500]),
                system_prompt=LABEL_SYSTEM,
                max_tokens=20,
                temperature=0.0,
            )
            label = label.strip().lower().replace(" ", "_")
            if label not in SEMANTIC_LABELS:
                label = "general"
        except Exception:
            label = "general"
        labels.append(label)
    return labels


# ── Page number mapping ───────────────────────────────────────────────────────

def _find_page_numbers(chunk_text: str, full_text: str, pages_text: List[str]) -> List[int]:
    """Determine which pages a chunk spans."""
    page_nums = []
    for i, page_text in enumerate(pages_text):
        # Check if any 50-char window of the chunk appears in this page
        sample = chunk_text[:50].strip()
        if sample and sample in page_text:
            page_nums.append(i + 1)
    return page_nums if page_nums else [1]


# ── Main chunking function ────────────────────────────────────────────────────

async def chunk_document(doc: ProcessedDocument) -> List[DocumentChunk]:
    """
    Convert a ProcessedDocument into semantically-chunked DocumentChunks
    with rich metadata for downstream retrieval and filtering.
    """
    logger.info(f"Chunking document: {doc.filename} ({doc.doc_id})")

    pages_text = [p.cleaned_text for p in doc.pages]
    sf: StructuredFields = doc.structured_fields

    # ── Split into sections ───────────────────────────────────────────────
    sections = _split_into_sections(doc.full_text)
    logger.debug(f"  Sections found: {len(sections)}")

    # ── Chunk each section ────────────────────────────────────────────────
    raw_chunks: List[Tuple[Optional[str], str]] = []  # (heading, chunk_text)
    for heading, body in sections:
        sub_chunks = _chunk_text(body)
        for sc in sub_chunks:
            raw_chunks.append((heading, sc))

    logger.info(f"  Raw chunks: {len(raw_chunks)}")

    # ── Semantic labelling (batch) ────────────────────────────────────────
    chunk_texts = [c for _, c in raw_chunks]
    labels = await _label_chunk_batch(chunk_texts)

    # ── Build DocumentChunk objects ───────────────────────────────────────
    document_chunks: List[DocumentChunk] = []
    char_cursor = 0

    for idx, ((heading, text), label) in enumerate(zip(raw_chunks, labels)):
        page_nums = _find_page_numbers(text, doc.full_text, pages_text)

        # Detect special chunk types
        is_table = bool(re.search(r"\|.*\||\t.*\t", text))
        is_definition = label == "definitions" or bool(
            re.search(r"\b(means|defined as|shall mean|refers to)\b", text, re.I)
        )
        is_date_heavy = label == "dates_timeline" or len(
            re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", text)
        ) >= 2

        metadata = ChunkMetadata(
            doc_id=doc.doc_id,
            filename=doc.filename,
            chunk_index=idx,
            page_numbers=page_nums,
            document_type=sf.document_type.value,
            case_number=sf.case_number,
            parties=sf.parties,
            jurisdiction=sf.jurisdiction,
            section_heading=heading,
            is_table=is_table,
            is_definition=is_definition,
            is_date_heavy=is_date_heavy,
            semantic_label=label,
            char_start=char_cursor,
            char_end=char_cursor + len(text),
            token_count=_approx_tokens(text),
        )

        document_chunks.append(DocumentChunk(text=text, metadata=metadata))
        char_cursor += len(text) + 1

    logger.info(
        f"  Chunks created: {len(document_chunks)} | "
        f"Labels: {set(labels)}"
    )
    return document_chunks
