"""
Unit and integration tests for the Legal AI pipeline.
Run with: pytest tests/ -v
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# ── Unit tests: Chunker ───────────────────────────────────────────────────────

from app.services.chunker import _chunk_text, _split_into_sections, _approx_tokens


def test_chunk_text_respects_sentence_boundaries():
    text = (
        "The plaintiff filed a complaint on January 1, 2024. "
        "The defendant responded on February 15, 2024. "
        "The court scheduled a hearing for March 10, 2024. "
        "Both parties agreed to mediation. "
        "The mediator was appointed on April 1, 2024."
    )
    chunks = _chunk_text(text, chunk_size=50, overlap=10, min_size=5)
    # Each chunk should end at a sentence boundary
    for chunk in chunks:
        assert not chunk.endswith(","), f"Chunk ends mid-sentence: {chunk[-30:]}"


def test_chunk_text_overlap():
    text = " ".join([f"Sentence {i} about legal matters." for i in range(50)])
    chunks = _chunk_text(text, chunk_size=100, overlap=20, min_size=10)
    assert len(chunks) > 1
    # Verify overlap: last words of chunk N should appear in chunk N+1
    if len(chunks) >= 2:
        last_words = chunks[0].split()[-3:]
        next_chunk_words = chunks[1].split()
        # At least some overlap should exist
        overlap_found = any(w in next_chunk_words for w in last_words)
        assert overlap_found, "No overlap detected between consecutive chunks"


def test_approx_tokens():
    assert _approx_tokens("") == 1
    assert _approx_tokens("hello") == 1
    assert _approx_tokens("a" * 400) == 100


def test_split_into_sections():
    text = """--- Page 1 ---
BACKGROUND
This is the background section.

FACTS
These are the facts.

--- Page 2 ---
CLAIMS
These are the claims."""
    sections = _split_into_sections(text)
    assert len(sections) >= 1
    # All sections should have non-empty body
    for heading, body in sections:
        assert body.strip()


# ── Unit tests: Preference Store ─────────────────────────────────────────────

from app.services.preference_store import _compute_diff, _extract_additions_deletions


def test_compute_diff_detects_changes():
    original = "The case involves breach of contract.\nDamages are $100,000."
    edited = "CASE NUMBER: 2024-CV-001\n\nThe case involves breach of contract.\nDamages are $100,000.\n\nDISCLAIMER: Draft only."
    diff = _compute_diff(original, edited)
    assert "+" in diff
    assert "CASE NUMBER" in diff or "DISCLAIMER" in diff


def test_extract_additions_deletions():
    diff = """--- original
+++ edited
@@ -1,2 +1,3 @@
+HEADER LINE
 unchanged line
-removed line
+added line"""
    additions, deletions = _extract_additions_deletions(diff)
    assert "HEADER LINE" in additions
    assert "added line" in additions
    assert "removed line" in deletions


def test_no_diff_on_identical():
    text = "The parties agree to the following terms."
    diff = _compute_diff(text, text)
    assert diff.strip() == ""


# ── Unit tests: Schemas ───────────────────────────────────────────────────────

from app.models.schemas import (
    DocumentChunk, ChunkMetadata, ProcessedDocument,
    StructuredFields, PageExtraction, ExtractionMethod,
    PreferenceRule, GeneratedDraft, DraftType,
)


def test_chunk_metadata_serialisation():
    meta = ChunkMetadata(
        doc_id="test-123",
        filename="test.pdf",
        chunk_index=0,
        page_numbers=[1, 2],
        document_type="case_file",
        parties=["Party A", "Party B"],
    )
    dumped = meta.model_dump()
    assert dumped["page_numbers"] == [1, 2]
    assert dumped["parties"] == ["Party A", "Party B"]


def test_preference_rule_defaults():
    rule = PreferenceRule(
        rule_text="Always include case number in header",
        rule_category="structure",
        source_edit_ids=["edit-1"],
        confidence=0.85,
    )
    assert rule.active is True
    assert rule.times_applied == 0
    assert rule.rule_id is not None


def test_structured_fields_defaults():
    sf = StructuredFields()
    assert sf.document_type.value == "unknown"
    assert sf.parties == []
    assert sf.dates == []
    assert sf.extraction_confidence == 0.0


# ── Integration test: Document store ─────────────────────────────────────────

import tempfile
import os


def test_document_store_save_load(tmp_path, monkeypatch):
    from app.utils import document_store
    monkeypatch.setattr(document_store, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(document_store, "DRAFTS_DIR", tmp_path / "drafts")

    from app.utils.document_store import save_document_record, get_document_record

    doc = ProcessedDocument(
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        total_pages=2,
        pages=[
            PageExtraction(
                page_number=1,
                raw_text="Test text",
                cleaned_text="Test text",
                extraction_method=ExtractionMethod.DIRECT_TEXT,
                confidence_score=0.95,
                word_count=2,
            )
        ],
        full_text="Test text",
        structured_fields=StructuredFields(
            document_type="case_file",
            case_number="2024-CV-001",
        ),
    )

    save_document_record(doc)
    loaded = get_document_record(doc.doc_id)

    assert loaded is not None
    assert loaded.doc_id == doc.doc_id
    assert loaded.filename == "test.pdf"
    assert loaded.structured_fields.case_number == "2024-CV-001"


# ── Unit tests: Vector store helpers (no ChromaDB connection needed) ──────────

def test_build_where_filter():
    """Test metadata filter builder — pure logic, no ChromaDB needed."""
    import sys
    from unittest.mock import MagicMock
    # Mock chromadb before importing vector_store
    sys.modules.setdefault("chromadb", MagicMock())
    sys.modules.setdefault("chromadb.config", MagicMock())

    from app.services.vector_store import _build_where_filter

    # Single filter
    f = _build_where_filter(doc_id="abc-123")
    assert f == {"doc_id": {"$eq": "abc-123"}}

    # Multiple filters → $and
    f = _build_where_filter(doc_id="abc-123", document_type="case_file")
    assert "$and" in f
    assert len(f["$and"]) == 2

    # No filters → None
    f = _build_where_filter()
    assert f is None


def test_serialise_deserialise_metadata():
    """Test metadata serialisation — pure logic, no ChromaDB needed."""
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("chromadb", MagicMock())
    sys.modules.setdefault("chromadb.config", MagicMock())

    from app.services.vector_store import _serialise_metadata, _deserialise_metadata

    original = {
        "doc_id": "test-123",
        "page_numbers": [1, 2, 3],
        "parties": ["Alice", "Bob"],
        "case_number": "2024-CV-001",
        "is_table": False,
        "chunk_index": 5,
    }

    serialised = _serialise_metadata(original)
    # Lists should be JSON strings
    assert isinstance(serialised["page_numbers"], str)
    assert isinstance(serialised["parties"], str)

    deserialised = _deserialise_metadata(serialised)
    assert deserialised["page_numbers"] == [1, 2, 3]
    assert deserialised["parties"] == ["Alice", "Bob"]
    assert deserialised["chunk_index"] == 5
