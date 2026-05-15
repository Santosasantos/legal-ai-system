"""
FastAPI Routes
──────────────
POST /ingest          — Upload and process a document
POST /draft           — Generate a grounded draft
POST /edit            — Submit operator edit → extract preference rules
GET  /retrieve        — Retrieve relevant passages
GET  /documents       — List processed documents
GET  /documents/{id}  — Get document details
GET  /chunks/{id}     — Inspect a specific chunk (evidence tracing)
GET  /rules           — List active preference rules
DELETE /rules/{id}    — Deactivate a preference rule
GET  /health          — System health check
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    DraftRequest, DraftResponse, EditSubmitRequest, EditSubmitResponse,
    IngestResponse, RetrievalRequest, HealthResponse,
    OperatorEdit, DraftType, StructuredFields,
)
from app.services.document_processor import process_document
from app.services.chunker import chunk_document
from app.services.vector_store import (
    index_chunks, retrieve, get_chunk_by_id,
    delete_document_chunks, get_collection_stats, is_connected,
)
from app.services.draft_generator import generate_draft
from app.services.preference_store import (
    process_operator_edit, load_preference_rules, save_preference_rules,
)
from app.utils.document_store import (
    save_document_record, get_document_record,
    list_document_records, save_draft_record, get_draft_record,
)

logger = get_logger(__name__)
router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    from app.services.llm_provider import get_provider
    import httpx

    provider = get_provider()

    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        llm_provider=provider.provider_name,
        vision_model=provider.vision_model,
        text_model=provider.text_model,
        embed_model=provider.embed_model,
        chroma_connected=is_connected(),
        ollama_connected=ollama_ok,
    )


# ── Document Ingestion ────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(
    file: UploadFile = File(...),
    force_vision: bool = Form(False),
):
    """
    Upload a PDF document for processing.
    Handles scanned, noisy, and handwritten documents via OCR + Qwen2.5-VL.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    start = time.time()

    # Save uploaded file
    upload_path = Path(settings.SAMPLE_DOCS_DIR) / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded: {file.filename} ({upload_path.stat().st_size} bytes)")

    try:
        # Process document
        doc = await process_document(str(upload_path), force_vision=force_vision)

        # Chunk document
        chunks = await chunk_document(doc)

        # Index into vector store
        indexed = await index_chunks(chunks)

        # Persist document record
        save_document_record(doc)

        elapsed = time.time() - start
        return IngestResponse(
            doc_id=doc.doc_id,
            filename=doc.filename,
            total_pages=doc.total_pages,
            chunks_created=indexed,
            structured_fields=doc.structured_fields,
            warnings=doc.processing_warnings,
            processing_time_seconds=round(elapsed, 2),
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(500, f"Document processing failed: {str(e)}")


# ── Draft Generation ──────────────────────────────────────────────────────────

@router.post("/draft", response_model=DraftResponse, tags=["Drafts"])
async def create_draft(request: DraftRequest):
    """
    Generate a grounded legal draft for a processed document.
    The draft is anchored to retrieved evidence with full traceability.
    """
    start = time.time()

    doc_record = get_document_record(request.doc_id)
    if not doc_record:
        raise HTTPException(404, f"Document not found: {request.doc_id}")

    try:
        draft = await generate_draft(
            doc=doc_record,
            draft_type=request.draft_type,
            custom_query=request.custom_query,
        )

        # Persist draft
        save_draft_record(draft)

        elapsed = time.time() - start
        return DraftResponse(
            draft_id=draft.draft_id,
            doc_id=draft.doc_id,
            draft_type=draft.draft_type.value,
            content=draft.content,
            evidence_links=draft.evidence_links,
            model_used=draft.model_used,
            preference_rules_applied=draft.preference_rules_applied,
            generation_time_seconds=round(elapsed, 2),
        )

    except Exception as e:
        logger.error(f"Draft generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Draft generation failed: {str(e)}")


# ── Operator Edit Submission ──────────────────────────────────────────────────

@router.post("/edit", response_model=EditSubmitResponse, tags=["Improvement"])
async def submit_edit(request: EditSubmitRequest):
    """
    Submit an operator-edited draft.
    The system analyses the diff, extracts reusable preference rules,
    and stores them to improve future drafts.
    """
    edit = OperatorEdit(
        draft_id=request.draft_id,
        doc_id=request.doc_id,
        original_draft=request.original_draft,
        edited_draft=request.edited_draft,
        operator_notes=request.operator_notes,
    )

    try:
        new_rules = await process_operator_edit(edit)
        return EditSubmitResponse(
            edit_id=edit.edit_id,
            rules_extracted=len(new_rules),
            rules=new_rules,
            message=(
                f"Edit processed. {len(new_rules)} new preference rule(s) extracted "
                f"and will be applied to future drafts."
                if new_rules
                else "Edit processed. No new rules extracted (changes may be document-specific)."
            ),
        )
    except Exception as e:
        logger.error(f"Edit processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Edit processing failed: {str(e)}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

@router.get("/retrieve", tags=["Retrieval"])
async def retrieve_passages(
    query: str = Query(..., description="Search query"),
    doc_id: Optional[str] = Query(None, description="Filter to specific document"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    top_k: int = Query(8, ge=1, le=20),
):
    """
    Retrieve relevant passages from the vector store.
    Supports metadata filtering by doc_id and document_type.
    """
    try:
        result = await retrieve(
            query=query,
            doc_id=doc_id,
            document_type=document_type,
            top_k=top_k,
        )
        return {
            "query": result.query,
            "total_retrieved": result.total_retrieved,
            "passages": [
                {
                    "rank": p.rank,
                    "score": p.score,
                    "text": p.text,
                    "chunk_id": p.chunk_id,
                    "metadata": p.metadata,
                }
                for p in result.passages
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {str(e)}")


# ── Chunk Inspection ──────────────────────────────────────────────────────────

@router.get("/chunks/{chunk_id}", tags=["Retrieval"])
async def get_chunk(chunk_id: str):
    """
    Retrieve a specific chunk by ID for evidence inspection.
    Used to trace which source text supported a draft claim.
    """
    chunk = get_chunk_by_id(chunk_id)
    if not chunk:
        raise HTTPException(404, f"Chunk not found: {chunk_id}")
    return chunk


# ── Document Management ───────────────────────────────────────────────────────

@router.get("/documents", tags=["Documents"])
async def list_documents():
    """List all processed documents."""
    docs = list_document_records()
    return {
        "total": len(docs),
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "total_pages": d.total_pages,
                "document_type": d.structured_fields.document_type.value,
                "case_number": d.structured_fields.case_number,
                "parties": d.structured_fields.parties,
                "processing_timestamp": d.processing_timestamp.isoformat(),
                "warnings": len(d.processing_warnings),
            }
            for d in docs
        ],
    }


@router.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Get full details for a processed document."""
    doc = get_document_record(doc_id)
    if not doc:
        raise HTTPException(404, f"Document not found: {doc_id}")
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "total_pages": doc.total_pages,
        "structured_fields": doc.structured_fields.model_dump(),
        "processing_warnings": doc.processing_warnings,
        "processing_timestamp": doc.processing_timestamp.isoformat(),
        "page_summary": [
            {
                "page": p.page_number,
                "method": p.extraction_method.value,
                "confidence": p.confidence_score,
                "word_count": p.word_count,
            }
            for p in doc.pages
        ],
    }


@router.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its indexed chunks."""
    doc = get_document_record(doc_id)
    if not doc:
        raise HTTPException(404, f"Document not found: {doc_id}")
    deleted = delete_document_chunks(doc_id)
    return {"message": f"Deleted {deleted} chunks for document {doc_id}"}


# ── Preference Rules ──────────────────────────────────────────────────────────

@router.get("/rules", tags=["Improvement"])
async def get_rules():
    """List all active preference rules learned from operator edits."""
    rules = load_preference_rules()
    return {
        "total": len(rules),
        "active": sum(1 for r in rules if r.active),
        "rules": [r.model_dump() for r in rules],
    }


@router.delete("/rules/{rule_id}", tags=["Improvement"])
async def deactivate_rule(rule_id: str):
    """Deactivate a preference rule (it won't be applied to future drafts)."""
    rules = load_preference_rules()
    for rule in rules:
        if rule.rule_id == rule_id:
            rule.active = False
            save_preference_rules(rules)
            return {"message": f"Rule {rule_id} deactivated"}
    raise HTTPException(404, f"Rule not found: {rule_id}")


# ── Vector Store Stats ────────────────────────────────────────────────────────

@router.get("/stats", tags=["System"])
async def get_stats():
    """Get vector store and system statistics."""
    stats = get_collection_stats()
    rules = load_preference_rules()
    docs = list_document_records()
    return {
        "documents_processed": len(docs),
        "total_chunks_indexed": stats["total_chunks"],
        "preference_rules": len(rules),
        "active_rules": sum(1 for r in rules if r.active),
        "collection_name": stats["collection_name"],
    }
