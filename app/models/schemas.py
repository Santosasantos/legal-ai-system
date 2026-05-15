"""
Pydantic schemas — the single source of truth for all data shapes
flowing through the pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


# ── Enums ────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    CONTRACT = "contract"
    CASE_FILE = "case_file"
    NOTICE = "notice"
    AFFIDAVIT = "affidavit"
    COURT_ORDER = "court_order"
    MEMO = "memo"
    UNKNOWN = "unknown"


class DraftType(str, Enum):
    CASE_FACT_SUMMARY = "case_fact_summary"
    TITLE_REVIEW = "title_review"
    NOTICE_SUMMARY = "notice_summary"
    DOCUMENT_CHECKLIST = "document_checklist"
    INTERNAL_MEMO = "internal_memo"


class ExtractionMethod(str, Enum):
    DIRECT_TEXT = "direct_text"       # clean PDF text layer
    OCR_STANDARD = "ocr_standard"     # pytesseract fallback
    VISION_LLM = "vision_llm"         # Qwen2.5-VL for complex/scanned


# ── Document Processing ───────────────────────────────────────────────────────

class PageExtraction(BaseModel):
    page_number: int
    raw_text: str
    cleaned_text: str
    extraction_method: ExtractionMethod
    confidence_score: float = Field(ge=0.0, le=1.0)
    has_tables: bool = False
    has_handwriting: bool = False
    word_count: int = 0


class StructuredFields(BaseModel):
    """Key structured fields extracted from the document."""
    document_type: DocumentType = DocumentType.UNKNOWN
    case_number: Optional[str] = None
    parties: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    judge_name: Optional[str] = None
    attorneys: List[str] = Field(default_factory=list)
    key_claims: List[str] = Field(default_factory=list)
    monetary_amounts: List[str] = Field(default_factory=list)
    statutes_cited: List[str] = Field(default_factory=list)
    exhibits: List[str] = Field(default_factory=list)
    summary_sentence: Optional[str] = None
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ProcessedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    total_pages: int
    pages: List[PageExtraction]
    full_text: str
    structured_fields: StructuredFields
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_warnings: List[str] = Field(default_factory=list)


# ── Chunking ──────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Rich metadata attached to every chunk — enables precise filtering."""
    doc_id: str
    filename: str
    chunk_index: int
    page_numbers: List[int]
    document_type: str
    case_number: Optional[str] = None
    parties: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    section_heading: Optional[str] = None
    is_table: bool = False
    is_definition: bool = False
    is_date_heavy: bool = False
    semantic_label: Optional[str] = None   # e.g. "facts", "claims", "relief"
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievedPassage(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int


class RetrievalResult(BaseModel):
    query: str
    passages: List[RetrievedPassage]
    total_retrieved: int
    retrieval_timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Draft Generation ──────────────────────────────────────────────────────────

class EvidenceLink(BaseModel):
    """Maps a draft section to the source passages that support it."""
    section: str
    supporting_chunk_ids: List[str]
    supporting_texts: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


class GeneratedDraft(BaseModel):
    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    draft_type: DraftType
    content: str
    evidence_links: List[EvidenceLink] = Field(default_factory=list)
    retrieved_passages: List[RetrievedPassage] = Field(default_factory=list)
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = ""
    preference_rules_applied: List[str] = Field(default_factory=list)


# ── Operator Edit & Improvement Loop ─────────────────────────────────────────

class OperatorEdit(BaseModel):
    edit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    draft_id: str
    doc_id: str
    original_draft: str
    edited_draft: str
    operator_notes: Optional[str] = None
    edit_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PreferenceRule(BaseModel):
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_text: str                          # natural-language instruction
    rule_category: str                      # style | structure | content | tone
    source_edit_ids: List[str]              # which edits produced this rule
    confidence: float = Field(ge=0.0, le=1.0)
    times_applied: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_applied: Optional[datetime] = None
    active: bool = True


class PreferenceRuleStore(BaseModel):
    rules: List[PreferenceRule] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# ── API Request / Response ────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    force_vision: bool = False   # force Qwen2.5-VL even on clean PDFs


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    chunks_created: int
    structured_fields: StructuredFields
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: float


class DraftRequest(BaseModel):
    doc_id: str
    draft_type: DraftType = DraftType.CASE_FACT_SUMMARY
    custom_query: Optional[str] = None


class DraftResponse(BaseModel):
    draft_id: str
    doc_id: str
    draft_type: str
    content: str
    evidence_links: List[EvidenceLink]
    model_used: str
    preference_rules_applied: List[str]
    generation_time_seconds: float


class EditSubmitRequest(BaseModel):
    draft_id: str
    doc_id: str
    original_draft: str
    edited_draft: str
    operator_notes: Optional[str] = None


class EditSubmitResponse(BaseModel):
    edit_id: str
    rules_extracted: int
    rules: List[PreferenceRule]
    message: str


class RetrievalRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None          # filter to one document
    document_type: Optional[str] = None   # metadata filter
    top_k: int = 8


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    vision_model: str
    text_model: str
    embed_model: str
    chroma_connected: bool
    ollama_connected: bool
