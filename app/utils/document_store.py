"""
Simple JSON-based document and draft persistence.
In production this would be a proper database.
"""

import json
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import ProcessedDocument, GeneratedDraft

logger = get_logger(__name__)

DOCS_DIR = Path(settings.OUTPUTS_DIR) / "documents"
DRAFTS_DIR = Path(settings.OUTPUTS_DIR) / "drafts"


def _ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)


def save_document_record(doc: ProcessedDocument) -> None:
    _ensure_dirs()
    path = DOCS_DIR / f"{doc.doc_id}.json"
    path.write_text(doc.model_dump_json(indent=2))
    logger.debug(f"Saved document record: {doc.doc_id}")


def get_document_record(doc_id: str) -> Optional[ProcessedDocument]:
    _ensure_dirs()
    path = DOCS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    try:
        return ProcessedDocument.model_validate_json(path.read_text())
    except Exception as e:
        logger.error(f"Failed to load document {doc_id}: {e}")
        return None


def list_document_records() -> List[ProcessedDocument]:
    _ensure_dirs()
    docs = []
    for path in sorted(DOCS_DIR.glob("*.json")):
        try:
            docs.append(ProcessedDocument.model_validate_json(path.read_text()))
        except Exception as e:
            logger.warning(f"Skipping corrupt document record {path.name}: {e}")
    return docs


def save_draft_record(draft: GeneratedDraft) -> None:
    _ensure_dirs()
    path = DRAFTS_DIR / f"{draft.draft_id}.json"
    path.write_text(draft.model_dump_json(indent=2))
    logger.debug(f"Saved draft record: {draft.draft_id}")


def get_draft_record(draft_id: str) -> Optional[GeneratedDraft]:
    _ensure_dirs()
    path = DRAFTS_DIR / f"{draft_id}.json"
    if not path.exists():
        return None
    try:
        return GeneratedDraft.model_validate_json(path.read_text())
    except Exception as e:
        logger.error(f"Failed to load draft {draft_id}: {e}")
        return None
