"""
Document Processing Service
────────────────────────────
Pipeline:
  1. Try direct text extraction (PyMuPDF) — fast, lossless
  2. If page is image-only or low-confidence → pytesseract OCR
  3. If still low-confidence or handwriting detected → Qwen2.5-VL vision LLM
  4. Clean and normalise text
  5. Extract structured fields via LLM
"""

import re
import io
import os
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import fitz          # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    PageExtraction, ProcessedDocument, StructuredFields,
    ExtractionMethod, DocumentType,
)
from app.services.llm_provider import get_provider

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 60.0   # tesseract confidence below this → try vision
MIN_TEXT_CHARS_PER_PAGE = 50      # fewer chars → treat as image page
DPI_FOR_OCR = 300                 # render resolution for OCR


# ── Image pre-processing helpers ─────────────────────────────────────────────

def _preprocess_image(img: Image.Image) -> Image.Image:
    """Enhance image quality before OCR."""
    img = img.convert("L")                                  # greyscale
    img = ImageEnhance.Contrast(img).enhance(2.0)           # boost contrast
    img = ImageEnhance.Sharpness(img).enhance(2.0)          # sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))      # denoise
    return img


def _tesseract_with_confidence(img: Image.Image) -> Tuple[str, float]:
    """Run tesseract and return (text, mean_confidence)."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = [w for w in data["text"] if w.strip()]
    confs = [c for c, w in zip(data["conf"], data["text"]) if w.strip() and c != -1]
    text = " ".join(words)
    confidence = sum(confs) / len(confs) if confs else 0.0
    return text, confidence


# ── Page-level extraction ─────────────────────────────────────────────────────

async def _extract_page(
    page: fitz.Page,
    page_num: int,
    force_vision: bool = False,
) -> PageExtraction:
    """Extract text from a single PDF page using the best available method."""
    provider = get_provider()

    # ── Step 1: Direct text layer ─────────────────────────────────────────
    raw_text = page.get_text("text").strip()
    method = ExtractionMethod.DIRECT_TEXT
    confidence = 1.0

    if len(raw_text) >= MIN_TEXT_CHARS_PER_PAGE and not force_vision:
        logger.debug(f"Page {page_num}: direct text ({len(raw_text)} chars)")
    else:
        # ── Step 2: Render page to image ──────────────────────────────────
        mat = fitz.Matrix(DPI_FOR_OCR / 72, DPI_FOR_OCR / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_preprocessed = _preprocess_image(img)

        # ── Step 3: Tesseract OCR ─────────────────────────────────────────
        ocr_text, ocr_conf = _tesseract_with_confidence(img_preprocessed)
        logger.debug(f"Page {page_num}: tesseract confidence={ocr_conf:.1f}")

        if ocr_conf >= OCR_CONFIDENCE_THRESHOLD and not force_vision:
            raw_text = ocr_text
            method = ExtractionMethod.OCR_STANDARD
            confidence = ocr_conf / 100.0
        else:
            # ── Step 4: Qwen2.5-VL vision LLM ────────────────────────────
            logger.info(f"Page {page_num}: low OCR confidence ({ocr_conf:.1f}) → Qwen2.5-VL")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, "PNG")
                tmp_path = tmp.name
            try:
                vision_prompt = (
                    "You are a legal document OCR specialist. "
                    "Extract ALL text from this document image exactly as it appears. "
                    "Preserve paragraph structure, headings, dates, names, case numbers, "
                    "and any handwritten annotations. "
                    "If text is partially illegible, mark it as [ILLEGIBLE]. "
                    "Output only the extracted text, nothing else."
                )
                raw_text = await provider.generate_with_image(
                    prompt="Extract all text from this legal document page.",
                    image_path=tmp_path,
                    system_prompt=vision_prompt,
                )
                method = ExtractionMethod.VISION_LLM
                confidence = 0.85  # vision LLM is generally reliable
            finally:
                os.unlink(tmp_path)

    # ── Step 5: Clean text ────────────────────────────────────────────────
    cleaned = _clean_text(raw_text)

    # ── Detect features ───────────────────────────────────────────────────
    has_tables = bool(re.search(r"\|.*\||\t.*\t", raw_text))
    has_handwriting = method == ExtractionMethod.VISION_LLM and "[ILLEGIBLE]" in raw_text

    return PageExtraction(
        page_number=page_num,
        raw_text=raw_text,
        cleaned_text=cleaned,
        extraction_method=method,
        confidence_score=min(confidence, 1.0),
        has_tables=has_tables,
        has_handwriting=has_handwriting,
        word_count=len(cleaned.split()),
    )


def _clean_text(text: str) -> str:
    """Normalise extracted text."""
    # Remove null bytes and control chars (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace within lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


# ── Structured field extraction ───────────────────────────────────────────────

FIELD_EXTRACTION_SYSTEM = """You are a legal document analysis expert.
Extract structured information from the provided legal document text.
Return ONLY valid JSON matching the schema exactly. No markdown, no explanation.
If a field is not found, use null for strings or [] for lists.
Be precise — only extract information explicitly stated in the document."""

FIELD_EXTRACTION_PROMPT = """Extract the following fields from this legal document text:

{text}

Return JSON with exactly these keys:
{{
  "document_type": one of ["contract","case_file","notice","affidavit","court_order","memo","unknown"],
  "case_number": string or null,
  "parties": [list of party names],
  "dates": [list of dates mentioned, in original format],
  "jurisdiction": string or null,
  "judge_name": string or null,
  "attorneys": [list of attorney names],
  "key_claims": [list of main legal claims or allegations],
  "monetary_amounts": [list of dollar amounts or damages],
  "statutes_cited": [list of statutes, codes, or case citations],
  "exhibits": [list of exhibit references],
  "summary_sentence": one sentence summarising the document,
  "extraction_confidence": float between 0.0 and 1.0
}}"""


async def _extract_structured_fields(full_text: str) -> StructuredFields:
    """Use LLM to extract structured fields from document text."""
    provider = get_provider()
    # Use first 6000 chars to stay within context limits
    truncated = full_text[:6000]
    prompt = FIELD_EXTRACTION_PROMPT.format(text=truncated)

    try:
        raw = await provider.generate(
            prompt=prompt,
            system_prompt=FIELD_EXTRACTION_SYSTEM,
            max_tokens=1024,
            temperature=0.0,
        )
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        # Extract JSON object if there's surrounding text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        data = json.loads(raw)
        return StructuredFields(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Structured field extraction failed: {e}")
        return StructuredFields(
            extraction_confidence=0.0,
            summary_sentence="Extraction failed — manual review required.",
        )


# ── Main processor ────────────────────────────────────────────────────────────

async def process_document(
    file_path: str,
    force_vision: bool = False,
) -> ProcessedDocument:
    """
    Full document processing pipeline.
    Returns a ProcessedDocument ready for chunking and indexing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    logger.info(f"Processing document: {path.name}")
    warnings: List[str] = []

    # ── Open PDF ──────────────────────────────────────────────────────────
    try:
        pdf = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")

    total_pages = len(pdf)
    logger.info(f"  Pages: {total_pages}")

    # ── Extract each page ─────────────────────────────────────────────────
    pages: List[PageExtraction] = []
    for i, page in enumerate(pdf):
        try:
            page_extraction = await _extract_page(page, i + 1, force_vision)
            pages.append(page_extraction)
            if page_extraction.confidence_score < 0.5:
                warnings.append(
                    f"Page {i+1}: low extraction confidence ({page_extraction.confidence_score:.2f})"
                )
        except Exception as e:
            logger.error(f"Page {i+1} extraction failed: {e}")
            warnings.append(f"Page {i+1}: extraction failed — {e}")
            pages.append(PageExtraction(
                page_number=i + 1,
                raw_text="",
                cleaned_text="[PAGE EXTRACTION FAILED]",
                extraction_method=ExtractionMethod.DIRECT_TEXT,
                confidence_score=0.0,
            ))

    pdf.close()

    # ── Combine full text ─────────────────────────────────────────────────
    full_text = "\n\n".join(
        f"--- Page {p.page_number} ---\n{p.cleaned_text}"
        for p in pages
        if p.cleaned_text.strip()
    )

    # ── Extract structured fields ─────────────────────────────────────────
    logger.info("  Extracting structured fields...")
    structured_fields = await _extract_structured_fields(full_text)

    doc = ProcessedDocument(
        filename=path.name,
        file_path=str(path.absolute()),
        total_pages=total_pages,
        pages=pages,
        full_text=full_text,
        structured_fields=structured_fields,
        processing_warnings=warnings,
    )

    logger.info(
        f"  Done: {total_pages} pages, "
        f"doc_type={structured_fields.document_type}, "
        f"confidence={structured_fields.extraction_confidence:.2f}, "
        f"warnings={len(warnings)}"
    )
    return doc
