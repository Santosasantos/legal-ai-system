"""
Draft Generation Service
─────────────────────────
Generates grounded legal-style drafts anchored to retrieved evidence.
Every claim in the draft is traceable to a source chunk.
Preference rules from operator edits are injected into the system prompt.
"""

import json
import re
import time
from typing import List, Optional, Dict

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    ProcessedDocument, GeneratedDraft, DraftType,
    RetrievedPassage, EvidenceLink, RetrievalResult,
)
from app.services.llm_provider import get_provider
from app.services.vector_store import retrieve
from app.services.preference_store import load_preference_rules

logger = get_logger(__name__)


# ── Draft-type query templates ────────────────────────────────────────────────

DRAFT_QUERIES: Dict[DraftType, List[str]] = {
    DraftType.CASE_FACT_SUMMARY: [
        "parties involved in the case",
        "key facts and background of the dispute",
        "legal claims and allegations",
        "relief requested and damages",
        "procedural history and court orders",
        "dates and timeline of events",
    ],
    DraftType.TITLE_REVIEW: [
        "property description and title information",
        "ownership history and chain of title",
        "encumbrances liens and easements",
        "title defects and exceptions",
        "legal descriptions and boundaries",
    ],
    DraftType.NOTICE_SUMMARY: [
        "notice recipient and sender",
        "subject matter of the notice",
        "deadlines and response requirements",
        "legal basis for the notice",
        "consequences of non-compliance",
    ],
    DraftType.DOCUMENT_CHECKLIST: [
        "required documents and exhibits",
        "missing or incomplete information",
        "signatures and attestations",
        "filing requirements and deadlines",
    ],
    DraftType.INTERNAL_MEMO: [
        "key facts and background",
        "legal issues and analysis",
        "recommendations and next steps",
        "risks and considerations",
    ],
}


# ── System prompts per draft type ─────────────────────────────────────────────

DRAFT_SYSTEM_PROMPTS: Dict[DraftType, str] = {
    DraftType.CASE_FACT_SUMMARY: """You are a senior legal analyst at Pearson Specter Litt.
Your task is to produce a precise, well-structured Case Fact Summary.

CRITICAL RULES:
1. Every factual statement MUST be directly supported by the provided evidence passages.
2. If information is not in the evidence, write "Not stated in available documents."
3. Do NOT invent facts, dates, names, or legal conclusions.
4. Use clear section headers.
5. Cite evidence by referencing [Evidence N] inline.
6. Flag any gaps or inconsistencies you notice.""",

    DraftType.TITLE_REVIEW: """You are a real estate attorney at Pearson Specter Litt.
Produce a Title Review Summary grounded strictly in the provided document evidence.
Flag any title defects, encumbrances, or missing information explicitly.""",

    DraftType.NOTICE_SUMMARY: """You are a legal analyst at Pearson Specter Litt.
Produce a Notice Summary that clearly identifies the parties, subject, deadlines,
and legal basis. Ground every statement in the provided evidence.""",

    DraftType.DOCUMENT_CHECKLIST: """You are a paralegal at Pearson Specter Litt.
Produce a Document Checklist based on what is present and what appears to be missing
from the provided documents. Be specific and actionable.""",

    DraftType.INTERNAL_MEMO: """You are a senior associate at Pearson Specter Litt.
Produce a first-pass Internal Memo covering the key facts, legal issues, and
recommended next steps. Ground every statement in the provided evidence.""",
}


# ── Draft templates ───────────────────────────────────────────────────────────

CASE_FACT_SUMMARY_TEMPLATE = """
Based on the following evidence passages from the legal documents, produce a Case Fact Summary.

EVIDENCE PASSAGES:
{evidence_block}

DOCUMENT METADATA:
- Document Type: {doc_type}
- Case Number: {case_number}
- Parties: {parties}
- Jurisdiction: {jurisdiction}
- Key Dates: {dates}

PREFERENCE RULES (apply these to improve the draft):
{preference_rules}

Produce the Case Fact Summary with these sections:
1. CASE OVERVIEW
2. PARTIES
3. BACKGROUND FACTS
4. LEGAL CLAIMS
5. RELIEF REQUESTED
6. KEY DATES & TIMELINE
7. GAPS & UNCERTAINTIES (information not found in documents)

For each factual statement, cite the supporting evidence as [Evidence N].
"""

GENERIC_DRAFT_TEMPLATE = """
Based on the following evidence passages from the legal documents, produce a {draft_type}.

EVIDENCE PASSAGES:
{evidence_block}

DOCUMENT METADATA:
- Document Type: {doc_type}
- Case Number: {case_number}
- Parties: {parties}

PREFERENCE RULES (apply these to improve the draft):
{preference_rules}

Produce a well-structured, grounded {draft_type}.
Cite supporting evidence as [Evidence N] for each factual claim.
If information is not available in the evidence, state "Not stated in available documents."
"""


# ── Evidence block builder ────────────────────────────────────────────────────

def _build_evidence_block(passages: List[RetrievedPassage]) -> str:
    lines = []
    for i, p in enumerate(passages, 1):
        meta = p.metadata
        source_info = f"[{meta.get('filename', 'unknown')}, p.{meta.get('page_numbers', '?')}]"
        label = meta.get("semantic_label", "general")
        lines.append(
            f"[Evidence {i}] (score={p.score:.3f}, label={label}, source={source_info})\n"
            f"{p.text}\n"
        )
    return "\n".join(lines)


# ── Evidence link extraction ──────────────────────────────────────────────────

def _extract_evidence_links(
    draft_content: str,
    passages: List[RetrievedPassage],
) -> List[EvidenceLink]:
    """
    Parse [Evidence N] citations in the draft and map them to source chunks.
    """
    links: List[EvidenceLink] = []
    # Find all sections (split by numbered headers or double newlines)
    sections = re.split(r"\n(?=\d+\.\s+[A-Z]|\#{1,3}\s)", draft_content)

    for section in sections:
        if not section.strip():
            continue
        # Find evidence references in this section
        refs = re.findall(r"\[Evidence (\d+)\]", section)
        if not refs:
            continue

        section_title = section.split("\n")[0].strip()[:100]
        supporting_ids = []
        supporting_texts = []

        for ref in refs:
            idx = int(ref) - 1
            if 0 <= idx < len(passages):
                supporting_ids.append(passages[idx].chunk_id)
                supporting_texts.append(passages[idx].text[:200])

        if supporting_ids:
            avg_score = sum(
                passages[int(r) - 1].score
                for r in refs
                if 0 <= int(r) - 1 < len(passages)
            ) / len(refs)
            links.append(EvidenceLink(
                section=section_title,
                supporting_chunk_ids=supporting_ids,
                supporting_texts=supporting_texts,
                confidence=round(avg_score, 3),
            ))

    return links


# ── Main generation function ──────────────────────────────────────────────────

async def generate_draft(
    doc: ProcessedDocument,
    draft_type: DraftType = DraftType.CASE_FACT_SUMMARY,
    custom_query: Optional[str] = None,
) -> GeneratedDraft:
    """
    Generate a grounded draft for the given document.
    1. Retrieve relevant passages using multiple targeted queries
    2. Load active preference rules
    3. Build prompt with evidence + rules
    4. Generate draft
    5. Extract evidence links for traceability
    """
    start_time = time.time()
    provider = get_provider()
    sf = doc.structured_fields

    logger.info(f"Generating {draft_type.value} for doc_id={doc.doc_id}")

    # ── Step 1: Multi-query retrieval ─────────────────────────────────────
    queries = DRAFT_QUERIES.get(draft_type, ["key facts and information"])
    if custom_query:
        queries = [custom_query] + queries[:3]

    all_passages: List[RetrievedPassage] = []
    seen_chunk_ids = set()

    for query in queries:
        result: RetrievalResult = await retrieve(
            query=query,
            doc_id=doc.doc_id,
            top_k=4,
            expand_context=True,
        )
        for p in result.passages:
            if p.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(p.chunk_id)
                all_passages.append(p)

    # Sort by score, keep top passages
    all_passages.sort(key=lambda p: p.score, reverse=True)
    top_passages = all_passages[:12]  # max 12 passages to stay in context

    logger.info(f"  Retrieved {len(top_passages)} unique passages")

    # ── Step 2: Load preference rules ────────────────────────────────────
    rules = load_preference_rules()
    active_rules = [r for r in rules if r.active]
    rules_text = "\n".join(
        f"- [{r.rule_category.upper()}] {r.rule_text}"
        for r in active_rules
    ) if active_rules else "No preference rules yet."
    applied_rule_ids = [r.rule_id for r in active_rules]

    logger.info(f"  Applying {len(active_rules)} preference rules")

    # ── Step 3: Build prompt ──────────────────────────────────────────────
    evidence_block = _build_evidence_block(top_passages)

    if draft_type == DraftType.CASE_FACT_SUMMARY:
        prompt = CASE_FACT_SUMMARY_TEMPLATE.format(
            evidence_block=evidence_block,
            doc_type=sf.document_type.value,
            case_number=sf.case_number or "Not identified",
            parties=", ".join(sf.parties) if sf.parties else "Not identified",
            jurisdiction=sf.jurisdiction or "Not identified",
            dates=", ".join(sf.dates[:5]) if sf.dates else "Not identified",
            preference_rules=rules_text,
        )
    else:
        prompt = GENERIC_DRAFT_TEMPLATE.format(
            draft_type=draft_type.value.replace("_", " ").title(),
            evidence_block=evidence_block,
            doc_type=sf.document_type.value,
            case_number=sf.case_number or "Not identified",
            parties=", ".join(sf.parties) if sf.parties else "Not identified",
            preference_rules=rules_text,
        )

    system_prompt = DRAFT_SYSTEM_PROMPTS.get(draft_type, DRAFT_SYSTEM_PROMPTS[DraftType.CASE_FACT_SUMMARY])

    # ── Step 4: Generate ──────────────────────────────────────────────────
    draft_content = await provider.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=settings.MAX_DRAFT_TOKENS,
        temperature=0.1,
    )

    # ── Step 5: Extract evidence links ────────────────────────────────────
    evidence_links = _extract_evidence_links(draft_content, top_passages)

    elapsed = time.time() - start_time
    logger.info(f"  Draft generated in {elapsed:.1f}s | Evidence links: {len(evidence_links)}")

    return GeneratedDraft(
        doc_id=doc.doc_id,
        draft_type=draft_type,
        content=draft_content,
        evidence_links=evidence_links,
        retrieved_passages=top_passages,
        model_used=f"{provider.provider_name}/{provider.text_model}",
        preference_rules_applied=applied_rule_ids,
    )
