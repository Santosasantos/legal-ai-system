"""
Evaluation Script
──────────────────
Runs end-to-end evaluation of the system:
  1. Ingest all sample documents
  2. Generate drafts
  3. Measure retrieval quality (precision, recall, MRR)
  4. Measure draft grounding (evidence citation rate)
  5. Simulate operator edits and measure rule extraction
  6. Print a full evaluation report
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any

API_BASE = "http://localhost:8000/api/v1"
SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_documents"


def api_post(endpoint, json_data=None, files=None, data=None):
    if files:
        resp = requests.post(f"{API_BASE}{endpoint}", files=files, data=data, timeout=300)
    else:
        resp = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=300)
    resp.raise_for_status()
    return resp.json()


def api_get(endpoint, params=None):
    resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_retrieval(doc_id: str, test_queries: List[Dict]) -> Dict:
    """
    Evaluate retrieval quality using known query-answer pairs.
    Metrics: Hit@K, MRR, average score.
    """
    hits = 0
    mrr_sum = 0.0
    score_sum = 0.0
    total = len(test_queries)

    for tq in test_queries:
        result = api_get("/retrieve", {"query": tq["query"], "doc_id": doc_id, "top_k": 5})
        passages = result.get("passages", [])

        # Check if expected keyword appears in top-K results
        found_rank = None
        for p in passages:
            if any(kw.lower() in p["text"].lower() for kw in tq["expected_keywords"]):
                found_rank = p["rank"]
                break

        if found_rank:
            hits += 1
            mrr_sum += 1.0 / found_rank

        if passages:
            score_sum += passages[0]["score"]

    return {
        "hit_at_5": hits / total if total > 0 else 0,
        "mrr": mrr_sum / total if total > 0 else 0,
        "avg_top_score": score_sum / total if total > 0 else 0,
        "total_queries": total,
    }


def evaluate_draft_grounding(draft_content: str, evidence_links: List) -> Dict:
    """
    Measure how well the draft is grounded in evidence.
    """
    total_sections = max(1, draft_content.count("\n\n"))
    citation_count = draft_content.count("[Evidence ")
    sections_with_evidence = len(evidence_links)

    return {
        "citation_count": citation_count,
        "evidence_links": sections_with_evidence,
        "grounding_rate": min(1.0, citation_count / max(1, total_sections / 2)),
        "has_unsupported_claims": citation_count == 0,
    }


def simulate_operator_edit(original: str) -> str:
    """Simulate a realistic operator edit."""
    edited = original

    # Add case number to header if not present
    if "CASE NUMBER:" not in edited.upper():
        edited = "CASE NUMBER: 2024-CV-08847-SDNY\n\n" + edited

    # Replace generic language with more specific
    edited = edited.replace(
        "Not stated in available documents.",
        "[REQUIRES MANUAL REVIEW — information not found in source documents]"
    )

    # Add a standard disclaimer
    if "DISCLAIMER" not in edited.upper():
        edited += "\n\n---\nDISCLAIMER: This is a first-pass AI-generated draft. All facts must be verified against source documents before use."

    return edited


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("LEGAL AI SYSTEM — EVALUATION REPORT")
    print("=" * 70)

    results = {}

    # ── 1. Health check ───────────────────────────────────────────────────
    print("\n[1/5] System Health Check")
    health = api_get("/health")
    print(f"  Provider:  {health['llm_provider']}")
    print(f"  Vision:    {health['vision_model']}")
    print(f"  Text:      {health['text_model']}")
    print(f"  ChromaDB:  {'✓' if health['chroma_connected'] else '✗'}")
    print(f"  Ollama:    {'✓' if health['ollama_connected'] else '✗'}")

    # ── 2. Document ingestion ─────────────────────────────────────────────
    print("\n[2/5] Document Ingestion")
    pdf_files = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("  No sample documents found. Run: python scripts/generate_sample_docs.py")
        return

    ingested_docs = []
    for pdf_path in pdf_files:
        print(f"  Ingesting: {pdf_path.name}...")
        t0 = time.time()
        with open(pdf_path, "rb") as f:
            result = api_post("/ingest", files={"file": (pdf_path.name, f, "application/pdf")}, data={"force_vision": "false"})
        elapsed = time.time() - t0
        ingested_docs.append(result)
        print(f"    ✓ {result['total_pages']} pages, {result['chunks_created']} chunks, "
              f"type={result['structured_fields']['document_type']}, "
              f"time={elapsed:.1f}s, warnings={len(result.get('warnings', []))}")

    results["ingestion"] = {
        "documents_processed": len(ingested_docs),
        "total_chunks": sum(d["chunks_created"] for d in ingested_docs),
        "avg_warnings": sum(len(d.get("warnings", [])) for d in ingested_docs) / len(ingested_docs),
    }

    # ── 3. Retrieval evaluation ───────────────────────────────────────────
    print("\n[3/5] Retrieval Quality Evaluation")
    complaint_doc = next((d for d in ingested_docs if "complaint" in d["filename"].lower()), ingested_docs[0])
    doc_id = complaint_doc["doc_id"]

    test_queries = [
        {"query": "breach of contract damages", "expected_keywords": ["breach", "contract", "damages", "4,200,000"]},
        {"query": "parties involved in the case", "expected_keywords": ["Pearson", "Hardman", "plaintiff", "defendant"]},
        {"query": "non-solicitation agreement", "expected_keywords": ["non-solicitation", "solicitation", "agreement"]},
        {"query": "trade secrets misappropriation", "expected_keywords": ["trade secret", "misappropriation", "client list"]},
        {"query": "relief requested injunction", "expected_keywords": ["injunctive", "relief", "damages"]},
    ]

    retrieval_metrics = evaluate_retrieval(doc_id, test_queries)
    print(f"  Hit@5:          {retrieval_metrics['hit_at_5']:.2%}")
    print(f"  MRR:            {retrieval_metrics['mrr']:.3f}")
    print(f"  Avg Top Score:  {retrieval_metrics['avg_top_score']:.3f}")
    results["retrieval"] = retrieval_metrics

    # ── 4. Draft generation ───────────────────────────────────────────────
    print("\n[4/5] Draft Generation & Grounding")
    t0 = time.time()
    draft_result = api_post("/draft", {
        "doc_id": doc_id,
        "draft_type": "case_fact_summary",
    })
    elapsed = time.time() - t0

    grounding = evaluate_draft_grounding(
        draft_result["content"],
        draft_result.get("evidence_links", []),
    )
    print(f"  Generation time:    {elapsed:.1f}s")
    print(f"  Evidence citations: {grounding['citation_count']}")
    print(f"  Evidence links:     {grounding['evidence_links']}")
    print(f"  Grounding rate:     {grounding['grounding_rate']:.2%}")
    print(f"  Model used:         {draft_result['model_used']}")
    results["draft_generation"] = {**grounding, "generation_time": elapsed}

    # ── 5. Improvement loop ───────────────────────────────────────────────
    print("\n[5/5] Operator Edit & Improvement Loop")
    original = draft_result["content"]
    edited = simulate_operator_edit(original)

    edit_result = api_post("/edit", {
        "draft_id": draft_result["draft_id"],
        "doc_id": doc_id,
        "original_draft": original,
        "edited_draft": edited,
        "operator_notes": "Always include case number in header; add disclaimer at end",
    })

    print(f"  Rules extracted: {edit_result['rules_extracted']}")
    for rule in edit_result.get("rules", []):
        print(f"    [{rule['rule_category'].upper()}] {rule['rule_text'][:70]}...")

    # Generate a second draft to verify rules are applied
    draft_result_2 = api_post("/draft", {
        "doc_id": doc_id,
        "draft_type": "case_fact_summary",
    })
    rules_applied = len(draft_result_2.get("preference_rules_applied", []))
    print(f"  Rules applied in next draft: {rules_applied}")
    results["improvement_loop"] = {
        "rules_extracted": edit_result["rules_extracted"],
        "rules_applied_next_draft": rules_applied,
    }

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Documents processed:     {results['ingestion']['documents_processed']}")
    print(f"Total chunks indexed:    {results['ingestion']['total_chunks']}")
    print(f"Retrieval Hit@5:         {results['retrieval']['hit_at_5']:.2%}")
    print(f"Retrieval MRR:           {results['retrieval']['mrr']:.3f}")
    print(f"Draft grounding rate:    {results['draft_generation']['grounding_rate']:.2%}")
    print(f"Evidence citations:      {results['draft_generation']['citation_count']}")
    print(f"Rules extracted:         {results['improvement_loop']['rules_extracted']}")
    print(f"Rules applied (next):    {results['improvement_loop']['rules_applied_next_draft']}")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "outputs" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
