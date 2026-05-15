"""
Streamlit UI — Legal AI System
Pearson Specter Litt — Internal Document Workflow
"""

import os
import streamlit as st
import requests
import json
import time
from typing import Optional

API_BASE = os.environ.get("BACKEND_URL", "http://backend:8000") + "/api/v1"

st.set_page_config(
    page_title="PSL Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .evidence-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .rule-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    .warning-box {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    .draft-content {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        white-space: pre-wrap;
    }
    .stTextArea textarea {
        font-family: 'Georgia', serif;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the API running?"
    except Exception as e:
        return None, str(e)


def api_post(endpoint: str, json_data: dict = None, files=None, data=None):
    try:
        if files:
            resp = requests.post(f"{API_BASE}{endpoint}", files=files, data=data, timeout=300)
        else:
            resp = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=300)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the API running?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ PSL Legal AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📤 Ingest Document", "📝 Generate Draft", "✏️ Submit Edit", "🔍 Retrieve Passages", "📊 Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Health check
    health, err = api_get("/health")
    if health:
        st.success("✅ System Online")
        st.caption(f"Provider: **{health.get('llm_provider', '?')}**")
        st.caption(f"Vision: `{health.get('vision_model', '?')}`")
        st.caption(f"Text: `{health.get('text_model', '?')}`")
        chroma_ok = health.get("chroma_connected", False)
        ollama_ok = health.get("ollama_connected", False)
        st.caption(f"ChromaDB: {'✅' if chroma_ok else '❌'}")
        st.caption(f"Ollama: {'✅' if ollama_ok else '❌'}")
    else:
        st.error("❌ Backend Offline")
        st.caption(err)


# ── Page: Ingest Document ─────────────────────────────────────────────────────

if "📤 Ingest Document" in page:
    st.markdown('<div class="main-header"><h1>📤 Document Ingestion</h1><p>Upload messy legal documents — OCR, extraction, and indexing handled automatically</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a legal document (PDF)",
            type=["pdf"],
            help="Supports scanned PDFs, low-quality scans, and handwritten documents",
        )

    with col2:
        force_vision = st.checkbox(
            "Force Vision LLM (Qwen2.5-VL)",
            help="Force Qwen2.5-VL for all pages, even if text layer exists. Use for complex scanned documents.",
        )
        st.info("💡 Vision LLM is automatically used for low-quality pages.")

    if uploaded_file and st.button("🚀 Process Document", type="primary"):
        with st.spinner(f"Processing {uploaded_file.name}... (this may take a minute for scanned docs)"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"force_vision": str(force_vision).lower()}
            result, err = api_post("/ingest", files=files, data=data)

        if err:
            st.error(f"❌ Processing failed: {err}")
        else:
            st.success(f"✅ Document processed in {result['processing_time_seconds']}s")
            st.session_state["last_doc_id"] = result["doc_id"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pages", result["total_pages"])
            col2.metric("Chunks Indexed", result["chunks_created"])
            col3.metric("Doc Type", result["structured_fields"]["document_type"].replace("_", " ").title())
            col4.metric("Warnings", len(result.get("warnings", [])))

            st.markdown("### 📋 Extracted Structured Fields")
            sf = result["structured_fields"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Case Number:** {sf.get('case_number') or 'Not found'}")
                st.markdown(f"**Jurisdiction:** {sf.get('jurisdiction') or 'Not found'}")
                st.markdown(f"**Judge:** {sf.get('judge_name') or 'Not found'}")
                parties = sf.get("parties", [])
                st.markdown(f"**Parties:** {', '.join(parties) if parties else 'Not found'}")
            with col2:
                dates = sf.get("dates", [])
                st.markdown(f"**Dates:** {', '.join(dates[:3]) if dates else 'Not found'}")
                amounts = sf.get("monetary_amounts", [])
                st.markdown(f"**Amounts:** {', '.join(amounts[:3]) if amounts else 'Not found'}")
                statutes = sf.get("statutes_cited", [])
                st.markdown(f"**Statutes:** {', '.join(statutes[:3]) if statutes else 'Not found'}")

            if sf.get("summary_sentence"):
                st.info(f"📌 **Summary:** {sf['summary_sentence']}")

            if result.get("warnings"):
                st.markdown("### ⚠️ Processing Warnings")
                for w in result["warnings"]:
                    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

            st.markdown(f"**Document ID:** `{result['doc_id']}`")
            st.caption("Copy this ID to use in the Draft Generation tab.")


# ── Page: Generate Draft ──────────────────────────────────────────────────────

elif "📝 Generate Draft" in page:
    st.markdown('<div class="main-header"><h1>📝 Draft Generation</h1><p>Generate grounded legal drafts anchored to source evidence</p></div>', unsafe_allow_html=True)

    # Load documents
    docs_data, err = api_get("/documents")
    doc_options = {}
    if docs_data:
        for d in docs_data.get("documents", []):
            label = f"{d['filename']} ({d['document_type']}) — {d['doc_id'][:8]}..."
            doc_options[label] = d["doc_id"]

    col1, col2 = st.columns([2, 1])

    with col1:
        if doc_options:
            selected_label = st.selectbox("Select Document", list(doc_options.keys()))
            doc_id = doc_options[selected_label]
        else:
            doc_id = st.text_input("Document ID", value=st.session_state.get("last_doc_id", ""),
                                    placeholder="Paste document ID from ingestion step")

    with col2:
        draft_type = st.selectbox(
            "Draft Type",
            ["case_fact_summary", "title_review", "notice_summary", "document_checklist", "internal_memo"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    custom_query = st.text_input(
        "Custom Query (optional)",
        placeholder="e.g. 'What are the key financial terms?' — leave blank for default",
    )

    if st.button("⚡ Generate Draft", type="primary"):
        if not doc_id:
            st.error("Please select or enter a document ID.")
        else:
            with st.spinner("Retrieving evidence and generating draft..."):
                result, err = api_post("/draft", {
                    "doc_id": doc_id,
                    "draft_type": draft_type,
                    "custom_query": custom_query or None,
                })

            if err:
                st.error(f"❌ Generation failed: {err}")
            else:
                st.success(f"✅ Draft generated in {result['generation_time_seconds']}s using `{result['model_used']}`")

                # Store for edit tab
                st.session_state["last_draft_id"] = result["draft_id"]
                st.session_state["last_draft_content"] = result["content"]
                st.session_state["last_draft_doc_id"] = doc_id

                # Rules applied
                if result.get("preference_rules_applied"):
                    st.info(f"🎯 {len(result['preference_rules_applied'])} preference rule(s) applied from previous operator edits.")

                # Draft content
                st.markdown("### 📄 Generated Draft")
                st.markdown(f'<div class="draft-content">{result["content"]}</div>', unsafe_allow_html=True)

                # Evidence links
                if result.get("evidence_links"):
                    st.markdown("### 🔗 Evidence Traceability")
                    st.caption("Each section is linked to the source passages that support it.")
                    for link in result["evidence_links"]:
                        with st.expander(f"📎 {link['section'][:80]} (confidence: {link['confidence']:.2f})"):
                            for i, (cid, text) in enumerate(zip(link["supporting_chunk_ids"], link["supporting_texts"])):
                                st.markdown(f'<div class="evidence-box"><strong>Source {i+1}</strong> (chunk: <code>{cid[:12]}...</code>)<br>{text}</div>', unsafe_allow_html=True)

                st.markdown(f"**Draft ID:** `{result['draft_id']}`")
                st.caption("Go to 'Submit Edit' tab to improve this draft and train the system.")


# ── Page: Submit Edit ─────────────────────────────────────────────────────────

elif "✏️ Submit Edit" in page:
    st.markdown('<div class="main-header"><h1>✏️ Operator Edit & Learning</h1><p>Edit the draft and submit — the system learns your preferences for future drafts</p></div>', unsafe_allow_html=True)

    st.info("💡 Edit the draft below, then submit. The system will analyse your changes and extract reusable preference rules that improve all future drafts.")

    col1, col2 = st.columns(2)
    with col1:
        draft_id = st.text_input("Draft ID", value=st.session_state.get("last_draft_id", ""),
                                  placeholder="Draft ID from generation step")
    with col2:
        doc_id = st.text_input("Document ID", value=st.session_state.get("last_draft_doc_id", ""),
                                placeholder="Document ID")

    original_draft = st.text_area(
        "Original Draft (auto-filled from last generation)",
        value=st.session_state.get("last_draft_content", ""),
        height=300,
        key="original_draft_area",
    )

    edited_draft = st.text_area(
        "Your Edited Version ✏️",
        value=st.session_state.get("last_draft_content", ""),
        height=300,
        key="edited_draft_area",
        help="Make your edits here. The system will learn from the differences.",
    )

    operator_notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g. 'Always include case number in header' — helps the system understand your intent",
    )

    if st.button("📤 Submit Edit & Extract Rules", type="primary"):
        if not draft_id or not doc_id or not original_draft or not edited_draft:
            st.error("Please fill in all required fields.")
        elif original_draft == edited_draft:
            st.warning("No changes detected. Please edit the draft before submitting.")
        else:
            with st.spinner("Analysing your edits and extracting preference rules..."):
                result, err = api_post("/edit", {
                    "draft_id": draft_id,
                    "doc_id": doc_id,
                    "original_draft": original_draft,
                    "edited_draft": edited_draft,
                    "operator_notes": operator_notes or None,
                })

            if err:
                st.error(f"❌ Edit processing failed: {err}")
            else:
                st.success(result["message"])

                if result.get("rules"):
                    st.markdown("### 🧠 Preference Rules Extracted")
                    st.caption("These rules will be applied to all future drafts.")
                    for rule in result["rules"]:
                        category_emoji = {"style": "🎨", "structure": "🏗️", "content": "📝", "tone": "🎭", "format": "📐"}.get(rule["rule_category"], "💡")
                        st.markdown(
                            f'<div class="rule-box">{category_emoji} <strong>[{rule["rule_category"].upper()}]</strong> '
                            f'{rule["rule_text"]} <em>(confidence: {rule["confidence"]:.0%})</em></div>',
                            unsafe_allow_html=True,
                        )


# ── Page: Retrieve Passages ───────────────────────────────────────────────────

elif "🔍 Retrieve Passages" in page:
    st.markdown('<div class="main-header"><h1>🔍 Evidence Retrieval</h1><p>Search the document corpus with semantic + metadata filtering</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Search Query", placeholder="e.g. 'breach of contract damages' or 'settlement payment terms'")
    with col2:
        doc_filter = st.text_input("Filter by Doc ID (optional)", placeholder="doc_id...")
    with col3:
        top_k = st.slider("Top K", 1, 20, 8)

    doc_type_filter = st.selectbox(
        "Filter by Document Type (optional)",
        ["", "contract", "case_file", "notice", "affidavit", "court_order", "memo"],
        format_func=lambda x: x.replace("_", " ").title() if x else "All Types",
    )

    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            params = {"query": query, "top_k": top_k}
            if doc_filter:
                params["doc_id"] = doc_filter
            if doc_type_filter:
                params["document_type"] = doc_type_filter
            result, err = api_get("/retrieve", params=params)

        if err:
            st.error(f"❌ Retrieval failed: {err}")
        elif result:
            st.success(f"Found {result['total_retrieved']} relevant passages")
            for p in result["passages"]:
                meta = p["metadata"]
                with st.expander(
                    f"Rank {p['rank']} | Score: {p['score']:.3f} | "
                    f"{meta.get('filename', '?')} | "
                    f"Label: {meta.get('semantic_label', '?')} | "
                    f"Page: {meta.get('page_numbers', '?')}"
                ):
                    st.markdown(f'<div class="evidence-box">{p["text"]}</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.caption(f"Doc Type: {meta.get('document_type', '?')}")
                    col2.caption(f"Case #: {meta.get('case_number', 'N/A')}")
                    col3.caption(f"Chunk ID: `{p['chunk_id'][:16]}...`")


# ── Page: Dashboard ───────────────────────────────────────────────────────────

elif "📊 Dashboard" in page:
    st.markdown('<div class="main-header"><h1>📊 System Dashboard</h1><p>Overview of processed documents, indexed chunks, and learned preferences</p></div>', unsafe_allow_html=True)

    stats, err = api_get("/stats")
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents Processed", stats["documents_processed"])
        col2.metric("Chunks Indexed", stats["total_chunks_indexed"])
        col3.metric("Preference Rules", stats["preference_rules"])
        col4.metric("Active Rules", stats["active_rules"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📁 Processed Documents")
        docs_data, err = api_get("/documents")
        if docs_data and docs_data.get("documents"):
            for d in docs_data["documents"]:
                with st.expander(f"📄 {d['filename']}"):
                    st.markdown(f"**ID:** `{d['doc_id']}`")
                    st.markdown(f"**Type:** {d['document_type'].replace('_', ' ').title()}")
                    st.markdown(f"**Pages:** {d['total_pages']}")
                    st.markdown(f"**Case #:** {d.get('case_number') or 'N/A'}")
                    parties = d.get("parties", [])
                    st.markdown(f"**Parties:** {', '.join(parties[:2]) if parties else 'N/A'}")
                    if d.get("warnings", 0) > 0:
                        st.warning(f"⚠️ {d['warnings']} processing warning(s)")
        else:
            st.info("No documents processed yet. Go to 'Ingest Document' to get started.")

    with col2:
        st.markdown("### 🧠 Learned Preference Rules")
        rules_data, err = api_get("/rules")
        if rules_data and rules_data.get("rules"):
            for rule in rules_data["rules"]:
                if rule.get("active"):
                    category_emoji = {"style": "🎨", "structure": "🏗️", "content": "📝", "tone": "🎭", "format": "📐"}.get(rule.get("rule_category", ""), "💡")
                    st.markdown(
                        f'<div class="rule-box">{category_emoji} <strong>[{rule.get("rule_category", "?").upper()}]</strong> '
                        f'{rule["rule_text"]}<br>'
                        f'<small>Confidence: {rule["confidence"]:.0%} | Applied: {rule.get("times_applied", 0)}x</small></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No preference rules yet. Submit an edited draft to start learning.")
