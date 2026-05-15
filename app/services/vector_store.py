"""
Vector Store Service — ChromaDB
─────────────────────────────────
Features:
  • Semantic similarity search with cosine distance
  • Rich metadata filtering (doc_id, document_type, semantic_label, etc.)
  • Context-aware retrieval: expands retrieved chunks with neighbours
  • Hybrid scoring: semantic score + metadata relevance boost
  • Chunk inspection: retrieve by chunk_id for evidence tracing
"""

import json
from typing import List, Optional, Dict, Any

import chromadb

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    DocumentChunk, RetrievedPassage, RetrievalResult,
)
from app.services.llm_provider import get_provider

logger = get_logger(__name__)


# ── Client singleton ──────────────────────────────────────────────────────────

_chroma_client: Optional[chromadb.HttpClient] = None
_collection = None


def _get_client() -> chromadb.HttpClient:
    global _chroma_client
    if _chroma_client is None:
        host = settings.CHROMA_HOST
        port = settings.CHROMA_PORT
        # Support full URLs (e.g. https://legal-ai-chromadb.onrender.com)
        if host.startswith("http://") or host.startswith("https://"):
            import urllib.parse
            parsed = urllib.parse.urlparse(host)
            ssl = parsed.scheme == "https"
            host = parsed.hostname
            port = parsed.port or (443 if ssl else 8000)
            _chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
            )
        else:
            _chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
            )
        logger.info(f"ChromaDB connected: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
    return _chroma_client


def _get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection: {settings.CHROMA_COLLECTION_NAME} ({_collection.count()} docs)")
    return _collection


def is_connected() -> bool:
    try:
        _get_client().heartbeat()
        return True
    except Exception:
        return False


def _parse_chroma_host():
    host = settings.CHROMA_HOST
    if host.startswith("http://") or host.startswith("https://"):
        import urllib.parse
        parsed = urllib.parse.urlparse(host)
        ssl = parsed.scheme == "https"
        return parsed.hostname, parsed.port or (443 if ssl else 8000), ssl
    return host, settings.CHROMA_PORT, False


# ── Metadata serialisation ────────────────────────────────────────────────────

def _serialise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB only accepts str/int/float/bool metadata values."""
    serialised = {}
    for k, v in meta.items():
        if isinstance(v, list):
            serialised[k] = json.dumps(v)   # lists → JSON string
        elif isinstance(v, bool):
            serialised[k] = v
        elif isinstance(v, (int, float, str)):
            serialised[k] = v
        elif v is None:
            serialised[k] = ""
        else:
            serialised[k] = str(v)
    return serialised


def _deserialise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse the serialisation for list fields."""
    result = {}
    list_fields = {"page_numbers", "parties", "statutes_cited"}
    for k, v in meta.items():
        if k in list_fields and isinstance(v, str):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = [v] if v else []
        else:
            result[k] = v
    return result


# ── Indexing ──────────────────────────────────────────────────────────────────

async def index_chunks(chunks: List[DocumentChunk]) -> int:
    """Embed and index a list of DocumentChunks into ChromaDB."""
    if not chunks:
        return 0

    provider = get_provider()
    collection = _get_collection()

    # Batch embed
    texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = await provider.embed(texts)

    ids = [c.chunk_id for c in chunks]
    metadatas = [_serialise_metadata(c.metadata.model_dump()) for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(f"Indexed {len(chunks)} chunks. Collection total: {collection.count()}")
    return len(chunks)


# ── Metadata filter builder ───────────────────────────────────────────────────

def _build_where_filter(
    doc_id: Optional[str] = None,
    document_type: Optional[str] = None,
    semantic_label: Optional[str] = None,
    case_number: Optional[str] = None,
    is_table: Optional[bool] = None,
) -> Optional[Dict]:
    """Build a ChromaDB $and/$eq where filter from optional parameters."""
    conditions = []
    if doc_id:
        conditions.append({"doc_id": {"$eq": doc_id}})
    if document_type:
        conditions.append({"document_type": {"$eq": document_type}})
    if semantic_label:
        conditions.append({"semantic_label": {"$eq": semantic_label}})
    if case_number:
        conditions.append({"case_number": {"$eq": case_number}})
    if is_table is not None:
        conditions.append({"is_table": {"$eq": is_table}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Context-aware retrieval ───────────────────────────────────────────────────

async def retrieve(
    query: str,
    top_k: int = settings.RETRIEVAL_TOP_K,
    doc_id: Optional[str] = None,
    document_type: Optional[str] = None,
    semantic_label: Optional[str] = None,
    expand_context: bool = True,
) -> RetrievalResult:
    """
    Retrieve top-k relevant passages for a query.

    expand_context=True: for each retrieved chunk, also fetch its
    immediate neighbours (prev/next chunk) to provide surrounding context.
    This prevents cutting off important context at chunk boundaries.
    """
    provider = get_provider()
    collection = _get_collection()

    # Embed query
    query_embedding = (await provider.embed([query]))[0]

    where = _build_where_filter(
        doc_id=doc_id,
        document_type=document_type,
        semantic_label=semantic_label,
    )

    # Fetch more than top_k to allow for context expansion deduplication
    fetch_k = top_k * 2 if expand_context else top_k

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(fetch_k, max(1, collection.count())),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    passages: List[RetrievedPassage] = []
    seen_ids = set()

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for rank, (doc_text, meta, dist, chunk_id) in enumerate(
        zip(docs, metas, distances, ids)
    ):
        # Cosine distance → similarity score
        score = 1.0 - dist
        if score < settings.RETRIEVAL_SCORE_THRESHOLD:
            continue
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        deserialized_meta = _deserialise_metadata(meta)
        passages.append(RetrievedPassage(
            chunk_id=chunk_id,
            text=doc_text,
            score=round(score, 4),
            metadata=deserialized_meta,
            rank=rank + 1,
        ))

    # ── Context expansion: fetch neighbouring chunks ──────────────────────
    if expand_context and passages:
        expanded = await _expand_with_neighbours(passages, seen_ids, doc_id)
        passages.extend(expanded)

    # Re-rank by score and trim to top_k
    passages.sort(key=lambda p: p.score, reverse=True)
    passages = passages[:top_k]

    # Re-assign ranks
    for i, p in enumerate(passages):
        p.rank = i + 1

    return RetrievalResult(
        query=query,
        passages=passages,
        total_retrieved=len(passages),
    )


async def _expand_with_neighbours(
    passages: List[RetrievedPassage],
    seen_ids: set,
    doc_id: Optional[str],
) -> List[RetrievedPassage]:
    """Fetch adjacent chunks for context continuity."""
    collection = _get_collection()
    extra: List[RetrievedPassage] = []

    for passage in passages[:4]:  # only expand top 4 to avoid bloat
        chunk_idx = passage.metadata.get("chunk_index")
        p_doc_id = passage.metadata.get("doc_id")
        if chunk_idx is None or p_doc_id is None:
            continue

        for neighbour_idx in [chunk_idx - 1, chunk_idx + 1]:
            if neighbour_idx < 0:
                continue
            where = {"$and": [
                {"doc_id": {"$eq": p_doc_id}},
                {"chunk_index": {"$eq": neighbour_idx}},
            ]}
            try:
                result = collection.get(where=where, include=["documents", "metadatas"])
                if result["ids"]:
                    nid = result["ids"][0]
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        extra.append(RetrievedPassage(
                            chunk_id=nid,
                            text=result["documents"][0],
                            score=passage.score * 0.8,  # slight discount for neighbours
                            metadata=_deserialise_metadata(result["metadatas"][0]),
                            rank=999,
                        ))
            except Exception:
                pass

    return extra


# ── Chunk lookup by ID ────────────────────────────────────────────────────────

def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific chunk by its ID — used for evidence inspection."""
    collection = _get_collection()
    result = collection.get(
        ids=[chunk_id],
        include=["documents", "metadatas"],
    )
    if not result["ids"]:
        return None
    return {
        "chunk_id": chunk_id,
        "text": result["documents"][0],
        "metadata": _deserialise_metadata(result["metadatas"][0]),
    }


# ── Document deletion ─────────────────────────────────────────────────────────

def delete_document_chunks(doc_id: str) -> int:
    """Remove all chunks for a document from the vector store."""
    collection = _get_collection()
    result = collection.get(where={"doc_id": {"$eq": doc_id}})
    if result["ids"]:
        collection.delete(ids=result["ids"])
        logger.info(f"Deleted {len(result['ids'])} chunks for doc_id={doc_id}")
        return len(result["ids"])
    return 0


def get_collection_stats() -> Dict[str, Any]:
    """Return basic stats about the vector store."""
    collection = _get_collection()
    return {
        "total_chunks": collection.count(),
        "collection_name": settings.CHROMA_COLLECTION_NAME,
    }
