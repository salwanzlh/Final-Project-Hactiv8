"""
RAG service — dua jenis retrieval dari Pinecone:

1. search_similar_customers()      namespace: customers-data
   → profil pelanggan historis yang mirip situasi customer sekarang

2. search_conversation_patterns()  namespace: conversation-patterns
   → pola percakapan historis yang mirip sequence saat ini
   → retrieve: pertanyaan apa yang efektif dilanjutkan di tahap ini

Schema dokumen conversation-patterns (untuk di-upsert):
{
  "sequence": "Customer: Mau lihat-lihat\nSales: Silakan Pak\nCustomer: Cari MPV keluarga",
  "stage": "rapport",                         # early | rapport | probing | closing
  "effective_next_question": "Biasanya pergi bareng berapa orang Pak?",
  "why_effective": "Membuka cerita keluarga secara natural",
  "outcome": "customer_engaged"               # customer_engaged | question_deflected | sale_progressed
}

Uses Pinecone's integrated inference (llama-text-embed-v2) to match the model
used when the index was originally seeded.
"""
import time
import logging
import asyncio
from functools import lru_cache
from backend.config import settings
from backend.services.metrics import rag_latency, rag_fallback_total, rag_results_returned

logger = logging.getLogger(__name__)

EMBEDDING_MODEL          = "llama-text-embed-v2"
MIN_SCORE                = 0.30
CONVERSATION_PATTERNS_NS = "conversation-patterns"


@lru_cache(maxsize=1)
def _pinecone_client():
    from pinecone import Pinecone
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache(maxsize=1)
def _pinecone_index():
    return _pinecone_client().Index(settings.pinecone_index_name)


def _embed_sync(text: str) -> list[float]:
    """Embed using Pinecone's inference API (same model used at upsert time)."""
    pc = _pinecone_client()
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[text[:8000]],
        parameters={"input_type": "query", "truncate": "END"},
    )
    return result[0].values


async def _embed(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_sync, text)



async def search_similar_customers(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Return metadata dicts for up to top_k historical customers similar to query_text.
    Returns [] silently if Pinecone is not configured or any error occurs.
    """
    _ns = settings.pinecone_namespace or "customers-data"
    if not query_text or not query_text.strip():
        return []
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        rag_fallback_total.labels(namespace=_ns).inc()
        return []

    _start = time.perf_counter()
    try:
        embedding = await _embed(query_text)
        index = await asyncio.to_thread(_pinecone_index)

        results = await asyncio.to_thread(
            index.query,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=_ns,
        )
        candidates = [m.metadata for m in results.matches if m.score >= MIN_SCORE and m.metadata]
        if not candidates:
            rag_latency.labels(namespace=_ns).observe(time.perf_counter() - _start)
            rag_results_returned.labels(namespace=_ns).observe(0)
            return []

        final = candidates[:top_k]

        rag_latency.labels(namespace=_ns).observe(time.perf_counter() - _start)
        rag_results_returned.labels(namespace=_ns).observe(len(final))
        return final

    except Exception as exc:
        logger.warning(f"[RAG] Pinecone lookup failed: {exc}")
        rag_fallback_total.labels(namespace=_ns).inc()
        return []


async def search_conversation_patterns(
    conversation_text: str,
    top_k: int = 3,
    stage: str | None = None,
) -> list[dict]:
    """
    Cari pola percakapan historis yang mirip sequence percakapan saat ini.

    Query: gabungan 3-5 utterance terakhir sebagai teks.
    stage: filter opsional — hanya ambil pola dengan stage yang cocok.
    Returns: list metadata dengan 'effective_next_question', 'stage', 'why_effective'.
    """
    if not conversation_text or not conversation_text.strip():
        return []
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        rag_fallback_total.labels(namespace=CONVERSATION_PATTERNS_NS).inc()
        return []

    _start = time.perf_counter()
    try:
        embedding = await _embed(conversation_text)
        index = await asyncio.to_thread(_pinecone_index)

        query_kwargs: dict = dict(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=CONVERSATION_PATTERNS_NS,
        )
        if stage:
            query_kwargs["filter"] = {"stage": {"$eq": stage}}

        results = await asyncio.to_thread(index.query, **query_kwargs)
        candidates = [m.metadata for m in results.matches if m.score >= MIN_SCORE and m.metadata]
        if not candidates:
            rag_latency.labels(namespace=CONVERSATION_PATTERNS_NS).observe(time.perf_counter() - _start)
            rag_results_returned.labels(namespace=CONVERSATION_PATTERNS_NS).observe(0)
            return []

        final = candidates[:top_k]

        rag_latency.labels(namespace=CONVERSATION_PATTERNS_NS).observe(time.perf_counter() - _start)
        rag_results_returned.labels(namespace=CONVERSATION_PATTERNS_NS).observe(len(final))
        return final

    except Exception as exc:
        logger.warning(f"[RAG/Patterns] Pinecone lookup failed: {exc}")
        rag_fallback_total.labels(namespace=CONVERSATION_PATTERNS_NS).inc()
        return []


def format_pattern_context(patterns: list[dict]) -> str:
    """Format retrieved conversation patterns into a prompt-ready block."""
    if not patterns:
        return ""

    lines = ["Pola percakapan historis yang mirip (pertanyaan yang efektif dilanjutkan):"]
    for i, p in enumerate(patterns, 1):
        stage    = p.get("stage", "?")
        question = p.get("effective_next_question", "")
        why      = p.get("why_effective", "")
        outcome  = p.get("outcome", "")

        outcome_label = {
            "customer_engaged":  "customer terbuka",
            "sale_progressed":   "percakapan maju",
            "question_deflected": "customer kurang respon",
        }.get(outcome, outcome)

        lines.append(
            f"{i}. [tahap: {stage}] \"{question}\""
            + (f" — {why}" if why else "")
            + (f" ({outcome_label})" if outcome_label else "")
        )
    return "\n".join(lines)


def format_rag_context(customers: list[dict]) -> str:
    """Format retrieved customers into a prompt-ready block."""
    if not customers:
        return ""

    lines = ["Pelanggan serupa dari riwayat historis showroom:"]
    for i, c in enumerate(customers, 1):
        anggaran = f"Rp {c.get('anggaran_min', '?')}-{c.get('anggaran_max', '?')} juta"
        outcome = c.get("outcome", "")

        if outcome == "beli":
            outcome_info = f"BELI: {c.get('mobil_dibeli', '?')}"
        else:
            outcome_info = "TIDAK JADI"

        faktor = c.get("faktor_utama", [])
        if isinstance(faktor, list):
            faktor_str = ", ".join(faktor)
        else:
            faktor_str = str(faktor)

        kompetitor = c.get("kompetitor", [])
        if isinstance(kompetitor, list):
            kompetitor = ", ".join(kompetitor)
        kompetitor_info = f" | kompetitor: {kompetitor}" if kompetitor else ""

        lines.append(
            f"{i}. {c.get('pekerjaan','?')}, {c.get('usia','?')} thn, {c.get('kota','?')}"
            f" | anggaran {anggaran}"
            f" | faktor: {faktor_str}"
            f"{kompetitor_info}"
            f" → {outcome_info}"
        )
    return "\n".join(lines)
