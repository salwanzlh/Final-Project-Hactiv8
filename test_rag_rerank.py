"""
test_rag_rerank.py
==================
Tests for the re-ranking layer added to the RAG pipeline (rag.py).

Re-rank strategy:
  1. Embed query  → vector search (top_k × RERANK_PREFETCH candidates)
  2. Cross-encoder (_rerank via Pinecone bge-reranker-v2-m3) reorders them
  3. Return top_k best; fallback to vector order if reranker fails

Sections:
  1. Correctness  — rerank changes result order as expected
  2. Fallback     — graceful degradation when reranker fails/errors
  3. Over-fetch   — Pinecone query requests top_k × RERANK_PREFETCH
  4. Doc text     — correct text built from metadata for cross-encoder
  5. Latency      — overhead measurement (embed / vector / rerank / total)

Output CSVs:
  test_results/test_rag_rerank_results.csv   ← pass/fail/duration (conftest.py)
  test_results/rag_rerank_detail.csv         ← per-query rerank metrics (this file)

Run:
  uv run pytest test_rag_rerank.py -v -s
"""

import asyncio
import csv
import time
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.rag import (
    search_similar_customers,
    search_conversation_patterns,
    _customer_to_text,
    RERANK_PREFETCH,
    MIN_SCORE,
)


# ── Detail CSV ────────────────────────────────────────────────────────────────

_DETAIL_CSV = Path("test_results/rag_rerank_detail.csv")
_CSV_FIELDS = [
    "test",
    "query",
    "n_candidates",
    "top_k",
    "vector_top1",
    "rerank_top1",
    "top1_changed",
    "embed_ms",
    "vector_ms",
    "rerank_ms",
    "total_ms",
    "notes",
    "timestamp",
]
_records: list[dict] = []


def _record(
    test: str,
    query: str = "",
    n_candidates: int = 0,
    top_k: int = 0,
    vector_top1: str = "",
    rerank_top1: str = "",
    top1_changed: bool = False,
    embed_ms: float = 0.0,
    vector_ms: float = 0.0,
    rerank_ms: float = 0.0,
    total_ms: float = 0.0,
    notes: str = "",
) -> None:
    _records.append({
        "test":          test,
        "query":         query[:80],
        "n_candidates":  n_candidates,
        "top_k":         top_k,
        "vector_top1":   vector_top1,
        "rerank_top1":   rerank_top1,
        "top1_changed":  top1_changed,
        "embed_ms":      round(embed_ms, 2),
        "vector_ms":     round(vector_ms, 2),
        "rerank_ms":     round(rerank_ms, 2),
        "total_ms":      round(total_ms, 2),
        "notes":         notes,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


@pytest.fixture(scope="session", autouse=True)
def write_detail_csv():
    yield
    if not _records:
        return
    _DETAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(_DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerows(_records)

    W = 74
    print(f"\n{'=' * W}")
    print(f"  Re-rank Detail  →  {_DETAIL_CSV.resolve()}")
    print(f"{'=' * W}")
    print(f"  {'Test':<44} {'top1_changed':>13} {'total_ms':>9}")
    print(f"  {'-'*44} {'-'*13} {'-'*9}")
    for r in _records:
        changed = "YES  ↑" if r["top1_changed"] else "no"
        ms = r["total_ms"]
        print(f"  {r['test'][:44]:<44} {changed:>13} {ms:>8.1f}ms")
    print(f"{'=' * W}\n")


# ── Mock data ─────────────────────────────────────────────────────────────────

MOCK_EMB = [0.1] * 1536

# 5 mock customers ordered by vector similarity (index 0 = highest cosine score).
# For a "mudik keluarga besar luar kota" query, index 2 is semantically best.
CUSTOMERS = [
    {   # 0 — vector rank 1 (cosine highest), but NOT best match for mudik query
        "pekerjaan": "Pegawai negeri", "usia": 42, "kota": "Bandung",
        "anggaran_min": 200, "anggaran_max": 280,
        "faktor_utama": "irit BBM, dalam kota", "tags": "irit,kota",
        "mobil_dibeli": "Xpander CVT", "outcome": "beli",
    },
    {   # 1
        "pekerjaan": "Wiraswasta", "usia": 38, "kota": "Surabaya",
        "anggaran_min": 300, "anggaran_max": 400,
        "faktor_utama": "offroad, prestise", "tags": "offroad,premium",
        "mobil_dibeli": "Pajero Sport Dakar", "outcome": "beli",
    },
    {   # 2 — cross-encoder should rank this #1 for "mudik keluarga besar luar kota"
        "pekerjaan": "Karyawan swasta", "usia": 35, "kota": "Jakarta Selatan",
        "anggaran_min": 250, "anggaran_max": 320,
        "faktor_utama": "mudik, luar kota, 7 kursi keluarga besar", "tags": "keluarga,mudik",
        "mobil_dibeli": "Xpander Cross", "outcome": "beli",
    },
    {   # 3
        "pekerjaan": "Dokter", "usia": 45, "kota": "Medan",
        "anggaran_min": 400, "anggaran_max": 600,
        "faktor_utama": "prestise, nyaman", "tags": "premium,kota",
        "mobil_dibeli": "Eclipse Cross", "outcome": "beli",
    },
    {   # 4
        "pekerjaan": "Pengusaha", "usia": 50, "kota": "Makassar",
        "anggaran_min": 500, "anggaran_max": 700,
        "faktor_utama": "mewah, 4WD", "tags": "premium,4wd",
        "mobil_dibeli": "Pajero Sport Ultimate", "outcome": "beli",
    },
]

# 5 mock conversation patterns — index 2 most relevant for "mudik luar kota" context
PATTERNS = [
    {   # 0
        "stage": "rapport",
        "sequence": "Customer: Halo\nSales: Selamat datang",
        "effective_next_question": "Ini ke sini sendiri atau bareng keluarga Pak?",
        "why_effective": "Cairkan suasana", "outcome": "customer_engaged",
    },
    {   # 1
        "stage": "probing",
        "sequence": "Customer: Saya cari MPV\nSales: Untuk berapa orang?",
        "effective_next_question": "Biasanya kalau pergi weekend, sendirian atau ada yang ikut?",
        "why_effective": "Reveal keluarga natural", "outcome": "sale_progressed",
    },
    {   # 2 — cross-encoder best match for mudik/luar kota context
        "stage": "probing",
        "sequence": "Customer: Sering mudik ke Jawa tiap lebaran\nSales: Jalannya bagaimana?",
        "effective_next_question": "Jalan yang biasa dilalui gimana, mulus atau medan berat?",
        "why_effective": "Gali mobilitas dari konteks mudik", "outcome": "sale_progressed",
    },
    {   # 3
        "stage": "closing",
        "sequence": "Customer: Mau beli bulan ini\nSales: Budget berapa?",
        "effective_next_question": "Sudah lama nimbang-nimbang atau baru mulai cari?",
        "why_effective": "Reveal urgency", "outcome": "customer_engaged",
    },
    {   # 4
        "stage": "early",
        "sequence": "Customer: Lihat-lihat dulu\nSales: Silakan",
        "effective_next_question": "Bapak/Ibu dari sini atau dari daerah lain?",
        "why_effective": "Icebreaker ringan", "outcome": "customer_engaged",
    },
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def configure_settings():
    with patch("backend.services.rag.settings") as s:
        s.pinecone_api_key    = "test-key"
        s.pinecone_index_name = "test-index"
        s.pinecone_namespace  = "customers-data"
        yield s


def _pinecone_mock(
    matches: list[dict],
    scores: list[float] | None = None,
    embed_delay: float = 0.010,
    query_delay: float = 0.015,
):
    """Build (mock_pc, mock_idx) with optional per-hop simulated delays."""
    if scores is None:
        scores = [max(MIN_SCORE + 0.05, 0.90 - i * 0.04) for i in range(len(matches))]

    mock_results         = MagicMock()
    mock_results.matches = []
    for meta, score in zip(matches, scores):
        m          = MagicMock()
        m.score    = score
        m.metadata = meta
        mock_results.matches.append(m)

    def _embed_side(*a, **kw):
        time.sleep(embed_delay)
        r        = MagicMock()
        r.values = MOCK_EMB
        return [r]

    def _query_side(*a, **kw):
        time.sleep(query_delay)
        return mock_results

    mock_pc                        = MagicMock()
    mock_pc.inference.embed.side_effect = _embed_side
    mock_idx                       = MagicMock()
    mock_idx.query.side_effect     = _query_side
    return mock_pc, mock_idx


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Correctness: rerank changes result order
# ══════════════════════════════════════════════════════════════════════════════

class TestRerankCorrectness:

    @pytest.mark.asyncio
    async def test_rerank_reorders_customers(self):
        """
        Vector order: [0,1,2,3,4].  Cross-encoder promotes index 2 to top.
        Expected output[0] == CUSTOMERS[2], output[1] == CUSTOMERS[0].
        """
        query  = "Saya sering mudik ke Jawa, anak tiga, butuh 7 kursi, anggaran 300 juta"
        top_k  = 2
        # Cross-encoder says: candidate 2 best, then 0
        rerank_indices = [2, 0]

        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  return_value=rerank_indices) as mock_rerank,
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers(query, top_k=top_k)
            total_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == top_k
        assert results[0] == CUSTOMERS[2], "Cross-encoder top pick must be CUSTOMERS[2]"
        assert results[1] == CUSTOMERS[0], "Cross-encoder second pick must be CUSTOMERS[0]"

        # Verify _rerank was called with the right arguments
        mock_rerank.assert_awaited_once()
        call_query, call_docs, call_top_n = (
            mock_rerank.call_args.args[0],
            mock_rerank.call_args.args[1],
            mock_rerank.call_args.kwargs.get("top_n") or mock_rerank.call_args.args[2],
        )
        assert call_query == query
        assert call_top_n == top_k
        expected_docs = [_customer_to_text(c) for c in CUSTOMERS]
        assert call_docs == expected_docs, "Doc texts must be _customer_to_text output"

        vector_top1 = CUSTOMERS[0].get("pekerjaan", "")
        rerank_top1 = results[0].get("pekerjaan", "")
        _record(
            test="test_rerank_reorders_customers",
            query=query,
            n_candidates=len(CUSTOMERS),
            top_k=top_k,
            vector_top1=vector_top1,
            rerank_top1=rerank_top1,
            top1_changed=(vector_top1 != rerank_top1),
            total_ms=total_ms,
            notes=f"rerank_indices={rerank_indices}",
        )
        print(f"\n  [REORDER] vector_top1={vector_top1!r} → rerank_top1={rerank_top1!r}")

    @pytest.mark.asyncio
    async def test_rerank_reorders_patterns(self):
        """
        Vector order: [0,1,2,3,4].  Cross-encoder promotes index 2 to top.
        Expected output[0] == PATTERNS[2], output[1] == PATTERNS[1].
        """
        query  = "Customer: Sering mudik ke Jawa tiap lebaran, jalannya lumayan berat"
        top_k  = 2
        rerank_indices = [2, 1]

        mock_pc, mock_idx = _pinecone_mock(PATTERNS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  return_value=rerank_indices) as mock_rerank,
        ):
            t0      = time.perf_counter()
            results = await search_conversation_patterns(query, top_k=top_k)
            total_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == top_k
        assert results[0] == PATTERNS[2]
        assert results[1] == PATTERNS[1]

        # Verify sequence (stored in metadata) is used as doc text, not effective_next_question
        call_docs = mock_rerank.call_args.args[1]
        assert call_docs[2] == PATTERNS[2]["sequence"], \
            "Pattern doc text must use 'sequence' field"

        vector_top1 = PATTERNS[0].get("effective_next_question", "")[:40]
        rerank_top1 = results[0].get("effective_next_question", "")[:40]
        _record(
            test="test_rerank_reorders_patterns",
            query=query,
            n_candidates=len(PATTERNS),
            top_k=top_k,
            vector_top1=vector_top1,
            rerank_top1=rerank_top1,
            top1_changed=(vector_top1 != rerank_top1),
            total_ms=total_ms,
            notes=f"rerank_indices={rerank_indices}",
        )
        print(f"\n  [REORDER] pattern vector_top1={vector_top1!r} → rerank_top1={rerank_top1!r}")

    @pytest.mark.asyncio
    async def test_rerank_returns_exactly_top_k(self):
        """
        10 candidates from vector, top_k=3.  Reranker returns 3 indices.
        Result length must be exactly 3.
        """
        candidates = CUSTOMERS * 2       # 10 items
        top_k      = 3
        rerank_indices = [7, 2, 5]      # cross-encoder picks 3

        mock_pc, mock_idx = _pinecone_mock(candidates)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, return_value=rerank_indices),
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers("family car", top_k=top_k)
            total_ms = (time.perf_counter() - t0) * 1000

        assert len(results) == top_k
        assert results[0] == candidates[7]
        assert results[1] == candidates[2]
        assert results[2] == candidates[5]

        _record(
            test="test_rerank_returns_exactly_top_k",
            query="family car",
            n_candidates=len(candidates),
            top_k=top_k,
            vector_top1=candidates[0].get("pekerjaan", ""),
            rerank_top1=results[0].get("pekerjaan", ""),
            top1_changed=True,
            total_ms=total_ms,
            notes="10 candidates → top_k=3",
        )

    @pytest.mark.asyncio
    async def test_rerank_no_change_when_vector_order_already_optimal(self):
        """
        When the cross-encoder agrees with vector order, results stay the same.
        """
        query  = "mobil SUV offroad Pajero"
        top_k  = 2
        # Reranker confirms vector order [0, 1]
        rerank_indices = [0, 1]

        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, return_value=rerank_indices),
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers(query, top_k=top_k)
            total_ms = (time.perf_counter() - t0) * 1000

        assert results[0] == CUSTOMERS[0]
        assert results[1] == CUSTOMERS[1]

        _record(
            test="test_rerank_no_change_when_already_optimal",
            query=query,
            n_candidates=len(CUSTOMERS),
            top_k=top_k,
            vector_top1=CUSTOMERS[0].get("pekerjaan", ""),
            rerank_top1=results[0].get("pekerjaan", ""),
            top1_changed=False,
            total_ms=total_ms,
            notes="reranker confirms vector order",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Fallback: graceful degradation when reranker fails
# ══════════════════════════════════════════════════════════════════════════════

class TestRerankFallback:

    @pytest.mark.asyncio
    async def test_rerank_error_falls_back_to_vector_order_customers(self):
        """
        _rerank raises RuntimeError → silently fall back to vector order[:top_k].
        No exception must escape to the caller.
        """
        query = "Keluarga besar sering luar kota"
        top_k = 2

        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  side_effect=RuntimeError("Reranker model unavailable")),
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers(query, top_k=top_k)
            total_ms = (time.perf_counter() - t0) * 1000

        # Must fall back gracefully — no crash, vector order preserved
        assert results == CUSTOMERS[:top_k], \
            "Fallback must return first top_k candidates in vector order"

        _record(
            test="test_rerank_error_fallback_customers",
            query=query,
            n_candidates=len(CUSTOMERS),
            top_k=top_k,
            vector_top1=CUSTOMERS[0].get("pekerjaan", ""),
            rerank_top1=results[0].get("pekerjaan", ""),
            top1_changed=False,
            total_ms=total_ms,
            notes="reranker=RuntimeError → fallback to vector[:top_k]",
        )
        print(f"\n  [FALLBACK] Reranker error → vector fallback in {total_ms:.1f}ms")

    @pytest.mark.asyncio
    async def test_rerank_error_falls_back_to_vector_order_patterns(self):
        """
        Same fallback behaviour for search_conversation_patterns.
        """
        query = "Customer: Saya mau lihat MPV\nSales: Untuk keluarga?"
        top_k = 2

        mock_pc, mock_idx = _pinecone_mock(PATTERNS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  side_effect=ConnectionError("Pinecone inference unreachable")),
        ):
            results = await search_conversation_patterns(query, top_k=top_k)

        assert results == PATTERNS[:top_k]
        _record(
            test="test_rerank_error_fallback_patterns",
            query=query[:60],
            n_candidates=len(PATTERNS),
            top_k=top_k,
            vector_top1=PATTERNS[0].get("stage", ""),
            rerank_top1=results[0].get("stage", ""),
            top1_changed=False,
            notes="reranker=ConnectionError → fallback to vector[:top_k]",
        )

    @pytest.mark.asyncio
    async def test_rerank_skipped_when_no_candidates(self):
        """
        When vector search returns nothing (all scores below MIN_SCORE),
        _rerank must never be called and [] is returned.
        """
        # Scores all below MIN_SCORE so candidates list will be empty
        low_scores = [MIN_SCORE - 0.05] * len(CUSTOMERS)
        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS, scores=low_scores)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock) as mock_rerank,
        ):
            results = await search_similar_customers("any query", top_k=3)

        assert results == []
        mock_rerank.assert_not_awaited()

        _record(
            test="test_rerank_skipped_when_no_candidates",
            query="any query",
            n_candidates=0,
            top_k=3,
            notes="all scores below MIN_SCORE=0.30 → _rerank never called",
        )
        print("\n  [SKIP] No candidates → _rerank not called ✓")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Over-fetch: Pinecone query uses top_k × RERANK_PREFETCH
# ══════════════════════════════════════════════════════════════════════════════

class TestOverfetchMultiplier:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 3, 5])
    async def test_customers_query_uses_prefetch_multiplier(self, top_k):
        """
        index.query must be called with top_k = requested_top_k × RERANK_PREFETCH.
        """
        expected_fetch_k = top_k * RERANK_PREFETCH
        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, return_value=[0]),
        ):
            await search_similar_customers("test", top_k=top_k)

        actual_fetch_k = mock_idx.query.call_args.kwargs.get("top_k")
        assert actual_fetch_k == expected_fetch_k, (
            f"Expected index.query(top_k={expected_fetch_k}), got top_k={actual_fetch_k}"
        )

        _record(
            test=f"test_customers_overfetch_top_k={top_k}",
            query="test",
            n_candidates=len(CUSTOMERS),
            top_k=top_k,
            notes=f"query top_k={actual_fetch_k} == {top_k}×RERANK_PREFETCH({RERANK_PREFETCH})",
        )
        print(f"\n  [OVERFETCH] top_k={top_k} → query fetches {actual_fetch_k} ✓")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [2, 4])
    async def test_patterns_query_uses_prefetch_multiplier(self, top_k):
        """
        Same over-fetch check for search_conversation_patterns.
        """
        expected_fetch_k = top_k * RERANK_PREFETCH
        mock_pc, mock_idx = _pinecone_mock(PATTERNS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, return_value=[0]),
        ):
            await search_conversation_patterns("test convo", top_k=top_k)

        actual_fetch_k = mock_idx.query.call_args.kwargs.get("top_k")
        assert actual_fetch_k == expected_fetch_k

        _record(
            test=f"test_patterns_overfetch_top_k={top_k}",
            query="test convo",
            top_k=top_k,
            notes=f"query top_k={actual_fetch_k} == {top_k}×{RERANK_PREFETCH}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Document text construction for cross-encoder
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentText:

    def test_customer_to_text_includes_all_key_fields(self):
        """_customer_to_text must include job, age, city, budget, factors, car bought."""
        c    = CUSTOMERS[2]
        text = _customer_to_text(c)

        assert c["pekerjaan"] in text,           "pekerjaan must appear in doc text"
        assert str(c["usia"])  in text,           "usia must appear in doc text"
        assert c["kota"]       in text,           "kota must appear in doc text"
        assert str(c["anggaran_min"]) in text,    "anggaran_min must appear in doc text"
        assert str(c["anggaran_max"]) in text,    "anggaran_max must appear in doc text"
        assert "mudik" in text,                   "faktor_utama content must appear"
        print(f"\n  [DOC TEXT] {text!r}")

    def test_customer_to_text_handles_missing_fields(self):
        """_customer_to_text must not crash on partial metadata (graceful empty)."""
        sparse = {"pekerjaan": "Guru", "usia": 30}
        text   = _customer_to_text(sparse)
        assert "Guru" in text
        assert "30"   in text

    def test_customer_to_text_produces_distinct_text_per_customer(self):
        """Each customer in CUSTOMERS must produce a unique doc text."""
        texts = [_customer_to_text(c) for c in CUSTOMERS]
        assert len(texts) == len(set(texts)), "All customer doc texts must be unique"

    @pytest.mark.asyncio
    async def test_patterns_use_sequence_field_as_doc_text(self):
        """
        search_conversation_patterns must pass metadata['sequence'] to _rerank,
        not effective_next_question.  Sequence gives the conversation context
        the cross-encoder needs to judge relevance.
        """
        mock_pc, mock_idx = _pinecone_mock(PATTERNS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  return_value=[0]) as mock_rerank,
        ):
            await search_conversation_patterns("Customer: saya sering mudik", top_k=1)

        doc_texts = mock_rerank.call_args.args[1]
        for i, (pattern, doc) in enumerate(zip(PATTERNS, doc_texts)):
            expected = pattern.get("sequence") or pattern.get("effective_next_question", "")
            assert doc == expected, (
                f"Pattern[{i}] doc text mismatch: expected {expected!r}, got {doc!r}"
            )

        _record(
            test="test_patterns_use_sequence_as_doc_text",
            query="Customer: saya sering mudik",
            n_candidates=len(PATTERNS),
            top_k=1,
            notes="verified _rerank receives sequence field",
        )

    @pytest.mark.asyncio
    async def test_customers_use_customer_to_text_as_doc_text(self):
        """
        search_similar_customers must pass _customer_to_text(m) for each candidate.
        """
        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock,
                  return_value=[0]) as mock_rerank,
        ):
            await search_similar_customers("test", top_k=1)

        doc_texts = mock_rerank.call_args.args[1]
        expected  = [_customer_to_text(c) for c in CUSTOMERS]
        assert doc_texts == expected

        _record(
            test="test_customers_use_customer_to_text_as_doc_text",
            query="test",
            n_candidates=len(CUSTOMERS),
            top_k=1,
            notes="verified _rerank receives _customer_to_text output",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Latency: overhead of adding re-rank step
# ══════════════════════════════════════════════════════════════════════════════

class TestRerankLatency:

    @pytest.mark.asyncio
    async def test_rerank_latency_overhead_customers(self):
        """
        Measure total latency with and without reranking.
        Rerank overhead must stay below 2× the vector-only latency.
        """
        query = "Keluarga besar sering mudik luar kota, anggaran 300 juta"
        top_k = 3

        mock_pc_v, mock_idx_v = _pinecone_mock(
            CUSTOMERS, embed_delay=0.010, query_delay=0.020,
        )
        mock_pc_r, mock_idx_r = _pinecone_mock(
            CUSTOMERS, embed_delay=0.010, query_delay=0.020,
        )

        # ── Vector-only baseline (no rerank) ──────────────────────────────
        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc_v),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx_v),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, side_effect=RuntimeError("skip")),
        ):
            t0 = time.perf_counter()
            await search_similar_customers(query, top_k=top_k)
            vector_only_ms = (time.perf_counter() - t0) * 1000

        # ── With rerank (15ms simulated rerank delay) ─────────────────────
        async def slow_rerank(q, docs, top_n):
            await asyncio.sleep(0.015)   # simulate cross-encoder latency
            return list(range(top_n))

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc_r),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx_r),
            patch("backend.services.rag._rerank", side_effect=slow_rerank),
        ):
            t0 = time.perf_counter()
            await search_similar_customers(query, top_k=top_k)
            with_rerank_ms = (time.perf_counter() - t0) * 1000

        overhead_ms = with_rerank_ms - vector_only_ms
        print(
            f"\n  [LATENCY] vector_only={vector_only_ms:.1f}ms  "
            f"with_rerank={with_rerank_ms:.1f}ms  "
            f"overhead={overhead_ms:.1f}ms"
        )

        _record(
            test="test_rerank_latency_overhead_customers",
            query=query,
            n_candidates=len(CUSTOMERS),
            top_k=top_k,
            vector_ms=vector_only_ms,
            rerank_ms=overhead_ms,
            total_ms=with_rerank_ms,
            notes=f"overhead={overhead_ms:.1f}ms (simulated rerank=15ms)",
        )

        # Overhead must not more than double vector-only time
        assert overhead_ms < vector_only_ms * 2, (
            f"Rerank overhead {overhead_ms:.1f}ms is too high "
            f"(vector baseline={vector_only_ms:.1f}ms)"
        )

    @pytest.mark.asyncio
    async def test_rerank_latency_overhead_patterns(self):
        """
        Same overhead measurement for search_conversation_patterns.
        """
        query = (
            "Customer: Saya bareng keluarga, sering mudik ke Jawa, "
            "jalannya lumayan\nSales: Wah sering luar kota ya Pak?"
        )
        top_k = 3

        mock_pc_v, mock_idx_v = _pinecone_mock(PATTERNS, embed_delay=0.010, query_delay=0.020)
        mock_pc_r, mock_idx_r = _pinecone_mock(PATTERNS, embed_delay=0.010, query_delay=0.020)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc_v),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx_v),
            patch("backend.services.rag._rerank",
                  new_callable=AsyncMock, side_effect=RuntimeError("skip")),
        ):
            t0 = time.perf_counter()
            await search_conversation_patterns(query, top_k=top_k)
            vector_only_ms = (time.perf_counter() - t0) * 1000

        async def slow_rerank(q, docs, top_n):
            await asyncio.sleep(0.015)
            return list(range(top_n))

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc_r),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx_r),
            patch("backend.services.rag._rerank", side_effect=slow_rerank),
        ):
            t0 = time.perf_counter()
            await search_conversation_patterns(query, top_k=top_k)
            with_rerank_ms = (time.perf_counter() - t0) * 1000

        overhead_ms = with_rerank_ms - vector_only_ms
        print(
            f"\n  [LATENCY] patterns vector_only={vector_only_ms:.1f}ms  "
            f"with_rerank={with_rerank_ms:.1f}ms  "
            f"overhead={overhead_ms:.1f}ms"
        )

        _record(
            test="test_rerank_latency_overhead_patterns",
            query=query[:80],
            n_candidates=len(PATTERNS),
            top_k=top_k,
            vector_ms=vector_only_ms,
            rerank_ms=overhead_ms,
            total_ms=with_rerank_ms,
            notes=f"overhead={overhead_ms:.1f}ms (simulated rerank=15ms)",
        )

        assert overhead_ms < vector_only_ms * 2

    @pytest.mark.asyncio
    async def test_concurrent_rerank_queries_total_latency(self):
        """
        5 concurrent queries (customers + patterns) run in parallel.
        Total wall-time must be < 3× a single call latency (not additive).
        """
        N = 5
        mock_pc, mock_idx = _pinecone_mock(CUSTOMERS, embed_delay=0.010, query_delay=0.015)

        async def instant_rerank(q, docs, top_n):
            return list(range(min(top_n, len(docs))))

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index",  return_value=mock_idx),
            patch("backend.services.rag._rerank", side_effect=instant_rerank),
        ):
            # Single call baseline
            t0 = time.perf_counter()
            await search_similar_customers("baseline", top_k=3)
            single_ms = (time.perf_counter() - t0) * 1000

            # N concurrent calls
            t0 = time.perf_counter()
            await asyncio.gather(*[
                search_similar_customers(f"concurrent query {i}", top_k=3)
                for i in range(N)
            ])
            concurrent_ms = (time.perf_counter() - t0) * 1000

        print(
            f"\n  [CONCURRENT] single={single_ms:.1f}ms  "
            f"{N}× concurrent={concurrent_ms:.1f}ms  "
            f"ratio={concurrent_ms/single_ms:.1f}×"
        )
        _record(
            test=f"test_concurrent_rerank_{N}x",
            query=f"{N} concurrent queries",
            n_candidates=len(CUSTOMERS),
            top_k=3,
            total_ms=concurrent_ms,
            notes=f"single={single_ms:.1f}ms, ratio={concurrent_ms/single_ms:.1f}×",
        )

        assert concurrent_ms < single_ms * 3, (
            f"{N} concurrent calls took {concurrent_ms:.1f}ms — "
            f"expected < {single_ms * 3:.1f}ms (3× single)"
        )
