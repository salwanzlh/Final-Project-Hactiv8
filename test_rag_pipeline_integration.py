"""
test_rag_pipeline_integration.py
=================================
Comprehensive integration & stress test suite for the RAG pipeline of the
Mitsubishi showroom AI sales-coach system.

Architecture under test:
  [Customer speech]
       ↓
  STT (OpenAI Whisper)
       ↓
  Embedding Generation  ← Pinecone inference (llama-text-embed-v2)
       ↓
  Vector Retrieval      ← Pinecone (customers-data + conversation-patterns)
       ↓
  LLM Orchestration     ← OpenAI GPT-4.1
       ↓
  _rag_fallback_hint    ← P1: topic-transition | P2: RAG patterns | P3: elicitation

Test sections:
  1. Latency & Performance Benchmarking
  2. Resilience & Circuit-Breaker Testing
  3. Fallback Cascade Validation (3 priorities)
  4. Output Format Completeness
  5. Stress / Concurrency Load

Run:
  uv run pytest test_rag_pipeline_integration.py -v --tb=short -s
"""

import asyncio
import csv
import json
import time
import uuid
import logging
import pytest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import (
    ConversationContext,
    Utterance,
    WsMessageType,
)
from backend.services.rag import (
    search_similar_customers,
    search_conversation_patterns,
    format_rag_context,
    format_pattern_context,
)
from backend.services.elicitation import compute_missing_dimensions, get_next_question
from backend.routers.ws import _rag_fallback_hint

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

EMBED_RETRIEVAL_TARGET_MS  = 100   # ideal P50 goal
EMBED_RETRIEVAL_WARNING_MS = 150   # strict CI warning boundary
LLM_MIN_S = 1.0                    # floor — below this something is wrong/mocked wrong
LLM_MAX_S = 3.0                    # ceiling under normal network conditions

SESSION_ID = "test-integration-session"

# ── Global perf accumulator (written to CSV in session teardown) ──────────────

_perf_results: list[dict] = []


def _record_perf(test_name: str, step: str, latency_ms: float, notes: str = "") -> None:
    _perf_results.append(
        {
            "test": test_name,
            "step": step,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
    )


# ── Domain fixture data (Mitsubishi automotive) ───────────────────────────────

MOCK_EMBEDDING_VECTOR = [0.05 * (i % 20) for i in range(1024)]  # fake 1024-dim

MOCK_CUSTOMER_MATCHES = [
    {
        "pekerjaan": "Wiraswasta",
        "usia": 38,
        "kota": "Jakarta Selatan",
        "anggaran_min": 300,
        "anggaran_max": 400,
        "faktor_utama": ["keluarga besar", "kenyamanan perjalanan jauh"],
        "kompetitor": ["Toyota Innova Zenix"],
        "outcome": "beli",
        "mobil_dibeli": "Xpander Ultimate CVT",
    },
    {
        "pekerjaan": "Pegawai swasta",
        "usia": 34,
        "kota": "Bekasi",
        "anggaran_min": 250,
        "anggaran_max": 320,
        "faktor_utama": ["irit BBM", "banyak kursi"],
        "kompetitor": [],
        "outcome": "beli",
        "mobil_dibeli": "Xpander Cross",
    },
]

MOCK_PATTERN_MATCHES = [
    {
        "stage": "probing",
        "effective_next_question": "Biasanya kalau pergi weekend, sendirian atau ada yang ikut?",
        "why_effective": "Membuka cerita keluarga secara natural",
        "outcome": "customer_engaged",
        "sequence": "Customer: Saya cari MPV\nSales: Silakan Pak",
    },
    {
        "stage": "rapport",
        "effective_next_question": "Ini ke sini sendiri atau bareng keluarga Pak?",
        "why_effective": "Cairkan suasana sekaligus gali situasi keluarga",
        "outcome": "sale_progressed",
        "sequence": "Customer: Mau lihat-lihat\nSales: Silakan",
    },
]

MOCK_LLM_RESPONSE_JSON = {
    "tahap": "PENDALAMAN",
    "hint_text": "Customer keluarga besar, sering mudik luar kota",
    "suggested_question": "Sudah lama pakai kendaraan yang sekarang?",
    "probe_topics": ["finansial", "urgency"],
    "detected_needs": ["keluarga besar", "mobilitas tinggi", "luar kota rutin"],
    "recommended_car_ids": [],
    "recommendation_reason": "",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_utterance(text: str, speaker: str = "customer") -> Utterance:
    return Utterance(
        id=str(uuid.uuid4()),
        speaker=speaker,
        text=text,
        timestamp=datetime.now(timezone.utc),
        confidence=0.9,
    )


def _make_context(
    texts: list[str] | None = None,
    asked: list[str] | None = None,
    needs: list[str] | None = None,
) -> ConversationContext:
    utterances = [
        _make_utterance(t)
        for t in (texts or ["Saya tertarik mobil keluarga untuk perjalanan jauh"])
    ]
    return ConversationContext(
        session_id=SESSION_ID,
        utterances=utterances,
        asked_questions=list(asked or []),
        detected_needs=list(needs or []),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def configure_pinecone_settings():
    """
    Inject fake Pinecone credentials so the guard
    `if not settings.pinecone_api_key` passes for all tests.
    Without this, every search_* call returns [] immediately.
    """
    with (
        patch("backend.services.rag.settings") as mock_rag,
        patch("backend.services.topic_patterns.settings") as mock_tp,
    ):
        mock_rag.pinecone_api_key   = "test-key"
        mock_rag.pinecone_index_name = "mitsubishi-customers"
        mock_rag.pinecone_namespace  = "customers-data"
        mock_tp.pinecone_api_key    = "test-key"
        mock_tp.pinecone_index_name = "mitsubishi-customers"
        yield mock_rag


@pytest.fixture
def mock_pinecone_index_factory():
    """
    Returns a factory that builds a fully-wired Pinecone mock pair (pc, index)
    with optional controlled delays and injectable side-effect exceptions.

    Usage:
        mock_pc, mock_idx = factory(embed_delay_s=0.030, query_delay_s=0.040)
        with patch("backend.services.rag._pinecone_client", return_value=mock_pc):
            with patch("backend.services.rag._pinecone_index", return_value=mock_idx):
                ...
    """
    def _factory(
        embed_delay_s: float = 0.0,
        query_delay_s: float = 0.0,
        matches: list | None = None,
        embed_side_effect: Exception | None = None,
        query_side_effect: Exception | None = None,
    ):
        mock_matches = []
        for m_data in (matches if matches is not None else MOCK_CUSTOMER_MATCHES):
            m = MagicMock()
            m.score    = 0.85
            m.metadata = m_data
            mock_matches.append(m)

        mock_results         = MagicMock()
        mock_results.matches = mock_matches

        def sync_embed(*args, **kwargs):
            time.sleep(embed_delay_s)
            er        = MagicMock()
            er.values = MOCK_EMBEDDING_VECTOR
            return [er]

        def sync_query(*args, **kwargs):
            time.sleep(query_delay_s)
            return mock_results

        mock_index = MagicMock()
        mock_index.query.side_effect = (
            query_side_effect if query_side_effect else sync_query
        )

        mock_pc = MagicMock()
        mock_pc.inference.embed.side_effect = (
            embed_side_effect if embed_side_effect else sync_embed
        )
        mock_pc.Index.return_value = mock_index

        return mock_pc, mock_index

    return _factory


@pytest.fixture
def mock_async_context_manager():
    """Return an async context manager mock usable with `async with`."""
    ctx          = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=None)
    ctx.__aexit__  = AsyncMock(return_value=False)
    return ctx


@pytest.fixture(scope="session", autouse=True)
def write_perf_csv():
    """
    Session-scoped fixture: after all tests complete, flush _perf_results
    to rag_pipeline_perf_report.csv and print a formatted table.
    """
    yield  # tests run here

    csv_path   = Path("rag_pipeline_perf_report.csv")
    fieldnames = ["test", "step", "latency_ms", "timestamp", "notes"]

    if not _perf_results:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_perf_results)

    sep = "=" * 80
    print(f"\n\n{sep}")
    print(f"  RAG Pipeline Performance Report  →  {csv_path.resolve()}")
    print(sep)
    print(f"  {'Test':<52} {'Step':<28} {'ms':>6}")
    print(f"  {'-'*52} {'-'*28} {'-'*6}")
    for row in _perf_results:
        lat = f"{row['latency_ms']:.2f}" if row["latency_ms"] else "  N/A"
        print(f"  {row['test'][:52]:<52} {row['step']:<28} {lat:>6}")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Latency & Performance Benchmarking
# ══════════════════════════════════════════════════════════════════════════════

class TestLatencyBenchmarks:
    """
    Every step is individually timed. Controlled delays simulate realistic
    network latency so we can assert threshold compliance in CI without
    requiring live Pinecone/OpenAI credentials.
    """

    @pytest.mark.asyncio
    async def test_embedding_generation_latency(self, mock_pinecone_index_factory):
        """
        Embedding alone (30 ms simulated) must stay under the warning boundary.
        Prints: '[PERF] Embedding latency: Xms'
        """
        mock_pc, _ = mock_pinecone_index_factory(embed_delay_s=0.030)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=MagicMock()),
        ):
            from backend.services.rag import _embed

            t0     = time.perf_counter()
            result = await _embed("Saya cari mobil keluarga untuk weekend")
            ms     = (time.perf_counter() - t0) * 1000

        print(f"\n  [PERF] Embedding latency: {ms:.1f}ms")
        _record_perf("test_embedding_generation_latency", "embedding", ms,
                     "simulated_delay=30ms")

        assert result == MOCK_EMBEDDING_VECTOR, "Embedding must return expected vector"
        assert ms < EMBED_RETRIEVAL_WARNING_MS, (
            f"Embedding took {ms:.1f}ms — exceeds {EMBED_RETRIEVAL_WARNING_MS}ms warning boundary"
        )

    @pytest.mark.asyncio
    async def test_embedding_plus_retrieval_combined_latency(self, mock_pinecone_index_factory):
        """
        Embed (30ms) + Pinecone query (40ms) = ~70ms combined.
        TARGET ≈ 100ms  |  WARNING boundary = 150ms.
        Also verifies correct result count is logged.
        """
        mock_pc, mock_idx = mock_pinecone_index_factory(
            embed_delay_s=0.030,
            query_delay_s=0.040,
            matches=MOCK_CUSTOMER_MATCHES,
        )

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers(
                "Keluarga besar, sering luar kota, butuh 7 kursi", top_k=3
            )
            ms = (time.perf_counter() - t0) * 1000

        print(
            f"\n  [PERF] Embed+Retrieval (customers-data): {ms:.1f}ms  |  "
            f"Retrieved {len(results)} chunks from Pinecone"
        )
        _record_perf(
            "test_embedding_plus_retrieval_combined_latency",
            "embed_plus_retrieval",
            ms,
            f"retrieved={len(results)}, target={EMBED_RETRIEVAL_TARGET_MS}ms",
        )

        if ms > EMBED_RETRIEVAL_TARGET_MS:
            print(
                f"  [WARN] {ms:.1f}ms exceeds target {EMBED_RETRIEVAL_TARGET_MS}ms "
                f"(still within warning {EMBED_RETRIEVAL_WARNING_MS}ms)"
            )

        assert len(results) > 0, "Must retrieve at least one customer match"
        assert ms < EMBED_RETRIEVAL_WARNING_MS, (
            f"Combined embed+retrieval {ms:.1f}ms > {EMBED_RETRIEVAL_WARNING_MS}ms boundary"
        )

    @pytest.mark.asyncio
    async def test_conversation_patterns_retrieval_latency(self, mock_pinecone_index_factory):
        """
        conversation-patterns namespace: embed (30ms) + query (35ms) = ~65ms.
        Verifies pattern count logged correctly.
        """
        mock_pc, mock_idx = mock_pinecone_index_factory(
            embed_delay_s=0.030,
            query_delay_s=0.035,
            matches=MOCK_PATTERN_MATCHES,
        )

        conversation = (
            "Customer: Saya tertarik SUV untuk keluarga\n"
            "Sales: Boleh cerita lebih lanjut Pak?\n"
            "Customer: Anak saya tiga, sering pergi bareng ke luar kota"
        )

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            t0       = time.perf_counter()
            patterns = await search_conversation_patterns(conversation, top_k=3)
            ms       = (time.perf_counter() - t0) * 1000

        print(
            f"\n  [PERF] Conversation-patterns retrieval: {ms:.1f}ms  |  "
            f"Retrieved {len(patterns)} chunks from Pinecone"
        )
        _record_perf(
            "test_conversation_patterns_retrieval_latency",
            "pattern_retrieval",
            ms,
            f"retrieved={len(patterns)}",
        )

        assert len(patterns) > 0, "Must retrieve at least one pattern"
        assert ms < EMBED_RETRIEVAL_WARNING_MS, (
            f"Pattern retrieval {ms:.1f}ms > {EMBED_RETRIEVAL_WARNING_MS}ms boundary"
        )

    @pytest.mark.asyncio
    async def test_llm_generation_latency(self, mock_async_context_manager):
        """
        LLM step simulated at 1.5s — must fall inside [LLM_MIN_S, LLM_MAX_S].
        Asserts that tahap and suggested_question are populated in the result.
        """
        from backend.services.ai import _openai_analyze

        llm_delay = 1.5
        context   = _make_context(
            texts=[
                "Customer: Saya mau cari mobil buat keluarga, anak saya tiga orang",
                "Sales: Wah, keluarga besar ya Pak. Biasanya bepergian ke mana?",
                "Customer: Sering ke luar kota, mudik ke Jawa Tengah tiap lebaran",
            ]
        )

        mock_response                        = MagicMock()
        mock_response.choices                = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(MOCK_LLM_RESPONSE_JSON)
        mock_response.usage.prompt_tokens    = 450
        mock_response.usage.completion_tokens = 80

        async def delayed_create(**kwargs):
            await asyncio.sleep(llm_delay)
            return mock_response

        mock_client                          = MagicMock()
        mock_client.chat.completions.create  = delayed_create

        mock_trace      = MagicMock()
        mock_generation = MagicMock()
        mock_trace.generation.return_value = mock_generation
        mock_langfuse   = MagicMock()
        mock_langfuse.trace.return_value = mock_trace

        mock_metrics = MagicMock()

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            patch("backend.services.ai.settings") as mock_ai_settings,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",   mock_langfuse),
            patch("backend.services.ai.llm_latency", MagicMock()),
            patch("backend.services.ai.llm_requests", mock_metrics),
            patch("backend.services.ai.token_per_request", mock_metrics),
            patch("backend.services.ai.track_latency",
                  return_value=mock_async_context_manager),
        ):
            mock_ai_settings.openai_api_key = "test-openai-key"

            t0         = time.perf_counter()
            hint, cars = await _openai_analyze(context)
            elapsed_s  = time.perf_counter() - t0

        ms = elapsed_s * 1000
        print(f"\n  [PERF] LLM generation latency: {elapsed_s:.3f}s")
        _record_perf(
            "test_llm_generation_latency",
            "llm_generation",
            ms,
            f"simulated={llm_delay}s, tahap={hint.tahap}",
        )

        assert LLM_MIN_S <= elapsed_s <= LLM_MAX_S, (
            f"LLM took {elapsed_s:.3f}s — expected [{LLM_MIN_S}s, {LLM_MAX_S}s]"
        )
        assert hint.suggested_question != "", "suggested_question must not be empty"
        assert hint.tahap == "PENDALAMAN", f"Unexpected tahap: {hint.tahap}"

    @pytest.mark.asyncio
    async def test_parallel_rag_retrieval_latency(self, mock_pinecone_index_factory):
        """
        asyncio.gather(customers, patterns) runs both legs in parallel.
        Simulates 30ms embed + 40ms query for each leg.
        Combined time should be < 2× warning_ms (not additive).
        Logs: "Customers: N, Patterns: M"
        """
        mock_pc, mock_idx = mock_pinecone_index_factory(
            embed_delay_s=0.030,
            query_delay_s=0.040,
        )

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            t0 = time.perf_counter()
            customers, patterns = await asyncio.gather(
                search_similar_customers("Keluarga besar sering mudik", top_k=3),
                search_conversation_patterns("Customer: saya tertarik SUV", top_k=3),
            )
            ms = (time.perf_counter() - t0) * 1000

        print(
            f"\n  [PERF] Parallel RAG retrieval: {ms:.1f}ms  |  "
            f"Customers: {len(customers)}, Patterns: {len(patterns)}"
        )
        _record_perf(
            "test_parallel_rag_retrieval_latency",
            "parallel_rag",
            ms,
            f"customers={len(customers)}, patterns={len(patterns)}",
        )

        # Sequential would be ~140ms; parallel target is under 2× warning
        assert ms < EMBED_RETRIEVAL_WARNING_MS * 2, (
            f"Parallel RAG took {ms:.1f}ms — unexpectedly slow"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Resilience & Circuit-Breaker Testing
# ══════════════════════════════════════════════════════════════════════════════

class TestResilienceCircuitBreaker:
    """
    Simulate failures at every hop. The system must degrade gracefully:
    - search_* returns [] on embedding/Pinecone errors
    - LLM errors propagate so the caller can trigger _rag_fallback_hint
    - Fallback itself must handle Pinecone outage and reach elicitation (P3)
    """

    @pytest.mark.asyncio
    async def test_embedding_api_down_returns_empty(self, mock_pinecone_index_factory):
        """
        Embedding inference raises ConnectionError →
        search_similar_customers must return [] without crashing.
        """
        mock_pc, _ = mock_pinecone_index_factory(
            embed_side_effect=ConnectionError("Embedding service unreachable")
        )

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=MagicMock()),
        ):
            t0      = time.perf_counter()
            results = await search_similar_customers("test query", top_k=3)
            ms      = (time.perf_counter() - t0) * 1000

        _record_perf("test_embedding_api_down_returns_empty",
                     "embed_fail_graceful", ms, "embed_api=ConnectionError")
        print(f"\n  [RESILIENCE] Embedding API down → {len(results)} results in {ms:.1f}ms")
        assert results == [], "Must return [] when embedding API fails"

    @pytest.mark.asyncio
    async def test_pinecone_query_timeout_returns_empty(self, mock_pinecone_index_factory):
        """
        Pinecone .query() raises TimeoutError →
        both search_similar_customers and search_conversation_patterns return [].
        """
        mock_pc, mock_idx = mock_pinecone_index_factory(
            query_side_effect=TimeoutError("Pinecone timed out after 5000ms")
        )

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            t0 = time.perf_counter()
            customers = await search_similar_customers("keluarga besar", top_k=3)
            patterns  = await search_conversation_patterns("percakapan test", top_k=3)
            ms = (time.perf_counter() - t0) * 1000

        _record_perf("test_pinecone_query_timeout_returns_empty",
                     "pinecone_timeout", ms, "pinecone=TimeoutError")
        print(f"\n  [RESILIENCE] Pinecone timeout → customers={customers}, patterns={patterns}")
        assert customers == [], "customers must be [] on Pinecone timeout"
        assert patterns  == [], "patterns must be [] on Pinecone timeout"

    @pytest.mark.asyncio
    async def test_llm_rate_limit_error_propagates(self, mock_async_context_manager):
        """
        OpenAI RateLimitError must propagate out of _openai_analyze so the
        ws.py error handler can catch it and invoke _rag_fallback_hint.
        """
        from openai import RateLimitError
        from backend.services.ai import _openai_analyze

        context = _make_context(texts=["Saya mau lihat-lihat dulu", "Ada promo apa?"])

        bad_response        = MagicMock()
        bad_response.status_code = 429
        rate_exc = RateLimitError(
            "You exceeded your current quota",
            response=bad_response,
            body={"error": {"type": "rate_limit_error"}},
        )

        async def raise_rate_limit(**kwargs):
            raise rate_exc

        mock_client                         = MagicMock()
        mock_client.chat.completions.create = raise_rate_limit
        mock_langfuse                        = MagicMock()
        mock_langfuse.trace.return_value    = MagicMock()
        mock_langfuse.trace.return_value.generation.return_value = MagicMock()

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            patch("backend.services.ai.settings") as mock_ai_settings,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",     mock_langfuse),
            patch("backend.services.ai.llm_requests", MagicMock()),
            patch("backend.services.ai.error_total",  MagicMock()),
            patch("backend.services.ai.llm_latency",  MagicMock()),
            patch("backend.services.ai.track_latency",
                  return_value=mock_async_context_manager),
        ):
            mock_ai_settings.openai_api_key = "test-key"

            with pytest.raises(RateLimitError):
                await _openai_analyze(context)

        _record_perf("test_llm_rate_limit_error_propagates",
                     "llm_rate_limit", 0, "exception=RateLimitError_propagated")
        print("\n  [RESILIENCE] LLM RateLimitError propagated — circuit-breaker can fire")

    @pytest.mark.asyncio
    async def test_llm_busy_flag_triggers_fallback_hint(self):
        """
        When AI-busy flag is set, _rag_fallback_hint is invoked by the caller.
        Verify it returns a valid AI_HINT with {"tahap": "RAG_FALLBACK"}.
        """
        context   = _make_context(
            texts=["Customer: Saya lagi cari SUV", "Sales: Yang seperti apa Pak?"]
        )
        rag_q = "Jalan yang biasa dilalui kondisinya gimana, mulus atau lumayan berat?"

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock,
                  return_value=[{
                      "stage": "probing",
                      "effective_next_question": rag_q,
                      "outcome": "customer_engaged",
                  }]),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            t0 = time.perf_counter()
            await _rag_fallback_hint(SESSION_ID, context)
            ms = (time.perf_counter() - t0) * 1000

        _record_perf("test_llm_busy_flag_triggers_fallback_hint",
                     "fallback_trigger", ms, "source=rag_patterns")
        print(f"\n  [RESILIENCE] Fallback triggered in {ms:.1f}ms → '{rag_q[:60]}'")

        mock_send.assert_called_once()
        _, type_arg, payload = mock_send.call_args.args
        assert type_arg  == WsMessageType.AI_HINT
        assert payload["tahap"] == "RAG_FALLBACK", (
            f"Expected tahap=RAG_FALLBACK, got '{payload.get('tahap')}'"
        )
        assert payload["suggested_question"] == rag_q
        assert rag_q in context.asked_questions

    @pytest.mark.asyncio
    async def test_full_pinecone_outage_falls_through_to_elicitation(self):
        """
        When Pinecone is completely down, rag.py's internal try/except swallows
        the exception and search_conversation_patterns returns [].
        _rag_fallback_hint then receives [] and falls through to P3 (Elicitation).

        Note: _rag_fallback_hint does NOT wrap search_conversation_patterns in
        its own try/except — it relies on rag.py's exception boundary. This test
        simulates that end-to-end behaviour by returning [].
        """
        context = _make_context(texts=["Hmm saya masih bingung pilih yang mana"])
        e_q = {
            "question": "Kantornya jauh dari rumah atau masih deket-deket?",
            "reveals":  ["mobilitas", "rutinitas"],
        }

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            # rag.py catches the internal Pinecone exception → returns [] to caller
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.routers.ws.compute_missing_dimensions",
                  return_value=["mobilitas", "keluarga"]),
            patch("backend.routers.ws.get_next_question", return_value=e_q),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        _record_perf("test_full_pinecone_outage_falls_through_to_elicitation",
                     "p3_elicitation_fallback", 0,
                     "pinecone=down, rag.py_catches → [] → P3_fires")

        mock_send.assert_called_once()
        _, _, payload = mock_send.call_args.args
        assert payload["tahap"] == "RAG_FALLBACK"
        assert payload["suggested_question"] == e_q["question"]
        print(
            f"\n  [RESILIENCE] Pinecone outage (via rag.py boundary) → "
            f"elicitation: '{e_q['question'][:60]}'"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure_hop, exc_type, exc_msg",
        [
            ("embed",    ConnectionError, "Embedding service unreachable"),
            ("pinecone", TimeoutError,    "Vector DB timed out after 5000ms"),
            ("pinecone", RuntimeError,    "Pinecone index 'mitsubishi-customers' not found"),
        ],
        ids=["embed-ConnectionError", "pinecone-TimeoutError", "pinecone-RuntimeError"],
    )
    async def test_parameterized_component_failures_return_empty(
        self, failure_hop, exc_type, exc_msg, mock_pinecone_index_factory
    ):
        """
        Parameterized — each failure mode at embedding/retrieval hop must
        produce [] gracefully (no exception escapes to the caller).
        """
        kwargs = {}
        if failure_hop == "embed":
            kwargs["embed_side_effect"] = exc_type(exc_msg)
        else:
            kwargs["query_side_effect"] = exc_type(exc_msg)

        mock_pc, mock_idx = mock_pinecone_index_factory(**kwargs)

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            result = await search_similar_customers("test", top_k=3)

        _record_perf(
            f"test_param_{failure_hop}_{exc_type.__name__}",
            failure_hop, 0,
            f"exc={exc_type.__name__}: {exc_msg[:50]}",
        )
        assert result == [], (
            f"[{failure_hop}:{exc_type.__name__}] Expected [], got {result}"
        )
        print(
            f"\n  [PARAM] hop={failure_hop}, exc={exc_type.__name__} → "
            f"graceful []"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Fallback Cascade Validation (3 priorities)
# ══════════════════════════════════════════════════════════════════════════════

class TestFallbackCascade:
    """
    Every priority path of _rag_fallback_hint is tested in isolation.
    Each test verifies: correct question chosen, asked_questions updated,
    send called once, tahap == "RAG_FALLBACK".
    """

    @pytest.mark.asyncio
    async def test_p1_topic_transition_succeeds_skips_p2_p3(self):
        """
        P1 success: detect_topic returns a topic, transition question found.
        Asserts: P2/P3 NOT called, hint_text starts with 'Arahkan ke topik'.
        """
        context = _make_context(
            texts=["Saya bareng keluarga biasanya pergi ke luar kota"]
        )
        p1_q        = "Weekend biasanya ngapain kalau bareng keluarga?"
        transitions = [{
            "from_topic":       "family",
            "to_topic":         "mobility",
            "trigger_question": p1_q,
            "outcome":          "closing",
            "naturalness":      "smooth",
        }]

        with (
            patch("backend.routers.ws.detect_topic", return_value="family"),
            patch("backend.routers.ws.search_topic_transitions",
                  new_callable=AsyncMock, return_value=transitions),
            patch("backend.routers.ws.pick_transition_question",
                  return_value=(p1_q, "mobility")),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock) as mock_p2,
            patch("backend.routers.ws.compute_missing_dimensions") as mock_p3,
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_p2.assert_not_called()
        mock_p3.assert_not_called()
        mock_send.assert_called_once()

        _, type_arg, payload = mock_send.call_args.args
        assert type_arg == WsMessageType.AI_HINT
        assert payload["tahap"]              == "RAG_FALLBACK"
        assert payload["suggested_question"] == p1_q
        assert payload["hint_text"].startswith("Arahkan ke topik"), (
            f"hint_text must start with 'Arahkan ke topik', got: '{payload['hint_text']}'"
        )
        assert "mobility" in payload["hint_text"], "hint_text must name the to_topic"
        assert p1_q in context.asked_questions
        print(f"\n  [P1] hint='{payload['hint_text']}' | q='{p1_q[:60]}'")

    @pytest.mark.asyncio
    async def test_p2_rag_patterns_succeeds_when_p1_fails(self):
        """
        P1 skip (no topic detected), P2 success: RAG patterns found.
        Asserts: P3 NOT called, correct RAG question used.
        """
        context = _make_context(texts=["Saya belum tahu mau pilih yang mana"])
        rag_q   = "Jalan yang biasa Bapak lalui kondisinya gimana?"

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock,
                  return_value=[{
                      "stage":                  "probing",
                      "effective_next_question": rag_q,
                      "outcome":                "customer_engaged",
                  }]),
            patch("backend.routers.ws.compute_missing_dimensions") as mock_compute,
            patch("backend.routers.ws.get_next_question")           as mock_get_q,
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_compute.assert_not_called()
        mock_get_q.assert_not_called()
        mock_send.assert_called_once()

        _, type_arg, payload = mock_send.call_args.args
        assert type_arg                      == WsMessageType.AI_HINT
        assert payload["tahap"]              == "RAG_FALLBACK"
        assert payload["suggested_question"] == rag_q
        assert payload["probe_topics"]       == []
        assert rag_q in context.asked_questions
        print(f"\n  [P2] q='{rag_q[:60]}'")

    @pytest.mark.asyncio
    async def test_p3_elicitation_succeeds_when_p1_p2_fail(self):
        """
        P1 skip, P2 empty list, P3 success: elicitation engine provides question.
        Asserts: hint_text contains 'dari customer', dimensions logged.
        """
        context = _make_context(texts=["Hmm saya masih nimbang-nimbang"])
        e_q = {
            "question": "Kantornya jauh dari rumah atau masih deket-deket?",
            "reveals":  ["mobilitas", "rutinitas"],
        }

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.routers.ws.compute_missing_dimensions",
                  return_value=["mobilitas", "keluarga"]),
            patch("backend.routers.ws.get_next_question", return_value=e_q),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_send.assert_called_once()
        _, type_arg, payload = mock_send.call_args.args

        assert type_arg                      == WsMessageType.AI_HINT
        assert payload["tahap"]              == "RAG_FALLBACK"
        assert payload["suggested_question"] == e_q["question"]
        assert "mobilitas" in payload["hint_text"], (
            f"hint_text must name revealed dimensions, got: '{payload['hint_text']}'"
        )
        assert "dari customer" in payload["hint_text"], (
            f"hint_text must end 'dari customer.', got: '{payload['hint_text']}'"
        )
        assert e_q["question"] in context.asked_questions
        print(f"\n  [P3] hint='{payload['hint_text']}' | q='{e_q['question'][:60]}'")

    @pytest.mark.asyncio
    async def test_all_priorities_fail_early_return_no_send(self):
        """
        All three priorities exhausted: no message sent, asked_questions unchanged.
        """
        initial = ["Sudah lama pakai kendaraan sekarang?", "Kantornya jauh dari rumah?"]
        context = _make_context(
            texts=["iya saya mau lihat-lihat dulu"], asked=initial.copy()
        )

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.routers.ws.compute_missing_dimensions", return_value=[]),
            patch("backend.routers.ws.get_next_question", return_value=None),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_send.assert_not_called()
        assert context.asked_questions == initial, (
            "asked_questions must not change when all paths exhausted"
        )
        print("\n  [ALL-FAIL] No hint sent — correct early return")

    @pytest.mark.asyncio
    async def test_fallback_deprioritises_deflected_questions(self):
        """
        P2: outcome='question_deflected' patterns are deprioritized.
        System must pick the 'customer_engaged' pattern first.
        """
        context = _make_context(texts=["Oke saya lihat-lihat dulu ya"])
        good_q  = "Weekend biasanya ngapain Pak?"
        bad_q   = "Mau beli kapan Pak?"

        patterns = [
            {"stage": "closing",  "effective_next_question": bad_q,
             "outcome": "question_deflected"},
            {"stage": "probing",  "effective_next_question": good_q,
             "outcome": "customer_engaged"},
        ]

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=patterns),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_send.assert_called_once()
        _, _, payload = mock_send.call_args.args
        assert payload["suggested_question"] == good_q, (
            f"Must choose 'customer_engaged' question, got '{payload['suggested_question']}'"
        )
        print(f"\n  [DEFLECT] Correctly chose '{good_q[:60]}' over deflected question")

    @pytest.mark.asyncio
    async def test_fallback_skips_already_asked_rag_question(self):
        """
        P2: effective_next_question already in context.asked_questions →
        fallback must fall through to P3.
        """
        already = "Weekend biasanya ngapain Pak?"
        context = _make_context(texts=["Saya tertarik Xpander"], asked=[already])

        e_q = {
            "question": "Kantornya jauh dari rumah atau masih deket-deket?",
            "reveals":  ["mobilitas"],
        }

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock,
                  return_value=[{
                      "stage":                  "probing",
                      "effective_next_question": already,
                      "outcome":                "customer_engaged",
                  }]),
            patch("backend.routers.ws.compute_missing_dimensions",
                  return_value=["mobilitas"]),
            patch("backend.routers.ws.get_next_question", return_value=e_q),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        mock_send.assert_called_once()
        _, _, payload = mock_send.call_args.args
        assert payload["suggested_question"] != already, (
            "Must NOT repeat a question already in asked_questions"
        )
        print(
            f"\n  [SKIP-ASKED] Correctly skipped '{already[:40]}...' → "
            f"'{payload['suggested_question'][:60]}'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Output Format Completeness
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputFormatValidation:
    """
    Verify every field in every outbound payload matches the schema contract.
    """

    @pytest.mark.asyncio
    async def test_fallback_payload_has_all_required_keys(self):
        """
        RAG_FALLBACK payload must contain:
          hint_text, suggested_question, probe_topics (list),
          detected_needs (list), tahap == "RAG_FALLBACK"
        """
        context = _make_context(
            texts=["Saya tertarik Xpander untuk keluarga"],
            needs=["keluarga besar"],
        )
        rag_q = "Biasanya kalau pergi weekend, sendirian atau ada yang ikut?"

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock,
                  return_value=[{
                      "stage":                  "rapport",
                      "effective_next_question": rag_q,
                      "outcome":                "customer_engaged",
                  }]),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock) as mock_send,
        ):
            await _rag_fallback_hint(SESSION_ID, context)

        _, _, payload = mock_send.call_args.args
        required = {"hint_text", "suggested_question", "probe_topics",
                    "detected_needs", "tahap"}
        missing  = required - payload.keys()
        assert not missing, f"Payload missing keys: {missing}"
        assert payload["tahap"]                    == "RAG_FALLBACK"
        assert isinstance(payload["probe_topics"],  list)
        assert isinstance(payload["detected_needs"], list)
        assert isinstance(payload["suggested_question"], str)
        assert len(payload["suggested_question"]) > 0

    def test_format_rag_context_structure(self):
        """format_rag_context must include customer profile and BELI outcome."""
        result = format_rag_context(MOCK_CUSTOMER_MATCHES)
        assert "Pelanggan serupa" in result
        assert "Wiraswasta"       in result
        assert "BELI"             in result

    def test_format_pattern_context_structure(self):
        """format_pattern_context must include stage tag and question text."""
        result = format_pattern_context(MOCK_PATTERN_MATCHES)
        assert "probing"  in result
        assert "Biasanya" in result or "weekend" in result.lower()

    def test_format_rag_context_empty_input(self):
        """format_rag_context([]) must return empty string, not crash."""
        assert format_rag_context([]) == ""

    def test_format_pattern_context_empty_input(self):
        """format_pattern_context([]) must return empty string, not crash."""
        assert format_pattern_context([]) == ""

    def test_compute_missing_dimensions_all_missing_on_greeting(self):
        """A pure greeting has no signals → all 4 dimensions missing."""
        missing = compute_missing_dimensions("Halo selamat datang Pak")
        assert len(missing) == 4
        assert set(missing) == {"mobilitas", "keluarga", "finansial", "urgency"}

    def test_compute_missing_dimensions_partial_coverage(self):
        """Conversation with explicit family + mobility signals marks them found."""
        text    = "Saya sering luar kota bareng anak-anak, mudik ke Jawa tiap bulan"
        missing = compute_missing_dimensions(text)
        assert "mobilitas" not in missing, "mobilitas should be found"
        assert "keluarga"  not in missing, "keluarga should be found"

    def test_compute_missing_dimensions_financial_signal(self):
        """Mentioning a specific budget figure should mark finansial as found."""
        text    = "Kira-kira budget saya 300 juta"
        missing = compute_missing_dimensions(text)
        assert "finansial" not in missing

    def test_get_next_question_skips_high_overlap_asked(self):
        """
        get_next_question must not return a question with >40% word overlap
        against the already-asked list.
        """
        asked   = ["Kantornya jauh dari rumah atau masih deket-deket?"]
        missing = ["mobilitas"]
        q       = get_next_question(missing, asked)
        if q:
            assert q["question"] != asked[0], (
                "Must skip exact match or high-overlap question"
            )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Stress / Concurrency Load
# ══════════════════════════════════════════════════════════════════════════════

class TestStressLoad:
    """
    Fire bursts of concurrent requests to validate session isolation and
    sustained latency under load.
    """

    @pytest.mark.asyncio
    async def test_10_concurrent_fallback_no_cross_contamination(self):
        """
        10 concurrent _rag_fallback_hint calls on distinct session IDs.
        Each session's asked_questions must have exactly 1 entry with no
        cross-contamination between sessions.
        """
        N        = 10
        contexts = [
            _make_context(texts=[f"Customer {i}: Saya tertarik mobil keluarga"], asked=[])
            for i in range(N)
        ]
        sids = [f"stress-session-{i}" for i in range(N)]
        rag_q = "Weekend biasanya ngapain Pak?"

        with (
            patch("backend.routers.ws.detect_topic", return_value=None),
            patch("backend.routers.ws.search_conversation_patterns",
                  new_callable=AsyncMock,
                  return_value=[{
                      "stage":                  "rapport",
                      "effective_next_question": rag_q,
                      "outcome":                "customer_engaged",
                  }]),
            patch("backend.routers.ws.session_manager.send",
                  new_callable=AsyncMock),
        ):
            t0 = time.perf_counter()
            await asyncio.gather(*[
                _rag_fallback_hint(sids[i], contexts[i]) for i in range(N)
            ])
            ms = (time.perf_counter() - t0) * 1000

        _record_perf("test_10_concurrent_fallback_no_cross_contamination",
                     f"stress_{N}_concurrent", ms, f"N={N}")
        print(f"\n  [STRESS] {N} concurrent fallbacks in {ms:.1f}ms")

        for i, ctx in enumerate(contexts):
            assert rag_q in ctx.asked_questions, (
                f"Session {i}: question not recorded"
            )
            assert len(ctx.asked_questions) == 1, (
                f"Session {i}: expected 1 question, got {len(ctx.asked_questions)}"
            )

    @pytest.mark.asyncio
    async def test_20_sequential_pattern_queries_avg_latency(
        self, mock_pinecone_index_factory
    ):
        """
        20 sequential search_conversation_patterns calls.
        Average latency must stay under EMBED_RETRIEVAL_WARNING_MS.
        """
        mock_pc, mock_idx = mock_pinecone_index_factory(
            embed_delay_s=0.010,
            query_delay_s=0.015,
            matches=MOCK_PATTERN_MATCHES,
        )
        N         = 20
        latencies = []

        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_pc),
            patch("backend.services.rag._pinecone_index", return_value=mock_idx),
        ):
            for i in range(N):
                t0 = time.perf_counter()
                await search_conversation_patterns(
                    f"Customer: percakapan ke-{i} tentang keluarga dan mobil", top_k=3
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(latencies) / N
        max_ms = max(latencies)
        p95_ms = sorted(latencies)[int(N * 0.95)]

        _record_perf("test_20_sequential_pattern_queries_avg_latency",
                     f"high_freq_{N}x", avg_ms,
                     f"max_ms={max_ms:.1f}, p95_ms={p95_ms:.1f}")
        print(
            f"\n  [STRESS] {N}× pattern retrieval: "
            f"avg={avg_ms:.1f}ms  max={max_ms:.1f}ms  p95={p95_ms:.1f}ms"
        )

        assert avg_ms < EMBED_RETRIEVAL_WARNING_MS, (
            f"Avg latency {avg_ms:.1f}ms > {EMBED_RETRIEVAL_WARNING_MS}ms threshold"
        )

    @pytest.mark.asyncio
    async def test_burst_mixed_success_and_failure(self, mock_pinecone_index_factory):
        """
        Two concurrent bursts in sequence (each burst shares one patch context to
        avoid concurrent patching races on the same module attribute):
          Burst A — 3 calls with healthy Pinecone   → all return non-empty results
          Burst B — 3 calls with embed exception    → all return []
        Total wall time must be < 1s.
        """
        N = 3

        mock_ok,   idx_ok   = mock_pinecone_index_factory(
            embed_delay_s=0.020, query_delay_s=0.020, matches=MOCK_CUSTOMER_MATCHES
        )
        mock_fail, idx_fail = mock_pinecone_index_factory(
            embed_side_effect=ConnectionError("embed down")
        )

        t0 = time.perf_counter()

        # Burst A — successful concurrent calls, all sharing the same patch context
        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_ok),
            patch("backend.services.rag._pinecone_index",  return_value=idx_ok),
        ):
            ok_results = await asyncio.gather(*[
                search_similar_customers(f"query-ok-{i}", top_k=2) for i in range(N)
            ])

        # Burst B — failing concurrent calls, all sharing the same patch context
        with (
            patch("backend.services.rag._pinecone_client", return_value=mock_fail),
            patch("backend.services.rag._pinecone_index",  return_value=idx_fail),
        ):
            fail_results = await asyncio.gather(*[
                search_similar_customers(f"query-fail-{i}", top_k=2) for i in range(N)
            ])

        ms = (time.perf_counter() - t0) * 1000

        _record_perf("test_burst_mixed_success_and_failure",
                     "burst_3ok_3fail", ms, f"total_calls={N*2}")
        print(f"\n  [STRESS] {N*2}-call mixed burst (A:ok B:fail) in {ms:.1f}ms")

        for i, r in enumerate(ok_results):
            assert len(r) > 0, f"ok_call {i} must have results, got {r}"
        for i, r in enumerate(fail_results):
            assert r == [],    f"fail_call {i} must return [], got {r}"

        assert ms < 1000, f"Burst took {ms:.1f}ms — over 1s ceiling"
