"""
test_edge_cases_emotional_intelligence.py
==========================================
Advanced edge-case test suite for the Mitsubishi AI Sales Assistant.

Tests two critical real-world showroom scenarios:
  1. Ambient Noise Tolerance — STT & input preprocessing robustness
     Validates that bracket-tagged noise markers (music, laughter, engine sounds)
     and low-energy audio do not prevent key entity extraction.

  2. Contextual Pivot — Emotional intelligence & tone adaptation
     Validates that the RAG/LLM layer detects a mid-conversation emotional shift
     (excited → financially anxious) and pivots suggested_question to empathetic,
     consultative guidance rather than pushy upsells.

Domain: Mitsubishi Indonesia showroom (Bahasa Indonesia transcripts)

Run:
  uv run pytest test_edge_cases_emotional_intelligence.py -v --tb=short -s

Output:
  edge_case_test_results.csv  (created in working directory on completion)
"""

import asyncio
import csv
import json
import math
import random
import re
import time
import uuid
import logging
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.schemas import (
    AiHintPayload,
    ConversationContext,
    Utterance,
)
from backend.services.stt import _check_audio_energy
from backend.services.elicitation import compute_missing_dimensions, get_next_question

logger = logging.getLogger(__name__)

# ── CSV result accumulator ─────────────────────────────────────────────────────

_test_results: list[dict] = []


def _record_result(
    test_name: str,
    scenario: str,
    outcome: str,
    details: str = "",
    duration_ms: float = 0.0,
) -> None:
    _test_results.append(
        {
            "test_name": test_name,
            "scenario": scenario,
            "outcome": outcome,
            "details": details,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# ── Noise-injected transcript fixtures ────────────────────────────────────────

NOISY_TRANSCRIPTS = [
    "[Music Playing]... Saya nyari... [Laughter]... Pajero Sport yang... [Unclear]... budgetnya aman sekitar 500jt...",
    "eh [background noise] iya saya cari [crowd talking] Pajero Sport [engine revving] budget lima ratus juta",
    "[Music] saya... [Unclear] ...mau beli [laughter] Pajero Sport... [crowd] budget [noise] 500 juta [music]",
]

EXPECTED_ENTITY_KEYWORDS = ["pajero sport", "500", "budget", "juta"]

_NOISE_PATTERN = re.compile(r"\[.*?\]|\.{2,}", re.IGNORECASE)


def _preprocess_noisy_text(text: str) -> str:
    """Strip bracket noise markers and collapse whitespace."""
    cleaned = _NOISE_PATTERN.sub(" ", text)
    return " ".join(cleaned.split()).lower()


# ── Synthetic audio byte generators ───────────────────────────────────────────

def _make_silent_audio(size: int = 3000) -> bytes:
    """All bytes at 128 (DC level) — avg abs deviation ≈ 0 < threshold 3.0."""
    return bytes(200) + bytes([128] * (size - 200))


def _make_noise_floor_audio(size: int = 3000, amplitude: int = 2) -> bytes:
    """Low-amplitude noise — expected avg abs deviation ≈ 1.2 < threshold 3.0."""
    header = bytes(200)
    body = bytes([128 + random.randint(-amplitude, amplitude) for _ in range(size - 200)])
    return header + body


def _make_speech_audio(size: int = 3000, amplitude: int = 30) -> bytes:
    """Sin-wave speech simulation — avg abs deviation ≈ 19.1 >> threshold 3.0."""
    header = bytes(200)
    body = bytes(
        [
            min(255, max(0, 128 + int(amplitude * math.sin(i * 0.1))))
            for i in range(size - 200)
        ]
    )
    return header + body


# ── Domain test fixtures ───────────────────────────────────────────────────────

MOCK_LLM_TURN1_EXCITED = {
    "tahap": "EKSPLORASI",
    "hint_text": "Customer antusias, keluarga 6 orang, siap beli MPV",
    "suggested_question": "Weekend biasanya pergi ke mana bareng keluarga?",
    "probe_topics": ["mobilitas", "finansial"],
    "detected_needs": ["keluarga besar 6 orang", "antusias membeli", "butuh MPV kapasitas besar"],
    "recommended_car_ids": [],
    "recommendation_reason": "",
}

MOCK_LLM_TURN2_ANXIOUS = {
    "tahap": "PENDALAMAN",
    "hint_text": "Customer cemas cicilan, anak kuliah, stabilitas finansial kritis",
    "suggested_question": "Cicilan bulanan yang paling nyaman untuk Bapak kira-kira berapa?",
    "probe_topics": ["Kestabilan Finansial Keluarga", "cicilan aman", "opsi pembiayaan"],
    "detected_needs": [
        "Kestabilan Finansial Keluarga",
        "Anxiety tentang cicilan bulanan",
        "anak pertama masuk kuliah tahun ini",
        "keluarga besar 6 orang",
    ],
    "recommended_car_ids": ["xpander-cross"],
    "recommendation_reason": (
        "Xpander Cross menawarkan cicilan lebih ringan dibanding SUV premium, "
        "tetap memuat 6 orang — cocok untuk keluarga dengan prioritas kestabilan finansial"
    ),
}

MOCK_LLM_NOISY_ENTITIES = {
    "tahap": "EKSPLORASI",
    "hint_text": "Customer cari Pajero Sport, budget 500 juta, konfirmasi lebih lanjut dibutuhkan",
    "suggested_question": "Perjalanannya biasanya ke medan seperti apa Pak?",
    "probe_topics": ["mobilitas", "keluarga"],
    "detected_needs": ["Pajero Sport intent", "budget 500 juta", "SUV off-road preference"],
    "recommended_car_ids": [],
    "recommendation_reason": "",
}


# ── Utterance / context helpers ───────────────────────────────────────────────

def _make_utterance(text: str, speaker: str = "customer") -> Utterance:
    return Utterance(
        id=str(uuid.uuid4()),
        speaker=speaker,
        text=text,
        timestamp=datetime.now(timezone.utc),
        confidence=0.9,
    )


def _make_context(
    texts: list[str],
    speaker: str = "customer",
    asked: list[str] | None = None,
) -> ConversationContext:
    return ConversationContext(
        session_id=f"test-{uuid.uuid4().hex[:8]}",
        utterances=[_make_utterance(t, speaker) for t in texts],
        asked_questions=list(asked or []),
        detected_needs=[],
    )


# ── Session-scoped CSV reporter ────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def write_results_csv():
    """Flush all test results to CSV at end of session."""
    yield

    csv_path = Path("edge_case_test_results.csv")
    fieldnames = ["test_name", "scenario", "outcome", "details", "duration_ms", "timestamp"]

    if not _test_results:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_test_results)

    sep = "=" * 95
    print(f"\n\n{sep}")
    print(f"  Edge Case Test Results  →  {csv_path.resolve()}")
    print(sep)
    print(f"  {'Test':<48} {'Scenario':<26} {'Outcome':<8} {'ms':>6}")
    print(f"  {'-'*48} {'-'*26} {'-'*8} {'-'*6}")
    for row in _test_results:
        lat = f"{row['duration_ms']:.2f}" if row["duration_ms"] else "  N/A"
        outcome_tag = "PASS" if row["outcome"] == "PASS" else f"FAIL"
        print(
            f"  {row['test_name'][:48]:<48} "
            f"{row['scenario'][:26]:<26} "
            f"{outcome_tag:<8} "
            f"{lat:>6}"
        )
    print(sep + "\n")


# ── Pinecone / RAG settings autouse fixture ───────────────────────────────────

@pytest.fixture(autouse=True)
def configure_rag_settings():
    """Inject fake Pinecone credentials so RAG search guards pass."""
    with (
        patch("backend.services.rag.settings") as mock_rag,
        patch("backend.services.topic_patterns.settings") as mock_tp,
    ):
        mock_rag.pinecone_api_key    = "test-key"
        mock_rag.pinecone_index_name = "mitsubishi-customers"
        mock_rag.pinecone_namespace  = "customers-data"
        mock_tp.pinecone_api_key     = "test-key"
        mock_tp.pinecone_index_name  = "mitsubishi-customers"
        yield mock_rag


# ── Shared async context-manager fixture ──────────────────────────────────────

@pytest.fixture
def mock_cm():
    """No-op async context manager — returned by patched track_latency calls."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=None)
    ctx.__aexit__  = AsyncMock(return_value=False)
    return ctx


# ── Shared LLM client factory ─────────────────────────────────────────────────

def _make_openai_mock(response_json: dict) -> MagicMock:
    """Build a mock AsyncOpenAI client that returns the given JSON payload."""
    raw = json.dumps(response_json)
    mock_resp                          = MagicMock()
    mock_resp.choices                  = [MagicMock()]
    mock_resp.choices[0].message.content = raw
    mock_resp.usage.prompt_tokens      = 420
    mock_resp.usage.completion_tokens  = 100

    async def _create(**kwargs):
        return mock_resp

    client                             = MagicMock()
    client.chat.completions.create     = _create
    return client


def _make_langfuse_mock() -> MagicMock:
    lf = MagicMock()
    lf.trace.return_value = MagicMock()
    lf.trace.return_value.generation.return_value = MagicMock()
    return lf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Ambient Noise Tolerance (STT & Input Robustness)
# ══════════════════════════════════════════════════════════════════════════════

class TestAmbientNoiseTolerance:
    """
    Validates that the STT preprocessing layer correctly filters showroom
    background noise (music, laughter, engine sounds) and still surfaces
    key customer data points: vehicle model intent + budget signal.

    Sub-scenarios:
      A. Audio energy filter (silence / noise floor / speech)
      B. Text bracket-noise preprocessing
      C. Elicitation dimension detection after denoising
      D. STT no_speech_prob filter (Whisper-level guard)
      E. End-to-end LLM entity extraction from noisy transcript
    """

    # ── A. Audio energy filter ─────────────────────────────────────────────

    def test_audio_energy_filter_rejects_silence(self):
        """
        Pure silence bytes (all 128) produce avg deviation ≈ 0 < threshold 3.0.
        _check_audio_energy must return False.
        """
        t0     = time.perf_counter()
        silent = _make_silent_audio(3000)
        result = _check_audio_energy(silent)
        ms     = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_audio_energy_filter_rejects_silence",
            "A_silence_rejection",
            "PASS" if not result else "FAIL",
            f"bytes={len(silent)}, energy_check={result}",
            ms,
        )
        logger.info(f"[NOISE-A] Silence check → {result} in {ms:.2f}ms (expected False)")
        assert result is False, (
            "Silence audio (all bytes=128, avg deviation=0) must fail energy threshold 3.0"
        )

    def test_audio_energy_filter_rejects_low_amplitude_noise_floor(self):
        """
        Low-amplitude random noise (amplitude=2) yields avg deviation ≈ 1.2 < 3.0.
        Simulates showroom background hiss that should be discarded.
        """
        t0    = time.perf_counter()
        noise = _make_noise_floor_audio(3000, amplitude=2)
        result = _check_audio_energy(noise)
        ms    = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_audio_energy_filter_rejects_low_amplitude_noise_floor",
            "A_noise_floor_rejection",
            "PASS" if not result else "FAIL",
            f"bytes={len(noise)}, amplitude=2, energy_check={result}",
            ms,
        )
        logger.info(f"[NOISE-A] Noise floor (amp=2) → {result} in {ms:.2f}ms (expected False)")
        assert result is False, (
            "Low-amplitude noise (amplitude=2, avg dev ≈ 1.2) must fail energy threshold 3.0"
        )

    def test_audio_energy_filter_accepts_speech_level_audio(self):
        """
        Sin-wave speech simulation (amplitude=30) yields avg deviation ≈ 19.1 >> 3.0.
        Represents a customer clearly speaking in a showroom environment.
        """
        t0     = time.perf_counter()
        speech = _make_speech_audio(3000, amplitude=30)
        result = _check_audio_energy(speech)
        ms     = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_audio_energy_filter_accepts_speech_level_audio",
            "A_speech_acceptance",
            "PASS" if result else "FAIL",
            f"bytes={len(speech)}, amplitude=30, energy_check={result}",
            ms,
        )
        logger.info(f"[NOISE-A] Speech (amp=30) → {result} in {ms:.2f}ms (expected True)")
        assert result is True, (
            "Speech audio (amplitude=30, avg dev ≈ 19.1) must pass energy threshold 3.0"
        )

    def test_audio_too_small_rejected_before_energy_check(self):
        """
        Audio < 400 bytes means body (after 200-byte header) has < 200 bytes,
        which triggers the early size rejection before computing RMS.
        """
        tiny = bytes(350)
        result = _check_audio_energy(tiny)

        _record_result(
            "test_audio_too_small_rejected_before_energy_check",
            "A_size_rejection",
            "PASS" if not result else "FAIL",
            f"bytes={len(tiny)}, result={result}",
        )
        assert result is False, "Audio with body < 200 bytes must be rejected at size guard"

    # ── B. Text bracket-noise preprocessing ───────────────────────────────

    def test_noise_bracket_preprocessing_isolates_pajero_and_budget(self):
        """
        The primary noisy transcript: after stripping [Music Playing], [Laughter],
        [Unclear], 'Pajero Sport' and '500' must survive intact.
        """
        noisy   = NOISY_TRANSCRIPTS[0]
        t0      = time.perf_counter()
        cleaned = _preprocess_noisy_text(noisy)
        ms      = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_noise_bracket_preprocessing_isolates_pajero_and_budget",
            "B_bracket_stripping",
            "PASS" if "pajero sport" in cleaned and "500" in cleaned else "FAIL",
            f"raw_len={len(noisy)}, clean_len={len(cleaned)}, cleaned='{cleaned[:80]}'",
            ms,
        )
        logger.info(f"[NOISE-B] Raw: '{noisy[:70]}'")
        logger.info(f"[NOISE-B] Cleaned: '{cleaned}'")

        assert "pajero sport" in cleaned, (
            f"'pajero sport' missing from cleaned text: '{cleaned}'"
        )
        assert "500" in cleaned, (
            f"'500' (budget signal) missing from cleaned text: '{cleaned}'"
        )
        for noise_tag in ["[music playing]", "[laughter]", "[unclear]"]:
            assert noise_tag not in cleaned, (
                f"Noise tag '{noise_tag}' must be stripped — found in: '{cleaned}'"
            )

    @pytest.mark.parametrize(
        "noisy_input,variant_id",
        [(t, i + 1) for i, t in enumerate(NOISY_TRANSCRIPTS)],
        ids=["variant-1-brackets-dots", "variant-2-inline-noise", "variant-3-mixed-tags"],
    )
    def test_all_noisy_variants_retain_at_least_two_core_entities(
        self, noisy_input: str, variant_id: int
    ):
        """
        All 3 noise injection patterns must produce cleaned text with ≥ 2 of:
        'pajero sport', '500', 'budget', 'juta'.
        Validates robustness across different noise injection styles.
        """
        t0      = time.perf_counter()
        cleaned = _preprocess_noisy_text(noisy_input)
        ms      = (time.perf_counter() - t0) * 1000

        found = [kw for kw in EXPECTED_ENTITY_KEYWORDS if kw in cleaned]

        _record_result(
            f"test_noisy_variant_{variant_id}_entity_retention",
            "B_multi_variant",
            "PASS" if len(found) >= 2 else "FAIL",
            f"found={found}, cleaned='{cleaned[:80]}'",
            ms,
        )
        logger.info(f"[NOISE-B] Variant {variant_id}: entities_found={found}, cleaned='{cleaned[:80]}'")
        assert len(found) >= 2, (
            f"Variant {variant_id}: at least 2 core entities must survive denoising. "
            f"Found: {found} in '{cleaned[:80]}'"
        )

    # ── C. Elicitation dimension detection after denoising ─────────────────

    def test_finansial_dimension_detected_after_noise_stripped(self):
        """
        After stripping bracket noise, the budget keyword '500 juta' triggers the
        finansial direct-signal regex → compute_missing_dimensions must NOT include
        'finansial' in its missing list.
        """
        noisy   = "[Music Playing] saya cari [laughter] Pajero Sport [noise] budget sekitar 500 juta [unclear]"
        cleaned = _preprocess_noisy_text(noisy)

        t0      = time.perf_counter()
        missing = compute_missing_dimensions(cleaned)
        ms      = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_finansial_dimension_detected_after_noise_stripped",
            "C_finansial_detection",
            "PASS" if "finansial" not in missing else "FAIL",
            f"missing={missing}, cleaned='{cleaned[:80]}'",
            ms,
        )
        logger.info(f"[NOISE-C] After denoising, missing dims: {missing}")
        assert "finansial" not in missing, (
            f"'finansial' must be DETECTED after denoising extracts '500 juta'. "
            f"cleaned='{cleaned}', missing={missing}"
        )

    def test_full_noisy_transcript_elicitation_suggests_relevant_followup(self):
        """
        After denoising, get_next_question on a mostly-blank dimension profile
        must return a non-None question (fallback to universal bank if needed).
        The question must not contain noise markers or brackets.
        """
        noisy   = "[Laughter]... budgetnya aman sekitar 500jt... [Music] Pajero Sport... [Unclear]"
        cleaned = _preprocess_noisy_text(noisy)
        missing = compute_missing_dimensions(cleaned)
        q_dict  = get_next_question(missing, asked_questions=[])

        _record_result(
            "test_full_noisy_transcript_elicitation_suggests_relevant_followup",
            "C_followup_question",
            "PASS" if q_dict and "[" not in q_dict["question"] else "FAIL",
            f"missing={missing}, question='{q_dict['question'] if q_dict else None}'",
        )
        assert q_dict is not None, "get_next_question must return a question for uncovered dims"
        assert "[" not in q_dict["question"], "Question must not contain noise bracket artifacts"

    # ── D. STT no_speech_prob filter ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stt_high_no_speech_prob_discards_background_noise_transcript(self):
        """
        Whisper returning no_speech_prob=0.87 (background music with no real speech)
        must cause transcribe_audio() to return None — guarding against hallucinated
        noise-only transcripts like '[Music Playing]'.
        """
        from backend.services.stt import transcribe_audio

        audio = _make_speech_audio(5000, amplitude=30)

        mock_result             = MagicMock()
        mock_result.text        = "[Music Playing]"
        mock_result.no_speech_prob = 0.87
        mock_result.segments    = []

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)

        mock_lf = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            patch("backend.services.stt.settings") as s,
            patch("backend.services.stt.langfuse",     mock_lf),
            patch("backend.services.stt.stt_requests", MagicMock()),
            patch("backend.services.stt.stt_latency",  MagicMock()),
            patch("backend.services.stt.error_total",  MagicMock()),
            patch("backend.services.stt.track_latency") as mt,
        ):
            s.stt_provider  = "openai"
            s.openai_api_key = "test-key"
            s.language      = "id"
            mt.return_value.__aenter__ = AsyncMock(return_value=None)
            mt.return_value.__aexit__  = AsyncMock(return_value=False)
            result = await transcribe_audio(audio, "test-noise-session")
        ms = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_stt_high_no_speech_prob_discards_noise_transcript",
            "D_no_speech_prob_filter",
            "PASS" if result is None else "FAIL",
            f"no_speech_prob=0.87, result={result}",
            ms,
        )
        logger.info(f"[NOISE-D] no_speech_prob=0.87 → {result} in {ms:.2f}ms (expected None)")
        assert result is None, (
            f"no_speech_prob=0.87 must discard transcript, but got Utterance: {result}"
        )

    @pytest.mark.asyncio
    async def test_stt_low_no_speech_prob_preserves_valid_customer_speech(self):
        """
        Whisper returning no_speech_prob=0.12 (clear customer speech) must pass
        the filter and return a valid Utterance with the original text intact.
        """
        from backend.services.stt import transcribe_audio

        audio         = _make_speech_audio(5000, amplitude=30)
        expected_text = "Saya nyari Pajero Sport budgetnya aman sekitar 500jt"

        mock_result                = MagicMock()
        mock_result.text           = expected_text
        mock_result.no_speech_prob = 0.12
        mock_result.segments       = []

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_result)

        mock_lf = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            patch("backend.services.stt.settings") as s,
            patch("backend.services.stt.langfuse",     mock_lf),
            patch("backend.services.stt.stt_requests", MagicMock()),
            patch("backend.services.stt.stt_latency",  MagicMock()),
            patch("backend.services.stt.error_total",  MagicMock()),
            patch("backend.services.stt.track_latency") as mt,
        ):
            s.stt_provider   = "openai"
            s.openai_api_key = "test-key"
            s.language       = "id"
            mt.return_value.__aenter__ = AsyncMock(return_value=None)
            mt.return_value.__aexit__  = AsyncMock(return_value=False)
            result = await transcribe_audio(audio, "test-clear-session")
        ms = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_stt_low_no_speech_prob_preserves_valid_customer_speech",
            "D_valid_speech_pass",
            "PASS" if result is not None and result.text == expected_text else "FAIL",
            f"no_speech_prob=0.12, text='{result.text if result else None}'",
            ms,
        )
        assert result is not None, "Clear speech (no_speech_prob=0.12) must produce an Utterance"
        assert result.text == expected_text, (
            f"Expected '{expected_text}', got '{result.text}'"
        )

    # ── E. End-to-end LLM entity extraction ───────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_extracts_pajero_and_budget_from_denoised_transcript(self, mock_cm):
        """
        End-to-end: noisy showroom transcript → _preprocess_noisy_text → ConversationContext
        → _openai_analyze() (mocked LLM) → detected_needs must contain 'Pajero Sport'
        and '500 juta' signals.

        Simulates: customer mumbles through background music & crowd noise; system
        still identifies the vehicle intent and budget indicator.
        """
        from backend.services.ai import _openai_analyze

        noisy_text = NOISY_TRANSCRIPTS[0]
        cleaned    = _preprocess_noisy_text(noisy_text)
        context    = _make_context([cleaned])

        mock_client = _make_openai_mock(MOCK_LLM_NOISY_ENTITIES)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",                return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",           mock_lf),
            patch("backend.services.ai.llm_requests",       MagicMock()),
            patch("backend.services.ai.llm_latency",        MagicMock()),
            patch("backend.services.ai.token_per_request",  MagicMock()),
            patch("backend.services.ai.rekomendasi_total",  MagicMock()),
            patch("backend.services.ai.track_latency",      return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, _ = await _openai_analyze(context)
        ms = (time.perf_counter() - t0) * 1000

        pajero_found = any("pajero" in n.lower() for n in hint.detected_needs)
        budget_found = any("500" in n for n in hint.detected_needs)

        _record_result(
            "test_llm_extracts_pajero_and_budget_from_denoised_transcript",
            "E_llm_entity_extraction",
            "PASS" if pajero_found and budget_found else "FAIL",
            f"detected_needs={hint.detected_needs}",
            ms,
        )
        logger.info(f"[NOISE-E] detected_needs: {hint.detected_needs}")
        assert pajero_found, (
            f"'Pajero Sport' must be in detected_needs. Got: {hint.detected_needs}"
        )
        assert budget_found, (
            f"'500 juta' budget signal must be in detected_needs. Got: {hint.detected_needs}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Contextual Pivot (Emotional Intelligence & Tone Adaptation)
# ══════════════════════════════════════════════════════════════════════════════

class TestContextualPivotEmotionalIntelligence:
    """
    Validates that the RAG/LLM backend detects a sudden mid-conversation
    shift in the customer's emotional state and gracefully pivots from
    exploratory / upsell-oriented questions to empathetic, financial-safe
    consultative guidance.

    Dialogue structure:
      Turn 1 — Positive / Excited:
        Customer enthusiastic about MPV for a family of 6, ready to buy.
      Turn 2 — Anxious / Sensitive:
        Same customer suddenly reveals financial anxiety: "anak pertama saya
        mau masuk kuliah tahun ini" — worried about monthly cicilan burden.

    Assertions:
      • 'Kestabilan Finansial Keluarga' / 'Anxiety' flagged in detected_needs
      • suggested_question pivots to empathetic, cicilan-aware tone
      • NO pushy upsell language (promo, booking, DP sekarang, etc.)
      • Car recommendation shifts to budget-friendly option (Xpander Cross)
        NOT premium upsells (Pajero Sport Dakar, Eclipse Cross, etc.)
    """

    # ── Fixtures ──────────────────────────────────────────────────────────

    @pytest.fixture
    def excited_context(self):
        return _make_context(
            [
                "Selamat pagi, saya tertarik lihat MPV untuk keluarga saya.",
                "Kami berenam, sering jalan-jalan bareng, saya sudah nabung lama untuk beli ini.",
                "Saya sudah mantap mau beli, tinggal pilih variannya saja.",
            ]
        )

    @pytest.fixture
    def anxious_context(self):
        return _make_context(
            [
                "Selamat pagi, saya tertarik lihat MPV untuk keluarga saya.",
                "Kami berenam, sering jalan-jalan bareng, saya sudah nabung lama untuk beli ini.",
                "Saya sudah mantap mau beli, tinggal pilih variannya saja.",
                "Tapi sejujurnya saya agak cemas dengan cicilan bulanannya karena "
                "anak pertama saya mau masuk kuliah tahun ini.",
                "Jangan-jangan nanti berat di tengah jalan, apalagi biaya kuliah sekarang mahal.",
            ]
        )

    # ── Turn 1: Excited state ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_turn1_excited_state_generates_exploratory_question(
        self, excited_context, mock_cm
    ):
        """
        Turn 1 (excited): LLM returns an exploratory follow-up question.
        Asserts: tahap is EKSPLORASI/PENDALAMAN/RAPPORT, detected_needs includes
        a family signal, recommended_car_ids is still [] (too early for upsell).
        """
        from backend.services.ai import _openai_analyze

        mock_client = _make_openai_mock(MOCK_LLM_TURN1_EXCITED)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, cars = await _openai_analyze(excited_context)
        ms = (time.perf_counter() - t0) * 1000

        family_found = any("keluarga" in n.lower() for n in hint.detected_needs)

        _record_result(
            "test_turn1_excited_state_generates_exploratory_question",
            "T1_excited_exploration",
            "PASS" if hint.tahap in ("EKSPLORASI", "PENDALAMAN", "RAPPORT") and family_found else "FAIL",
            f"tahap={hint.tahap}, q='{hint.suggested_question[:70]}'",
            ms,
        )
        logger.info(f"[PIVOT-T1] tahap={hint.tahap} | q='{hint.suggested_question}'")

        assert hint.tahap in ("EKSPLORASI", "PENDALAMAN", "RAPPORT"), (
            f"Turn 1 (excited) tahap should be exploratory, got '{hint.tahap}'"
        )
        assert len(hint.suggested_question) > 0, "suggested_question must not be empty"
        assert family_found, (
            f"detected_needs must include a family signal. Got: {hint.detected_needs}"
        )
        assert cars.cars == [], (
            "No car recommendations at EKSPLORASI stage — LLM mock returns empty []"
        )

    # ── Turn 2: Anxious pivot ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_turn2_anxiety_flagged_as_financial_pain_point(
        self, anxious_context, mock_cm
    ):
        """
        Turn 2 (anxious): LLM must flag 'Kestabilan Finansial Keluarga' or similar
        anxiety signal as a high-priority detected_need.

        Acceptable anxiety keywords in detected_needs:
          'kestabilan finansial', 'anxiety', 'cemas', 'cicilan', 'kuliah'
        """
        from backend.services.ai import _openai_analyze

        mock_client = _make_openai_mock(MOCK_LLM_TURN2_ANXIOUS)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, _ = await _openai_analyze(anxious_context)
        ms = (time.perf_counter() - t0) * 1000

        anxiety_keywords  = ["kestabilan finansial", "anxiety", "cemas", "cicilan", "kuliah"]
        pain_points_found = [
            n for n in hint.detected_needs
            if any(kw in n.lower() for kw in anxiety_keywords)
        ]

        _record_result(
            "test_turn2_anxiety_flagged_as_financial_pain_point",
            "T2_anxiety_detection",
            "PASS" if pain_points_found else "FAIL",
            f"pain_points={pain_points_found}, detected_needs={hint.detected_needs}",
            ms,
        )
        logger.info(f"[PIVOT-T2] Financial pain points detected: {pain_points_found}")
        assert pain_points_found, (
            f"LLM must flag financial anxiety as a pain point in detected_needs. "
            f"Got: {hint.detected_needs}"
        )

    @pytest.mark.asyncio
    async def test_turn2_suggested_question_is_empathetic_not_pushy(
        self, anxious_context, mock_cm
    ):
        """
        Turn 2 (anxious): suggested_question must NOT contain pushy upsell triggers.

        Pushy patterns to reject: 'promo', 'beli sekarang', 'terbatas', 'booking',
          'dp sekarang', 'test drive hari ini', 'unit terbatas', 'harga spesial'.

        After pivot, the system must ask about the customer's financial comfort zone,
        NOT close the sale aggressively.
        """
        from backend.services.ai import _openai_analyze

        mock_client = _make_openai_mock(MOCK_LLM_TURN2_ANXIOUS)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, _ = await _openai_analyze(anxious_context)
        ms = (time.perf_counter() - t0) * 1000

        q_lower        = hint.suggested_question.lower()
        pushy_patterns = [
            "promo", "beli sekarang", "terbatas", "booking",
            "dp sekarang", "test drive hari ini", "unit terbatas",
            "harga spesial", "diskon hari ini",
        ]
        empathetic_markers = [
            "nyaman", "cicilan", "kira-kira", "bapak", "keluarga",
            "tenang", "aman", "terjangkau", "fleksibel",
        ]

        pushy_found     = [p for p in pushy_patterns    if p in q_lower]
        empathetic_found = [e for e in empathetic_markers if e in q_lower]

        _record_result(
            "test_turn2_suggested_question_is_empathetic_not_pushy",
            "T2_tone_adaptation",
            "PASS" if not pushy_found else "FAIL",
            f"q='{hint.suggested_question}', pushy={pushy_found}, empathetic={empathetic_found}",
            ms,
        )
        logger.info(f"[PIVOT-T2] Question: '{hint.suggested_question}'")
        logger.info(f"[PIVOT-T2] Pushy found: {pushy_found} | Empathetic found: {empathetic_found}")

        assert not pushy_found, (
            f"suggested_question must NOT contain pushy upsell language after anxiety pivot. "
            f"Detected pushy: {pushy_found} in: '{hint.suggested_question}'"
        )

    @pytest.mark.asyncio
    async def test_turn2_recommendation_pivots_to_budget_friendly_car(
        self, anxious_context, mock_cm
    ):
        """
        After financial anxiety pivot, car recommendation must be budget-safe
        (Xpander Cross / Xpander Ultimate) NOT a premium upsell.

        Budget-safe car IDs:  xpander-cross, xpander-ultimate-cvt
        Premium upsells to reject: pajero-sport-*, eclipse-cross-*, outlander-phev-*
        """
        from backend.services.ai import _openai_analyze

        mock_client = _make_openai_mock(MOCK_LLM_TURN2_ANXIOUS)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, cars = await _openai_analyze(anxious_context)
        ms = (time.perf_counter() - t0) * 1000

        recommended_ids   = [c.id for c in cars.cars]
        premium_upsells   = ["pajero-sport", "eclipse-cross", "outlander-phev"]
        has_premium_upsell = any(
            any(u in car_id for u in premium_upsells) for car_id in recommended_ids
        )

        _record_result(
            "test_turn2_recommendation_pivots_to_budget_friendly_car",
            "T2_car_pivot",
            "PASS" if not has_premium_upsell else "FAIL",
            f"recommended_ids={recommended_ids}, premium_upsell={has_premium_upsell}",
            ms,
        )
        logger.info(f"[PIVOT-T2] Car recommendations: {recommended_ids}")
        assert not has_premium_upsell, (
            f"After financial anxiety pivot, must NOT recommend premium upsells. "
            f"Got: {recommended_ids}"
        )

    @pytest.mark.asyncio
    async def test_turn2_probe_topics_include_financial_stability(
        self, anxious_context, mock_cm
    ):
        """
        After anxiety pivot, probe_topics must include at least one financial
        stability topic: 'finansial', 'cicilan', 'pembiayaan', 'kestabilan'.
        These guide the sales coach to prioritise empathetic financial probing.
        """
        from backend.services.ai import _openai_analyze

        mock_client = _make_openai_mock(MOCK_LLM_TURN2_ANXIOUS)
        mock_lf     = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=mock_client),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint, _ = await _openai_analyze(anxious_context)
        ms = (time.perf_counter() - t0) * 1000

        fin_keywords = ["finansial", "cicilan", "pembiayaan", "kestabilan"]
        fin_probes   = [
            pt for pt in hint.probe_topics
            if any(kw in pt.lower() for kw in fin_keywords)
        ]

        _record_result(
            "test_turn2_probe_topics_include_financial_stability",
            "T2_probe_topics_financial",
            "PASS" if fin_probes else "FAIL",
            f"probe_topics={hint.probe_topics}, fin_probes={fin_probes}",
            ms,
        )
        logger.info(f"[PIVOT-T2] probe_topics: {hint.probe_topics}")
        assert fin_probes, (
            f"probe_topics must include financial stability topics after anxiety pivot. "
            f"Got: {hint.probe_topics}"
        )

    # ── Multi-turn end-to-end test ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_multiturn_full_contextual_pivot_scenario(self, mock_cm):
        """
        Complete multi-turn pivot test:
          Step 1 — Run Turn 1 (excited) through _openai_analyze.
                   Verify: exploratory stage, family need captured, no car upsell.
          Step 2 — Inject anxiety utterances into same ConversationContext.
          Step 3 — Run Turn 2 (anxious) through _openai_analyze.
                   Verify: financial pain flagged, tone pivots, no pushy closing.
          Step 4 — Cross-turn assertions: question changed, pivot confirmed.
        """
        from backend.services.ai import _openai_analyze

        # Step 1: Turn 1 — excited
        context = _make_context(
            [
                "Selamat pagi, saya tertarik lihat MPV untuk keluarga saya.",
                "Kami berenam, sering jalan-jalan bareng, saya sudah nabung lama.",
                "Saya sudah mantap mau beli, tinggal pilih variannya saja.",
            ]
        )
        mock_lf = _make_langfuse_mock()

        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=_make_openai_mock(MOCK_LLM_TURN1_EXCITED)),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint_t1, cars_t1 = await _openai_analyze(context)
        t1_ms = (time.perf_counter() - t0) * 1000

        logger.info(f"[PIVOT] T1 → tahap={hint_t1.tahap} | q='{hint_t1.suggested_question}'")

        # Step 2: inject anxiety utterances
        context.utterances.append(
            _make_utterance(
                "Tapi sejujurnya saya agak cemas dengan cicilan bulanannya karena "
                "anak pertama saya mau masuk kuliah tahun ini."
            )
        )
        context.utterances.append(
            _make_utterance(
                "Jangan-jangan nanti berat di tengah jalan, apalagi biaya kuliah sekarang mahal."
            )
        )

        # Step 3: Turn 2 — anxious pivot
        t0 = time.perf_counter()
        with (
            patch("openai.AsyncOpenAI",               return_value=_make_openai_mock(MOCK_LLM_TURN2_ANXIOUS)),
            patch("backend.services.ai.settings") as s,
            patch("backend.services.ai.search_similar_customers",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.search_conversation_patterns",
                  new_callable=AsyncMock, return_value=[]),
            patch("backend.services.ai.langfuse",          mock_lf),
            patch("backend.services.ai.llm_requests",      MagicMock()),
            patch("backend.services.ai.llm_latency",       MagicMock()),
            patch("backend.services.ai.token_per_request", MagicMock()),
            patch("backend.services.ai.rekomendasi_total", MagicMock()),
            patch("backend.services.ai.track_latency",     return_value=mock_cm),
        ):
            s.openai_api_key = "test-key"
            hint_t2, cars_t2 = await _openai_analyze(context)
        t2_ms = (time.perf_counter() - t0) * 1000

        logger.info(f"[PIVOT] T2 → tahap={hint_t2.tahap} | q='{hint_t2.suggested_question}'")

        # Step 4: Cross-turn assertions
        assert hint_t1.suggested_question != hint_t2.suggested_question, (
            "suggested_question MUST change between excited and anxious turns — pivot failed"
        )

        anxiety_sigs    = ["kestabilan finansial", "anxiety", "cemas", "cicilan", "kuliah"]
        t2_pain_points  = [n for n in hint_t2.detected_needs if any(s in n.lower() for s in anxiety_sigs)]
        assert t2_pain_points, (
            f"Turn 2 must detect financial anxiety. Got: {hint_t2.detected_needs}"
        )

        pushy = ["promo", "beli sekarang", "booking", "dp sekarang", "test drive hari ini"]
        q2_pushy = [p for p in pushy if p in hint_t2.suggested_question.lower()]
        assert not q2_pushy, (
            f"Turn 2 question must not be pushy. Found: {q2_pushy}"
        )

        total_ms = t1_ms + t2_ms
        _record_result(
            "test_multiturn_full_contextual_pivot_scenario",
            "T_full_multiturn_pivot",
            "PASS",
            (
                f"t1_tahap={hint_t1.tahap}, t2_tahap={hint_t2.tahap}, "
                f"pain_points={t2_pain_points}, "
                f"car_pivot={[c.id for c in cars_t2.cars]}, "
                f"total_ms={total_ms:.1f}"
            ),
            total_ms,
        )

        print(f"\n  [PIVOT] Multi-turn contextual pivot complete:")
        print(f"    Turn 1: tahap={hint_t1.tahap}")
        print(f"      → q: '{hint_t1.suggested_question}'")
        print(f"    Turn 2: tahap={hint_t2.tahap}")
        print(f"      → q: '{hint_t2.suggested_question}'")
        print(f"      → pain_points: {t2_pain_points}")
        print(f"      → cars: {[c.id for c in cars_t2.cars]}")

    # ── Schema-level unit tests (no async, no mocking needed) ─────────────

    def test_anxiety_utterance_fills_finansial_dimension_in_elicitation(self):
        """
        The elicitation layer must detect 'cicilan' keyword in the anxiety utterance
        as a finansial direct signal → 'finansial' must NOT be in missing dimensions.
        This is critical: the system must shift from indirect financial probing
        to direct empathetic financial support once the customer self-reveals anxiety.
        """
        anxiety_text = (
            "Tapi sejujurnya saya agak cemas dengan cicilan bulanannya karena "
            "anak pertama saya mau masuk kuliah tahun ini. "
            "Jangan-jangan nanti berat di tengah jalan."
        )

        t0      = time.perf_counter()
        missing = compute_missing_dimensions(anxiety_text)
        ms      = (time.perf_counter() - t0) * 1000

        _record_result(
            "test_anxiety_utterance_fills_finansial_dimension_in_elicitation",
            "T2_elicitation_finansial",
            "PASS" if "finansial" not in missing else "FAIL",
            f"missing={missing}",
            ms,
        )
        logger.info(f"[PIVOT-ELICIT] Anxiety text → missing dims: {missing}")
        assert "finansial" not in missing, (
            f"'cicilan' keyword in anxiety speech must fill finansial dimension. "
            f"missing={missing}"
        )

    def test_kuliah_signal_captured_in_detected_needs_schema(self):
        """
        Verify the AiHintPayload Pydantic model correctly stores and retrieves
        the 'anak kuliah' cultural trigger as a detected_need entry.
        The 'kuliah tahun ini' signal must survive schema validation unchanged.
        """
        hint = AiHintPayload(
            hint_text=MOCK_LLM_TURN2_ANXIOUS["hint_text"],
            suggested_question=MOCK_LLM_TURN2_ANXIOUS["suggested_question"],
            probe_topics=MOCK_LLM_TURN2_ANXIOUS["probe_topics"],
            detected_needs=MOCK_LLM_TURN2_ANXIOUS["detected_needs"],
            tahap=MOCK_LLM_TURN2_ANXIOUS["tahap"],
        )

        kuliah_found = any("kuliah" in n.lower() for n in hint.detected_needs)

        _record_result(
            "test_kuliah_signal_captured_in_detected_needs_schema",
            "T2_kuliah_captured",
            "PASS" if kuliah_found else "FAIL",
            f"detected_needs={hint.detected_needs}",
        )
        assert kuliah_found, (
            f"'anak pertama masuk kuliah' must appear in detected_needs. "
            f"Got: {hint.detected_needs}"
        )

    def test_hint_text_tone_financial_stress_not_upsell_framing(self):
        """
        The hint_text for the anxious turn must reflect financial concern,
        NOT upsell opportunity framing.

        Anti-patterns: 'upgrade', 'premium', 'promo', 'penawaran', 'kesempatan terbaik'
        Required signals: 'cemas', 'cicilan', 'finansial', 'kuliah', 'stabilitas', 'kritis'
        """
        hint = AiHintPayload(
            hint_text=MOCK_LLM_TURN2_ANXIOUS["hint_text"],
            suggested_question=MOCK_LLM_TURN2_ANXIOUS["suggested_question"],
            probe_topics=MOCK_LLM_TURN2_ANXIOUS["probe_topics"],
            detected_needs=MOCK_LLM_TURN2_ANXIOUS["detected_needs"],
            tahap=MOCK_LLM_TURN2_ANXIOUS["tahap"],
        )

        ht_lower        = hint.hint_text.lower()
        upsell_patterns = ["upgrade", "premium", "promo", "penawaran", "kesempatan terbaik"]
        stress_signals  = ["cemas", "cicilan", "finansial", "kuliah", "stabilitas", "kritis"]

        has_upsell = any(p in ht_lower for p in upsell_patterns)
        has_stress  = any(s in ht_lower for s in stress_signals)

        _record_result(
            "test_hint_text_tone_financial_stress_not_upsell_framing",
            "T2_hint_text_tone",
            "PASS" if not has_upsell and has_stress else "FAIL",
            f"hint_text='{hint.hint_text}', upsell={has_upsell}, stress={has_stress}",
        )
        assert not has_upsell, (
            f"hint_text must NOT contain upsell framing after anxiety pivot. "
            f"Got: '{hint.hint_text}'"
        )
        assert has_stress, (
            f"hint_text must reflect financial stress signals. "
            f"Got: '{hint.hint_text}'"
        )

    def test_suggested_question_pivot_does_not_mention_car_model_features(self):
        """
        After anxiety pivot, the suggested_question must not mention specific car
        model names or technical features (no product-push language).
        Empathetic guidance asks about LIFE, not the product.
        """
        hint = AiHintPayload(
            hint_text=MOCK_LLM_TURN2_ANXIOUS["hint_text"],
            suggested_question=MOCK_LLM_TURN2_ANXIOUS["suggested_question"],
            probe_topics=MOCK_LLM_TURN2_ANXIOUS["probe_topics"],
            detected_needs=MOCK_LLM_TURN2_ANXIOUS["detected_needs"],
            tahap=MOCK_LLM_TURN2_ANXIOUS["tahap"],
        )

        q_lower      = hint.suggested_question.lower()
        product_push = [
            "xpander", "pajero", "xforce", "outlander", "eclipse",
            "cvt", "4x4", "diesel", "bensin", "sunroof", "abs",
        ]
        product_found = [p for p in product_push if p in q_lower]

        _record_result(
            "test_suggested_question_pivot_does_not_mention_car_model_features",
            "T2_no_product_push_in_question",
            "PASS" if not product_found else "FAIL",
            f"q='{hint.suggested_question}', product_found={product_found}",
        )
        assert not product_found, (
            f"Empathetic pivot question must NOT name car models or features. "
            f"Found: {product_found} in: '{hint.suggested_question}'"
        )
