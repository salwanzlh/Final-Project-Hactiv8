"""
test_rag_fallback_priority.py
=================================
Unit tests untuk _rag_fallback_hint — sistem fallback bertingkat saat LLM sibuk.

Fungsi ini memiliki 3 jalur prioritas:
  Prioritas 1 — Topic Transitions  : detect_topic → search_topic_transitions → pick_transition_question
  Prioritas 2 — RAG Patterns       : search_conversation_patterns → filter candidate → pertanyaan baru
  Prioritas 3 — Elicitation Engine : compute_missing_dimensions → get_next_question

Setup:
  uv add --dev pytest pytest-asyncio
  uv run pytest test_rag_fallback_priority.py -v
"""

import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from backend.models.schemas import ConversationContext, Utterance, WsMessageType
from backend.routers.ws import _rag_fallback_hint


SESSION_ID = "test-session-fallback"


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    utterances = [_make_utterance(t) for t in (texts or ["Saya tertarik mobil SUV"])]
    return ConversationContext(
        session_id=SESSION_ID,
        utterances=utterances,
        asked_questions=list(asked or []),
        detected_needs=list(needs or ["SUV"]),
    )


# ── Test 1: Prioritas 1 Sukses — Topic Transition ─────────────────────────────

@pytest.mark.asyncio
async def test_priority1_topic_transition_success():
    """
    P1 berhasil: detect_topic menemukan topik, search_topic_transitions mengembalikan
    data transisi historis, dan pick_transition_question memberikan pertanyaan baru
    yang belum pernah ditanya sebelumnya.

    Assert:
    - session_manager.send dipanggil tepat satu kali dengan hint_text berformat P1
      yaitu dimulai dengan "Arahkan ke topik <to_topic>"
    - suggested_question tersimpan di context.asked_questions
    - search_conversation_patterns (P2) tidak dipanggil karena P1 sudah cukup
    """
    context = _make_context(texts=["Saya bareng keluarga biasanya"], asked=[])

    expected_question = "Weekend biasanya ngapain kalau bareng keluarga?"
    expected_to_topic = "mobility"  # ada di TOPIC_TO_ELICITATION_DIM → "mobilitas"

    mock_transitions = [
        {
            "from_topic": "family",
            "to_topic": expected_to_topic,
            "trigger_question": expected_question,
            "outcome": "closing",
            "naturalness": "smooth",
        }
    ]

    with (
        patch("backend.routers.ws.detect_topic", return_value="family"),
        patch(
            "backend.routers.ws.search_topic_transitions",
            new_callable=AsyncMock,
            return_value=mock_transitions,
        ),
        patch(
            "backend.routers.ws.pick_transition_question",
            return_value=(expected_question, expected_to_topic),
        ),
        patch(
            "backend.routers.ws.search_conversation_patterns",
            new_callable=AsyncMock,
        ) as mock_rag,
        patch(
            "backend.routers.ws.session_manager.send",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await _rag_fallback_hint(SESSION_ID, context)

    # Pertanyaan P1 harus masuk ke asked_questions
    assert expected_question in context.asked_questions, (
        "Pertanyaan P1 harus tersimpan di context.asked_questions"
    )

    # P2 tidak boleh dipanggil — P1 sudah berhasil
    mock_rag.assert_not_called()

    # send dipanggil tepat satu kali
    mock_send.assert_called_once()
    session_arg, type_arg, payload = mock_send.call_args.args

    assert session_arg == SESSION_ID
    assert type_arg == WsMessageType.AI_HINT
    assert payload["suggested_question"] == expected_question
    assert payload["tahap"] == "RAG_FALLBACK"
    assert payload["probe_topics"] == []

    # Format hint_text khas Prioritas 1: "Arahkan ke topik <to_topic> ..."
    assert payload["hint_text"].startswith("Arahkan ke topik"), (
        f"hint_text P1 harus dimulai dengan 'Arahkan ke topik', dapat: '{payload['hint_text']}'"
    )
    assert "mobility" in payload["hint_text"], (
        "hint_text P1 harus menyebutkan to_topic 'mobility'"
    )


# ── Test 2: Prioritas 1 Gagal, Prioritas 2 Sukses — RAG Patterns ─────────────

@pytest.mark.asyncio
async def test_priority2_rag_patterns_success_when_p1_fails():
    """
    P1 gagal: detect_topic mengembalikan None sehingga seluruh blok topic-transition
    di-skip. P2 berhasil: search_conversation_patterns mengembalikan satu pola valid
    dengan outcome bukan 'question_deflected' dan pertanyaannya belum pernah ditanya.

    Assert:
    - Pertanyaan dari pola RAG tersimpan di context.asked_questions
    - Payload dikirim dengan suggested_question yang benar dan tahap RAG_FALLBACK
    - compute_missing_dimensions (P3) tidak dipanggil karena P2 sudah berhasil
    """
    context = _make_context(texts=["Saya belum tahu mau pilih yang mana"], asked=[])

    rag_question = "Jalan yang biasa Bapak lalui kondisinya gimana, mulus atau lumayan berat?"
    mock_patterns = [
        {
            "stage": "probing",
            "effective_next_question": rag_question,
            "outcome": "customer_engaged",  # bukan "question_deflected"
        },
    ]

    with (
        patch("backend.routers.ws.detect_topic", return_value=None),
        patch(
            "backend.routers.ws.search_conversation_patterns",
            new_callable=AsyncMock,
            return_value=mock_patterns,
        ),
        patch("backend.routers.ws.compute_missing_dimensions") as mock_compute,
        patch("backend.routers.ws.get_next_question") as mock_get_q,
        patch(
            "backend.routers.ws.session_manager.send",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await _rag_fallback_hint(SESSION_ID, context)

    # Pertanyaan dari RAG harus masuk ke asked_questions
    assert rag_question in context.asked_questions, (
        "Pertanyaan RAG harus tersimpan di context.asked_questions"
    )

    # P3 tidak boleh dipanggil — P2 sudah berhasil
    mock_compute.assert_not_called()
    mock_get_q.assert_not_called()

    # Payload dikirim dengan benar
    mock_send.assert_called_once()
    session_arg, type_arg, payload = mock_send.call_args.args

    assert session_arg == SESSION_ID
    assert type_arg == WsMessageType.AI_HINT
    assert payload["suggested_question"] == rag_question
    assert payload["tahap"] == "RAG_FALLBACK"
    assert payload["probe_topics"] == []


# ── Test 3: Prioritas 1 & 2 Gagal, Prioritas 3 Sukses — Elicitation ──────────

@pytest.mark.asyncio
async def test_priority3_elicitation_success_when_p1_and_p2_fail():
    """
    P1 & P2 gagal: detect_topic None, search_conversation_patterns mengembalikan
    list kosong. P3 berhasil: compute_missing_dimensions menemukan dimensi yang
    belum tergali, get_next_question mengembalikan template pertanyaan dari bank.

    Assert:
    - Pertanyaan dari Elicitation Engine tersimpan di context.asked_questions
    - hint_text berformat P3: "Gali <reveals> dari customer."
    - Payload dikirim ke frontend dengan type AI_HINT
    """
    context = _make_context(texts=["Hmm, saya masih nimbang-nimbang"], asked=[])

    elicitation_result = {
        "question": "Kantornya jauh dari rumah atau masih deket-deket?",
        "reveals": ["mobilitas", "rutinitas"],
        "natural_because": "obrolan ringan keseharian",
    }

    with (
        patch("backend.routers.ws.detect_topic", return_value=None),
        patch(
            "backend.routers.ws.search_conversation_patterns",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "backend.routers.ws.compute_missing_dimensions",
            return_value=["mobilitas", "keluarga"],
        ),
        patch(
            "backend.routers.ws.get_next_question",
            return_value=elicitation_result,
        ),
        patch(
            "backend.routers.ws.session_manager.send",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await _rag_fallback_hint(SESSION_ID, context)

    # Pertanyaan dari elicitation harus masuk ke asked_questions
    assert elicitation_result["question"] in context.asked_questions, (
        "Pertanyaan Elicitation harus tersimpan di context.asked_questions"
    )

    # send dipanggil tepat satu kali
    mock_send.assert_called_once()
    session_arg, type_arg, payload = mock_send.call_args.args

    assert session_arg == SESSION_ID
    assert type_arg == WsMessageType.AI_HINT
    assert payload["suggested_question"] == elicitation_result["question"]
    assert payload["tahap"] == "RAG_FALLBACK"

    # Format hint_text khas Prioritas 3: "Gali <dim1> & <dim2> dari customer."
    assert "mobilitas" in payload["hint_text"], (
        "hint_text P3 harus menyebutkan dimensi yang digali dari 'reveals'"
    )
    assert "dari customer" in payload["hint_text"], (
        f"hint_text P3 harus berformat 'Gali ... dari customer.', dapat: '{payload['hint_text']}'"
    )


# ── Test 4: Semua Prioritas Gagal — Return Dini (Zonk) ────────────────────────

@pytest.mark.asyncio
async def test_all_priorities_fail_early_return_no_send():
    """
    Semua jalur gagal: P1 (detect_topic None), P2 (RAG kosong), P3 (get_next_question
    mengembalikan None karena semua dimensi sudah tergali atau bank pertanyaan habis).
    Fungsi harus return dini sebelum memanggil send atau mengubah asked_questions.

    Assert:
    - session_manager.send TIDAK dipanggil sama sekali
    - context.asked_questions tidak berubah (tidak ada pertanyaan baru ditambahkan)
    """
    initial_asked = [
        "Sudah lama pakai kendaraan yang sekarang?",
        "Kantornya jauh dari rumah?",
    ]
    context = _make_context(
        texts=["iya saya mau lihat-lihat dulu"],
        asked=initial_asked.copy(),
    )

    with (
        patch("backend.routers.ws.detect_topic", return_value=None),
        patch(
            "backend.routers.ws.search_conversation_patterns",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("backend.routers.ws.compute_missing_dimensions", return_value=[]),
        patch("backend.routers.ws.get_next_question", return_value=None),
        patch(
            "backend.routers.ws.session_manager.send",
            new_callable=AsyncMock,
        ) as mock_send,
    ):
        await _rag_fallback_hint(SESSION_ID, context)

    # Tidak ada hint yang dikirim ke frontend
    mock_send.assert_not_called()

    # asked_questions tidak boleh berubah
    assert context.asked_questions == initial_asked, (
        "asked_questions tidak boleh berubah saat semua jalur gagal dan fungsi return dini"
    )
