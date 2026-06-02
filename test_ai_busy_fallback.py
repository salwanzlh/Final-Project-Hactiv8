"""
Test: _ai_busy RAG Fallback + Elicitation
==========================================
Verifikasi bahwa saat AI (LLM analyze) sedang sibuk memproses utterance sebelumnya,
sistem tetap bisa memberikan hint/pertanyaan ke sales via:
  1. RAG path   — dari conversation-patterns di Pinecone
  2. Elicitation path — dari question bank lokal (tanpa Pinecone)

Skenario yang diuji:
  A. RAG tersedia dan return hasil bagus → hint dari Pinecone pattern
  B. RAG kosong (Pinecone tidak ada/kosong) → fallback ke elicitation
  C. Semua pertanyaan elicitation sudah ditanya → fallback silent (tidak kirim hint)
  D. Deduplikasi → pertanyaan RAG yang sudah ditanya di-skip, pakai yang lain

Usage:
    uv run python test_ai_busy_fallback.py
    uv run python test_ai_busy_fallback.py --live   # pakai Pinecone sungguhan
"""

import asyncio
import argparse
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from backend.models.schemas import Utterance, ConversationContext


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_utterance(speaker: str, text: str) -> Utterance:
    return Utterance(
        id=str(uuid.uuid4()),
        speaker=speaker,
        text=text,
        timestamp=datetime.now(timezone.utc),
        confidence=0.9,
    )


def make_context(utterances_raw: list[tuple[str, str]], asked: list[str] = None) -> ConversationContext:
    ctx = ConversationContext(session_id="test-session")
    ctx.utterances = [make_utterance(spk, txt) for spk, txt in utterances_raw]
    ctx.asked_questions = asked or []
    ctx.last_speaker = utterances_raw[-1][0] if utterances_raw else "unknown"
    return ctx


CONVO_MID = [
    ("sales",    "Selamat datang Pak, ada yang bisa saya bantu?"),
    ("customer", "Iya Mbak, saya lagi cari mobil keluarga"),
    ("sales",    "Oh keluarga, wah bagus. Biasanya pergi bareng ada berapa orang?"),
    ("customer", "Ada 5 orang, saya istri sama 3 anak"),
    ("sales",    "Kalau weekend biasanya sekeluarga jalan-jalan kemana Pak?"),
    ("customer", "Sering ke Puncak atau ke pantai, lumayan jauh"),
]


# ── Shared capture helper ───────────────────────────────────────────────────────

class MessageCapture:
    def __init__(self):
        self.messages: list[dict] = []

    async def send(self, session_id: str, msg_type: str, payload: dict):
        self.messages.append({"type": msg_type, "payload": payload})
        print(f"  → [{msg_type}] {payload.get('suggested_question', payload)}")

    def hints(self) -> list[dict]:
        return [m for m in self.messages if m["type"] == "ai_hint"]


# ── Import target under test ────────────────────────────────────────────────────
# Import setelah mock setup supaya patch bisa intercept

from backend.routers.ws import _rag_fallback_hint  # noqa: E402


# ── Test A: RAG path ────────────────────────────────────────────────────────────

async def test_a_rag_returns_result():
    print("\n" + "="*60)
    print("SKENARIO A — RAG return hasil bagus")
    print("  Ekspektasi: hint diambil dari Pinecone conversation-patterns")
    print("="*60)

    rag_patterns = [
        {
            "stage": "probing",
            "effective_next_question": "Jalan yang biasa dilalui kondisinya gimana Pak, mulus atau berbatu?",
            "why_effective": "Membuka cerita mobilitas sekaligus preferensi medan",
            "outcome": "customer_engaged",
        },
        {
            "stage": "probing",
            "effective_next_question": "Sudah lama pakai kendaraan yang sekarang?",
            "why_effective": "Reveal urgency + proxy finansial",
            "outcome": "sale_progressed",
        },
    ]

    capture = MessageCapture()
    ctx = make_context(CONVO_MID)

    with patch("backend.routers.ws.search_conversation_patterns", return_value=rag_patterns), \
         patch("backend.routers.ws.session_manager.send", side_effect=capture.send):
        await _rag_fallback_hint("test-session", ctx)

    hints = capture.hints()
    assert hints, "GAGAL: tidak ada AI_HINT yang dikirim"
    q = hints[0]["payload"]["suggested_question"]
    assert q == rag_patterns[0]["effective_next_question"], f"GAGAL: pertanyaan bukan dari RAG: '{q}'"
    assert hints[0]["payload"]["tahap"] == "RAG_FALLBACK"
    assert q in ctx.asked_questions, "GAGAL: pertanyaan tidak masuk asked_questions"
    print(f"\n  ✓ PASS  |  Pertanyaan: '{q}'")
    print(f"  ✓ PASS  |  asked_questions updated: {ctx.asked_questions}")


# ── Test B: Elicitation fallback ────────────────────────────────────────────────

async def test_b_elicitation_fallback():
    print("\n" + "="*60)
    print("SKENARIO B — RAG kosong → elicitation fallback")
    print("  Ekspektasi: hint dari question bank lokal (tanpa Pinecone)")
    print("="*60)

    capture = MessageCapture()
    ctx = make_context(CONVO_MID)

    with patch("backend.routers.ws.search_conversation_patterns", return_value=[]), \
         patch("backend.routers.ws.session_manager.send", side_effect=capture.send):
        await _rag_fallback_hint("test-session", ctx)

    hints = capture.hints()
    assert hints, "GAGAL: tidak ada AI_HINT yang dikirim (elicitation juga tidak berhasil?)"
    q = hints[0]["payload"]["suggested_question"]
    assert q, "GAGAL: pertanyaan kosong"
    assert q in ctx.asked_questions
    print(f"\n  ✓ PASS  |  Pertanyaan dari elicitation: '{q}'")


# ── Test C: Semua dimensi sudah tergali ─────────────────────────────────────────

async def test_c_all_dimensions_covered():
    print("\n" + "="*60)
    print("SKENARIO C — Semua dimensi sudah tergali, elicitation habis")
    print("  Ekspektasi: tidak ada hint yang dikirim (silent skip)")
    print("="*60)

    # Percakapan yang sudah cover semua 4 dimensi
    full_convo = [
        ("sales",    "Selamat datang Pak"),
        ("customer", "Iya, saya lagi cari mobil. Anak saya 3, istri juga ikut"),
        ("customer", "Sering mudik ke Jawa Tengah, 400 km lebih tiap bulan"),
        ("customer", "Kira-kira cicilan 5 juta per bulan"),
        ("customer", "Mobil sekarang sudah sering masuk bengkel, sudah tua"),
    ]
    ctx = make_context(full_convo)

    # Tandai semua pertanyaan elicitation sebagai sudah ditanya
    from backend.services.elicitation import INDIRECT_QUESTIONS
    all_questions = []
    for qs in INDIRECT_QUESTIONS.values():
        all_questions.extend(q["question"] for q in qs)
    ctx.asked_questions = all_questions

    capture = MessageCapture()

    with patch("backend.routers.ws.search_conversation_patterns", return_value=[]), \
         patch("backend.routers.ws.session_manager.send", side_effect=capture.send):
        await _rag_fallback_hint("test-session", ctx)

    hints = capture.hints()
    assert not hints, f"GAGAL: hint tetap dikirim padahal semua pertanyaan sudah habis: {hints}"
    print(f"\n  ✓ PASS  |  Tidak ada hint dikirim (silent skip, benar)")


# ── Test D: RAG deduplication ────────────────────────────────────────────────────

async def test_d_rag_deduplication():
    print("\n" + "="*60)
    print("SKENARIO D — Pertanyaan RAG pertama sudah pernah ditanya")
    print("  Ekspektasi: skip ke pertanyaan RAG berikutnya")
    print("="*60)

    rag_patterns = [
        {
            "stage": "probing",
            "effective_next_question": "Sudah lama pakai kendaraan yang sekarang?",
            "outcome": "customer_engaged",
        },
        {
            "stage": "probing",
            "effective_next_question": "Jalan yang biasa dilalui kondisinya gimana Pak?",
            "outcome": "sale_progressed",
        },
    ]

    # Tandai pertanyaan pertama sudah ditanya
    ctx = make_context(CONVO_MID, asked=["Sudah lama pakai kendaraan yang sekarang?"])
    capture = MessageCapture()

    with patch("backend.routers.ws.search_conversation_patterns", return_value=rag_patterns), \
         patch("backend.routers.ws.session_manager.send", side_effect=capture.send):
        await _rag_fallback_hint("test-session", ctx)

    hints = capture.hints()
    assert hints, "GAGAL: tidak ada hint dikirim"
    q = hints[0]["payload"]["suggested_question"]
    assert q == rag_patterns[1]["effective_next_question"], \
        f"GAGAL: seharusnya pakai pertanyaan ke-2, dapat: '{q}'"
    print(f"\n  ✓ PASS  |  Pertanyaan ke-1 di-skip, pakai ke-2: '{q}'")


# ── Test E: Live Pinecone (opsional) ─────────────────────────────────────────────

async def test_e_live_pinecone():
    print("\n" + "="*60)
    print("SKENARIO E — Live test dengan Pinecone sungguhan")
    print("  (dijalankan hanya dengan flag --live)")
    print("="*60)

    from backend.services.rag import search_conversation_patterns

    ctx = make_context(CONVO_MID)
    query = " ".join(u.text for u in ctx.utterances[-5:])
    print(f"\n  Query ke Pinecone:\n  '{query[:100]}...'")

    patterns = await search_conversation_patterns(query, top_k=5)

    if not patterns:
        print("\n  ⚠ Pinecone kosong atau tidak terkonfigurasi — elicitation akan kick in")
    else:
        print(f"\n  ✓ {len(patterns)} pattern ditemukan:")
        for i, p in enumerate(patterns, 1):
            print(f"    {i}. [{p.get('stage','?')}] '{p.get('effective_next_question','')}'")

    capture = MessageCapture()

    with patch("backend.routers.ws.session_manager.send", side_effect=capture.send):
        await _rag_fallback_hint("test-session", ctx)

    hints = capture.hints()
    if hints:
        q = hints[0]["payload"]["suggested_question"]
        tahap = hints[0]["payload"]["tahap"]
        print(f"\n  ✓ PASS  |  [{tahap}] Pertanyaan: '{q}'")
    else:
        print("\n  ⚠ Tidak ada hint dikirim (semua pertanyaan sudah ditanya?)")


# ── Runner ──────────────────────────────────────────────────────────────────────

async def main(live: bool = False):
    print("\n" + "█"*60)
    print("  TEST: _ai_busy RAG Fallback + Elicitation")
    print("█"*60)

    await test_a_rag_returns_result()
    await test_b_elicitation_fallback()
    await test_c_all_dimensions_covered()
    await test_d_rag_deduplication()

    if live:
        await test_e_live_pinecone()

    print("\n" + "="*60)
    print("  SEMUA SKENARIO PASS ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Jalankan test dengan Pinecone sungguhan")
    args = parser.parse_args()
    asyncio.run(main(live=args.live))
