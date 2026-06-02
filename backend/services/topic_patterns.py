"""
backend/services/topic_patterns.py
====================================
Runtime topic detection dan query transisi topik dari Pinecone.

Dipakai di AI-busy fallback: deteksi topik percakapan saat ini,
lalu cari trigger question historis yang berhasil membuka topik berikutnya.

Records disimpan di namespace "conversation-patterns" dengan
metadata record_type="topic_transition" agar tidak campur dengan
sequence patterns yang sudah ada.
"""
import asyncio
import logging

from backend.config import settings

logger = logging.getLogger(__name__)

TOPICS: dict[str, list[str]] = {
    "greeting":        ["halo", "selamat", "datang", "pertama kali", "mampir", "lihat-lihat"],
    "small_talk":      ["dari mana", "asal", "jauh", "macet", "cuaca", "lama", "sendiri"],
    "current_vehicle": ["pakai", "motor", "mobil sekarang", "kendaraan", "avanza", "jazz",
                        "innova", "sudah lama", "tahun", "ganti"],
    "family":          ["anak", "istri", "suami", "keluarga", "orang tua", "bareng",
                        "berapa orang", "sekeluarga"],
    "mobility":        ["keseharian", "aktivitas", "kantor", "luar kota", "mudik",
                        "perjalanan", "jalan", "sering", "rutin", "medan"],
    "budget":          ["budget", "harga", "dp", "cicilan", "kredit", "juta", "cash",
                        "nabung", "bonus", "sanggup"],
    "urgency":         ["kapan", "rencana", "segera", "bulan ini", "tahun ini",
                        "masih nimbang", "sudah lama cari"],
    "hesitation":      ["pikir dulu", "diskusi dulu", "tanya istri", "belum pasti",
                        "nanti kabarin", "lihat-lihat dulu"],
    "interest_signal": ["tertarik", "bagus", "cocok", "menarik", "oke", "wah",
                        "boleh juga", "kayaknya"],
    "recommendation":  ["rekomendasi", "cocok untuk", "pas untuk", "saya sarankan",
                        "xpander", "pajero", "xforce", "outlander"],
}

# Mana topik → dimensi elicitation yang paling relevan
# Dipakai untuk re-prioritize elicitation saat AI busy
TOPIC_TO_ELICITATION_DIM: dict[str, str] = {
    "family":          "keluarga",
    "current_vehicle": "finansial",
    "budget":          "finansial",
    "mobility":        "mobilitas",
    "urgency":         "urgency",
    "hesitation":      "urgency",
    "interest_signal": "urgency",
}

TRANSITION_NS = "conversation-patterns"
MIN_SCORE     = 0.30


def detect_topic(text: str) -> str | None:
    """Detect topik dominan dari teks utterance dengan keyword scoring."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPICS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def extract_transitions(transcript: list[dict]) -> list[dict]:
    """
    Extract topic transition bigrams dari satu session transcript.

    Input : [{"speaker": "sales"|"customer", "text": "..."}]
    Output: list of transition dicts
    """
    transitions: list[dict] = []
    prev_topic            = None
    prev_sales_question   = None

    for i, utt in enumerate(transcript):
        topic = detect_topic(utt["text"])
        if topic is None:
            continue

        if utt["speaker"] == "sales" and "?" in utt["text"]:
            prev_sales_question = utt["text"]

        if topic != prev_topic and prev_topic is not None:
            customer_preview = ""
            for j in range(i, min(i + 3, len(transcript))):
                if transcript[j]["speaker"] == "customer":
                    customer_preview = transcript[j]["text"][:100]
                    break

            transitions.append({
                "from_topic":                prev_topic,
                "to_topic":                  topic,
                "trigger_question":          prev_sales_question or "",
                "customer_response_preview": customer_preview,
                "transition_turn":           i,
            })

        prev_topic = topic

    return transitions


def build_transition_embed_text(trans: dict, session_meta: dict) -> str:
    """Bangun teks yang akan di-embed untuk satu transisi — sama antara seed dan query."""
    outcome = session_meta.get("outcome", "unknown")
    tipe    = session_meta.get("customer_tipe", "unknown")

    lines = [
        f"Transisi topik: dari [{trans['from_topic']}] ke [{trans['to_topic']}]",
        f"Tipe customer: {tipe} | Outcome sesi: {outcome}",
    ]
    if trans["trigger_question"]:
        lines.append(f"Pertanyaan yang memicu: {trans['trigger_question']}")
    if trans["customer_response_preview"]:
        lines.append(f"Respons customer: {trans['customer_response_preview']}")
    return "\n".join(lines)


async def search_topic_transitions(
    current_topic: str,
    recent_utterances: list,
    top_k: int = 5,
) -> list[dict]:
    """
    Cari transisi historis dari current_topic di Pinecone.
    Reuses Pinecone inference dari rag.py — model sama dengan saat seed.

    Returns: list of metadata dicts (with trigger_question, to_topic, outcome, dll)
    """
    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        return []

    from backend.services.rag import _embed, _pinecone_index

    recent_text = " ".join(u.text for u in recent_utterances[-3:])
    query_text  = (
        f"Transisi topik: dari [{current_topic}] ke topik berikutnya\n"
        f"Konteks saat ini: {recent_text}"
    )

    try:
        embedding = await _embed(query_text)
        index     = await asyncio.to_thread(_pinecone_index)

        results = await asyncio.to_thread(
            index.query,
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=TRANSITION_NS,
            filter={
                "record_type": {"$eq": "topic_transition"},
                "from_topic":  {"$eq": current_topic},
            },
        )
        return [
            m.metadata for m in results.matches
            if m.score >= MIN_SCORE and m.metadata
        ]
    except Exception as exc:
        logger.warning(f"[TopicTransition] Pinecone lookup failed: {exc}")
        return []


def pick_transition_question(
    transitions: list[dict],
    asked_questions: list[str],
    preferred_outcome: str = "closing",
) -> tuple[str, str | None]:
    """
    Pilih trigger question terbaik dari hasil query transisi.

    Prioritas: outcome=closing > smooth > belum pernah ditanya.
    Returns: (question, to_topic) — keduanya "" / None jika tidak ada.
    """
    from backend.services.elicitation import _similar

    ordered = sorted(
        [t for t in transitions if t.get("trigger_question")],
        key=lambda t: (
            t.get("outcome") == preferred_outcome,
            t.get("naturalness") != "abrupt",
        ),
        reverse=True,
    )
    for t in ordered:
        q = t.get("trigger_question", "")
        if q and not any(_similar(q, asked) for asked in asked_questions):
            return q, t.get("to_topic")
    return "", None
