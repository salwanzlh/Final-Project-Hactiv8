"""
Speaker Classifier — klasifikasi siapa yang berbicara: sales atau customer.

Pipeline:
  1. Heuristic: 1-2 kata → prediksi LAWAN BICARA (reaksi pendek selalu alternating)
  2. LLM (gpt-4o-mini): analisis kalimat + history → (speaker, confidence)
  3. Fallback jika LLM tidak yakin:
     - Tanpa history       → trust LLM
     - conf < 0.4          → alternating (lawan dari last_speaker)
     - conf 0.4–0.6        → trust LLM result
"""
import time
import json
import logging
from backend.config import settings
from backend.models.schemas import Utterance
from backend.services.metrics import classifier_latency, classifier_confidence, classifier_lowconf_total

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6   # di bawah ini → fallback
_SHORT_TEXT_WORDS     = 2     # ≤ kata ini → skip LLM (hanya reaksi 1-2 kata)


def _format_history(utterances: list[Utterance]) -> str:
    if not utterances:
        return "(belum ada percakapan)"
    lines = []
    for u in utterances:
        if u.speaker == "sales":
            label = "SALES"
        elif u.speaker == "customer":
            label = "CUSTOMER"
        else:
            label = "?"
        lines.append(f"{label}: {u.text}")
    return "\n".join(lines)


async def classify_speaker(
    text: str,
    recent_history: list[Utterance],
    last_speaker: str,
) -> tuple[str, float]:
    _start = time.perf_counter()
    speaker, confidence = await _classify_speaker_impl(text, recent_history, last_speaker)
    classifier_latency.observe(time.perf_counter() - _start)
    classifier_confidence.observe(confidence)
    if confidence < _CONFIDENCE_THRESHOLD:
        classifier_lowconf_total.inc()
    return speaker, confidence


async def _classify_speaker_impl(
    text: str,
    recent_history: list[Utterance],
    last_speaker: str,
) -> tuple[str, float]:
    """
    Returns (speaker, confidence).
    speaker    : 'sales' | 'customer' | 'unknown'
    confidence : 0.0 – 1.0
    """
    clean = text.strip()
    if not clean:
        return last_speaker or "unknown", 0.0

    # ── Heuristic: hanya 1-2 kata (reaksi singkat) ────────────────────
    # Reaksi sangat pendek hampir selalu datang dari LAWAN BICARA, bukan pembicara yang sama.
    # Contoh: Sales bicara → "Iya." pasti Customer. Customer bicara → "Baik." pasti Sales.
    word_count = len(clean.split())
    if word_count <= _SHORT_TEXT_WORDS:
        opposite = "customer" if last_speaker == "sales" else "sales" if last_speaker == "customer" else "unknown"
        logger.debug(f"[Speaker] Short text ({word_count}w), predicting opposite of {last_speaker} → {opposite}")
        return opposite, 0.35

    # ── LLM classifier ────────────────────────────────────────────────
    history_text = _format_history(recent_history)
    last_label = (
        "SALES" if last_speaker == "sales"
        else "CUSTOMER" if last_speaker == "customer"
        else "tidak diketahui"
    )

    user_prompt = f"""Riwayat percakapan terakhir:
{history_text}

Pembicara terakhir yang diketahui: {last_label}

Kalimat baru yang harus diklasifikasikan:
"{clean}"

Berikan confidence RENDAH (< 0.5) jika kalimat ambigu, kemungkinan overlap, atau konteks belum cukup."""

    try:
        from openai import AsyncOpenAI
        from backend.services.langfuse_client import langfuse
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        messages = [
            {
                "role": "system",
                "content": (
                    """Klasifikasikan speaker percakapan showroom Mitsubishi Indonesia: SALES atau CUSTOMER.
                    ━━━ SINYAL ━━━
                    SALES pasti  : "Ada yang bisa saya bantu", "Selamat datang", "Silakan", memanggil "Pak/Bu",
                                menawarkan test drive/brosur/kredit, sebut "unit kami / promo bulan ini".
                    CUSTOMER pasti: memanggil "Mbak/Mas", "Saya lagi cari/mau/butuh", cerita keluarga/rutinitas pribadi,
                                "Budget/DP/cicilan saya...", bertanya harga/stok/kredit sebagai respons.
                    SALES kuat   : menjelaskan spesifikasi/harga atas inisiatif sendiri, menggali kebutuhan customer.
                    CUSTOMER kuat: menyatakan preferensi pribadi, ungkapkan keberatan ("mahal ya", "pikir dulu").
                    LEMAH        : sapaan netral / jawaban pendek → assign ke speaker BERBEDA dari sebelumnya.
                    AMBIGU       : confidence 0.50, tetap pilih, jangan null.

                    ━━━ EXAMPLES ━━━
                    "Halo Mbak selamat siang" | riwayat: - → {"speaker":"customer","confidence":0.95}
                    "Ada yang bisa saya bantu Pak?" | riwayat: [customer] → {"speaker":"sales","confidence":0.97}
                    "Ada 5 orang, saya istri sama 3 anak" | riwayat: [sales: tanya berapa orang] → {"speaker":"customer","confidence":0.95}
                    "Kalau gitu berarti kita butuh yang nyaman buat jarak jauh" | riwayat: [customer: sering mudik] → {"speaker":"sales","confidence":0.85}
                    "Wah lumayan ya, bisa dicicil tidak Mas?" | riwayat: [sales: sebut harga] → {"speaker":"customer","confidence":0.96}
                    "Iya" | riwayat: [sales: tanya keluar kota] → {"speaker":"customer","confidence":0.65}
                    "Selamat pagi" | riwayat: - → {"speaker":"sales","confidence":0.72}

                    Jawab HANYA JSON: {"speaker":"sales","confidence":0.85}"""
                    ),
            },
            {"role": "user", "content": user_prompt},
        ]
        trace = langfuse.trace(name="classify_speaker", input={"text": clean[:100]})
        generation = trace.generation(name="speaker_classifier_llm", model="gpt-4.1", input=user_prompt)
        response = await client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=30,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw  = response.choices[0].message.content.strip()
        generation.end(
            output=raw,
            usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
        )
        data = json.loads(raw)

        speaker    = data.get("speaker", "unknown")
        confidence = float(data.get("confidence", 0.5))

        if speaker not in ("sales", "customer"):
            speaker = "unknown"

        # Fallback ketika LLM tidak yakin
        if confidence < _CONFIDENCE_THRESHOLD:
            if not recent_history:
                # Awal percakapan: trust LLM, tapi set minimum confidence
                logger.info(f"[Speaker] No history, trust LLM: '{clean[:40]}' → {speaker} ({confidence:.2f})")
                return speaker, max(confidence, 0.35)

            # Ada history: asumsikan bergantian (percakapan sales-customer selalu alternating)
            # Jika LLM sangat tidak yakin (< 0.4), pakai lawan dari last_speaker
            if confidence < 0.4 and last_speaker in ("sales", "customer"):
                opposite = "customer" if last_speaker == "sales" else "sales"
                logger.info(
                    f"[Speaker] Very low conf ({confidence:.2f}) '{clean[:40]}' "
                    f"→ alternating fallback {last_speaker}→{opposite}"
                )
                return opposite, confidence

            # Moderate uncertainty (0.4–0.6): trust LLM result
            logger.info(f"[Speaker] Moderate conf ({confidence:.2f}) '{clean[:40]}' → trust LLM: {speaker}")
            return speaker, confidence

        logger.info(f"[Speaker] '{clean[:50]}' → {speaker} ({confidence:.2f})")
        return speaker, confidence

    except Exception as e:
        logger.warning(f"[Speaker] LLM failed: {e} — fallback last_speaker={last_speaker}")
        return last_speaker or "unknown", 0.0
