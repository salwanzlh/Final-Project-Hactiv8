"""
Comparison test: Featherless Gemma-4-E4B-it  vs  GPT-4o-mini
for the AI analysis task (analyze_conversation).

Usage:
    python test_ai_compare.py

Prints side-by-side results for 3 realistic conversation scenarios:
  1. Sales-led (customer just walked in, no car mention yet)
  2. Customer-led (customer opened with a car question)
  3. Mid-conversation (needs partially discovered, customer asking feature question)

What to watch:
  - JSON parse success / failure
  - hint_text: is it insightful and natural?
  - suggested_question: is it lifestyle-based, not feature-based?
  - probe_topics: relevant and concise?
  - detected_needs: correctly inferred?
  - recommended_car_ids: appropriate (empty while basa-basi, filled when needs clear)?
  - latency: how long does each call take?
"""

import asyncio
import json
import time
import textwrap
from datetime import datetime, timezone
from backend.config import settings
from backend.models.schemas import ConversationContext, Utterance
from backend.db.car_db import get_all_cars, get_car_by_id

# ── Fake utterance helper ────────────────────────────────────────────
_id_counter = 0

def utt(speaker: str, text: str) -> Utterance:
    global _id_counter
    _id_counter += 1
    return Utterance(
        id=f"test-{_id_counter}",
        speaker=speaker,
        text=text,
        timestamp=datetime.now(timezone.utc),
        confidence=0.9,
    )


# ── Test scenarios ───────────────────────────────────────────────────
SCENARIOS = {
    "1. Sales-led (basa-basi stage)": ConversationContext(
        session_id="test-1",
        utterances=[
            utt("sales",    "Selamat siang Pak, ada yang bisa saya bantu?"),
            utt("customer", "Iya siang, saya mau lihat-lihat dulu."),
            utt("sales",    "Silakan Pak, santai saja. Bapak dari mana asalnya?"),
            utt("customer", "Dari Depok. Tadi macet banget di jalan."),
            utt("sales",    "Iya Pak, Depok ke sini memang lumayan. Biasanya pakai kendaraan apa Pak?"),
            utt("customer", "Pakai motor sekarang. Nah itu yang mau saya pikir-pikir."),
        ],
    ),
    "2. Customer-led (needs discovery stage)": ConversationContext(
        session_id="test-2",
        utterances=[
            utt("customer", "Mas saya mau tanya soal Xpander, harganya berapa sekarang?"),
            utt("sales",    "Xpander ada beberapa varian Pak, mulai 260 jutaan sampai 310 jutaan."),
            utt("customer", "Oh gitu, saya lagi cari mobil keluarga. Anak saya tiga."),
            utt("sales",    "Wah pas banget Pak, Xpander memang populer untuk keluarga."),
            utt("customer", "Iya tapi saya juga sering bawa ke luar kota, ke Jawa Tengah gitu."),
        ],
    ),
    "3. Customer asks feature question (hybrid stage)": ConversationContext(
        session_id="test-3",
        utterances=[
            utt("customer", "Kemarin saya lihat-lihat di internet, katanya Xforce ada fitur ADAS ya?"),
            utt("sales",    "Betul Pak, ada Forward Collision Warning, Lane Departure Warning juga."),
            utt("customer", "Nah itu berguna nggak sih buat harian? Saya kerja di Jakarta, macet terus."),
            utt("sales",    "Sangat berguna Pak terutama waktu macet panjang, otomatis bantu jaga jarak."),
            utt("customer", "Rumah saya di Bekasi, pulang pergi tiap hari sekitar 40 km."),
            utt("sales",    "Wah cukup jauh juga Pak. Biasanya berangkat jam berapa?"),
            utt("customer", "Jam 6 pagi, pulang jam 8 malam. Capek di jalan terus."),
        ],
    ),
}


# ── Shared prompt builder (copied from ai.py logic) ─────────────────
def _build_conversation_text(context: ConversationContext) -> str:
    lines = []
    for u in context.utterances[-10:]:
        label = "Sales" if u.speaker == "sales" else "Customer"
        lines.append(f"{label}: {u.text}")
    return "\n".join(lines)


def _build_cars_summary() -> str:
    cars = get_all_cars()
    lines = []
    for c in cars:
        lines.append(
            f"- {c.id}: {c.brand} {c.model} {c.variant}, "
            f"Rp {c.price_otr_jakarta:,}, {c.seats} kursi, tipe {c.type}"
        )
    return "\n".join(lines)


def _build_prompt(context: ConversationContext) -> str:
    conversation = _build_conversation_text(context)
    cars_summary = _build_cars_summary()

    customer_utterances = [u for u in context.utterances if u.speaker == "customer"]
    car_keywords = {"mobil", "beli", "kredit", "harga", "dp", "angsuran", "cicilan",
                    "xpander", "pajero", "xforce", "outlander", "eclipse", "colt"}
    customer_led = any(
        any(kw in u.text.lower() for kw in car_keywords)
        for u in customer_utterances[:3]
    )

    if customer_led:
        stage_instruction = (
            "Customer sudah menyebut minat atau kebutuhan lebih dulu — "
            "langsung gali kebutuhan konkret: untuk siapa, pemakaian harian, medan jalan, budget. "
            "Boleh mulai mengarahkan ke produk jika kebutuhan sudah cukup jelas."
        )
    else:
        stage_instruction = (
            "Customer belum menyebut kebutuhan mobil — sales harus BASA-BASI dulu. "
            "Tanya kehidupan sehari-hari, rutinitas, keluarga, pekerjaan. "
            "JANGAN sebut produk atau fitur dulu — bangun kepercayaan terlebih dahulu."
        )

    return f"""Kamu adalah AI coach untuk sales di showroom Mitsubishi Indonesia.
Tugasmu: baca percakapan, lalu bantu sales dengan saran pertanyaan yang tepat dan rekomendasi produk yang relevan.

PRINSIP WAJIB:
1. Customer TIDAK TAHU semua fitur mobil — JANGAN PERNAH sarankan pertanyaan tentang fitur teknis ke customer.
2. Sales bertugas menggali KEHIDUPAN & KEBIASAAN customer melalui obrolan natural.
3. KAMU yang menerjemahkan cerita customer ke kebutuhan lalu ke produk — bukan customer yang pilih fitur sendiri.
4. Pertanyaan yang disarankan harus terasa seperti obrolan biasa, bukan kuesioner.

ARAH PERCAKAPAN SAAT INI:
{stage_instruction}

Percakapan:
{conversation}

Daftar mobil tersedia (gunakan id persis dari sini):
{cars_summary}

Berikan output JSON berikut (tanpa markdown, tanpa komentar):
{{
  "hint_text": "insight singkat situasi customer dari perspektif sales (max 12 kata)",
  "suggested_question": "satu pertanyaan natural tentang kehidupan/aktivitas, BUKAN tentang fitur (max 12 kata)",
  "probe_topics": ["Aspek kehidupan/situasi belum tergali, max 5 kata"],
  "detected_needs": ["kebutuhan yang TERSIRAT dari cerita customer"],
  "recommended_car_ids": [],
  "recommendation_reason": "alasan berdasarkan kebutuhan/kebiasaan customer"
}}"""


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


# ── Model callers ────────────────────────────────────────────────────
async def call_gemma(prompt: str) -> tuple[dict, float, int, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.featherless_api_key,
        base_url=settings.featherless_base_url,
    )
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model="google/gemma-4-E4B-it",
        max_tokens=350,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0
    raw = resp.choices[0].message.content.strip()
    data = _extract_json(raw)
    return data, latency, resp.usage.prompt_tokens, resp.usage.completion_tokens


async def call_gpt4o_mini(prompt: str) -> tuple[dict, float, int, int]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=350,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Kamu adalah AI coach sales mobil. Selalu jawab dengan JSON valid sesuai format yang diminta."},
            {"role": "user",   "content": prompt},
        ],
    )
    latency = time.perf_counter() - t0
    raw = resp.choices[0].message.content.strip()
    data = json.loads(raw)
    return data, latency, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ── Print helpers ────────────────────────────────────────────────────
def _wrap(text: str, width: int = 52) -> str:
    return "\n      ".join(textwrap.wrap(str(text), width))


def print_result(label: str, data: dict, latency: float, in_tok: int, out_tok: int):
    print(f"  [{label}]  {latency:.2f}s  |  in={in_tok} out={out_tok} tokens")
    print(f"  hint       : {_wrap(data.get('hint_text', '—'))}")
    print(f"  question   : {_wrap(data.get('suggested_question', '—'))}")
    print(f"  probe      : {data.get('probe_topics', [])}")
    print(f"  needs      : {data.get('detected_needs', [])}")
    print(f"  cars       : {data.get('recommended_car_ids', [])}")
    print(f"  reason     : {_wrap(data.get('recommendation_reason', '—'))}")


# ── Main ─────────────────────────────────────────────────────────────
async def main():
    print("=" * 70)
    print("  AI Analysis Comparison: Gemma-4-E4B-it  vs  GPT-4o-mini")
    print("=" * 70)

    for scenario_name, context in SCENARIOS.items():
        print(f"\n{'─' * 70}")
        print(f"  SCENARIO: {scenario_name}")
        print("─" * 70)

        prompt = _build_prompt(context)

        # Run both models concurrently
        gemma_task    = asyncio.create_task(call_gemma(prompt))
        gpt_mini_task = asyncio.create_task(call_gpt4o_mini(prompt))

        results = await asyncio.gather(gemma_task, gpt_mini_task, return_exceptions=True)

        # Gemma result
        if isinstance(results[0], Exception):
            print(f"\n  [Gemma]  ERROR: {results[0]}")
        else:
            data, latency, in_tok, out_tok = results[0]
            print()
            print_result("Gemma-4-E4B-it ", data, latency, in_tok, out_tok)

        print()

        # GPT-4o-mini result
        if isinstance(results[1], Exception):
            print(f"  [GPT-4o-mini]  ERROR: {results[1]}")
        else:
            data, latency, in_tok, out_tok = results[1]
            print_result("GPT-4o-mini    ", data, latency, in_tok, out_tok)

    print(f"\n{'=' * 70}")
    print("  Done. Compare hint quality, question naturalness, and JSON reliability.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
