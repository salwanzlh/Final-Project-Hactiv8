"""
AI Service — menganalisis konteks percakapan dan menghasilkan:
  1. Hint / saran pertanyaan untuk sales
  2. Rekomendasi mobil yang cocok
"""
import re
import json
import logging
import asyncio
from backend.config import settings
from backend.models.schemas import ConversationContext, AiHintPayload, CarRecommendPayload
from backend.db.car_db import get_all_cars, get_car_by_id
from backend.services.langfuse_client import langfuse
from backend.services.metrics import (
    llm_requests, rekomendasi_total, error_total,
    llm_latency, token_per_request, track_latency,
)
from backend.services.rag import (
    search_similar_customers, format_rag_context,
    search_conversation_patterns, format_pattern_context,
)
from backend.services.elicitation import (
    compute_missing_dimensions,
    get_next_question,
    build_elicitation_prompt_section,
    is_deflection,
    question_to_dimension,
)

logger = logging.getLogger(__name__)

# Sinyal kuat → rekomendasi awal tanpa perlu semua dimensi terpetakan
QUICK_SIGNAL_MAP = [
    {
        "patterns": [r"\b(mudik|pulang kampung|luar kota|perjalanan jauh|jalan jauh)\b"],
        "car_ids": ["xpander-ultimate-cvt", "destinator-ultimate-cvt"],
        "label": "Sering mudik / perjalanan jauh",
    },
    {
        "patterns": [r"\b(bbm|bensin|irit|hemat|boros)\b"],
        "car_ids": ["xpander-ultimate-cvt", "xforce-ultimate-diamond-sense"],
        "label": "Prioritas hemat BBM",
    },
    {
        "patterns": [r"\b(7 (orang|penumpang|kursi)|keluarga besar|rame.rame|bertujuh|berenam)\b"],
        "car_ids": ["xpander-ultimate-cvt", "xpander-cross"],
        "label": "Butuh kapasitas besar",
    },
    {
        "patterns": [r"\b(offroad|off.road|jalanan (rusak|jelek|berlumpur)|medan (berat|sulit|kasar))\b"],
        "car_ids": ["pajero-sport-dakar-ultimate-4x4-at", "xpander-cross"],
        "label": "Butuh medan berat / offroad",
    },
    {
        "patterns": [r"\b(angkut|muatan|niaga|pickup|proyek|tambang)\b"],
        "car_ids": ["triton-ultimate-4x4-at"],
        "label": "Kebutuhan angkut / bisnis",
    },
    {
        "patterns": [r"\b(mewah|premium|prestige|bergengsi)\b"],
        "car_ids": ["pajero-sport-dakar-ultimate-4x4-at", "xforce-ultimate-diamond-sense"],
        "label": "Preferensi premium",
    },
]


def _detect_quick_signals(text: str) -> tuple[list[str], list[str]]:
    """
    Deteksi sinyal kuat yang cukup untuk rekomendasi awal meski profil belum lengkap.
    Return: (car_ids unik, labels yang match)
    """
    car_ids: list[str] = []
    labels: list[str] = []
    for entry in QUICK_SIGNAL_MAP:
        if any(re.search(p, text, re.IGNORECASE) for p in entry["patterns"]):
            labels.append(entry["label"])
            for cid in entry["car_ids"]:
                if cid not in car_ids:
                    car_ids.append(cid)
    return car_ids, labels


# Nama model yang disebut customer → tampilkan spec langsung ke sales
CAR_MENTION_MAP = [
    {
        "patterns": [r"\b(xpander cross)\b"],
        "car_ids": ["xpander-cross"],
        "label": "Xpander Cross",
    },
    {
        "patterns": [r"\bxpander(?!\s+cross)\b"],
        "car_ids": ["xpander-ultimate-cvt", "xpander-cross"],
        "label": "Xpander",
    },
    {
        "patterns": [r"\b(xforce|x force)\b"],
        "car_ids": ["xforce-ultimate-diamond-sense"],
        "label": "Xforce",
    },
    {
        "patterns": [r"\b(pajero)\b"],
        "car_ids": ["pajero-sport-dakar-ultimate-4x4-at"],
        "label": "Pajero Sport",
    },
    {
        "patterns": [r"\b(triton)\b"],
        "car_ids": ["triton-ultimate-4x4-at"],
        "label": "Triton",
    },
    {
        "patterns": [r"\b(eclipse)\b"],
        "car_ids": ["eclipse-cross-ultimate"],
        "label": "Eclipse Cross",
    },
    {
        "patterns": [r"\b(destinator)\b"],
        "car_ids": ["destinator-ultimate-cvt"],
        "label": "Destinator",
    },
    {
        "patterns": [r"\b(outlander)\b"],
        "car_ids": ["outlander-sport-px"],
        "label": "Outlander Sport",
    },
]


def _detect_car_mentions(text: str) -> tuple[list[str], list[str]]:
    """
    Deteksi nama model yang disebut customer — tampilkan spec ke sales segera.
    Pola lebih spesifik (xpander cross) dicek duluan sebelum yang umum (xpander).
    Return: (car_ids unik, model labels)
    """
    car_ids: list[str] = []
    labels: list[str] = []
    for entry in CAR_MENTION_MAP:
        if any(re.search(p, text, re.IGNORECASE) for p in entry["patterns"]):
            labels.append(entry["label"])
            for cid in entry["car_ids"]:
                if cid not in car_ids:
                    car_ids.append(cid)
    return car_ids, labels


SYSTEM_PROMPT = """Kamu adalah AI coach untuk sales showroom Mitsubishi Indonesia.
Baca percakapan → tentukan tahap → beri saran pertanyaan dan rekomendasi produk.

━━━ PRINSIP ━━━
- Gali KEHIDUPAN & KEBIASAAN customer, bukan preferensi produk atau fitur teknis.
- Tanya hal yang bisa dijawab orang yang belum pernah beli mobil sekalipun.
- KAMU yang terjemahkan cerita customer → kebutuhan → produk.

━━━ TAHAP (nilai dari ISI, bukan jumlah kalimat) ━━━
PEMBUKA      — Baru salam/basa-basi. Sambut hangat, tanya sangat ringan.
RAPPORT      — Ada obrolan ringan tapi belum ada fakta konkret. Cairkan suasana,
               jangan singgung kebutuhan apapun. Tanya hal paling ringan: dari mana,
               sudah lama cari-cari, ini ke sini sendiri atau bareng siapa.
EKSPLORASI   — Minimal 1 fakta konkret sudah muncul (bukan basa-basi) — contoh:
               customer cerita tinggal di mana, kerjanya apa, atau kendaraan sekarang.
               Baru boleh gali rutinitas dan keseharian.
PENDALAMAN   — Ada fakta keluarga + salah satu: rutinitas/finansial/kendaraan lama.
               Isi celah penting yang tersisa.
SINYAL_KUAT  — Ada 1+ sinyal spesifik yang sangat jelas sebelum profil lengkap:
               "mudik/sering luar kota", "hemat BBM/irit", "7 penumpang/keluarga besar",
               "offroad/medan berat", "angkut barang". Rekomendasikan 1-2 mobil sementara,
               sambil terus menggali dimensi yang belum terungkap.
REKOMENDASI  — Ketiga ini sudah terpetakan dari cerita customer:
               (1) siapa pakai & berapa orang, (2) rutinitas perjalanan, (3) gambaran finansial.
               Berikan rekomendasi lengkap dan final.
WINDOW_SHOPPER — Customer bilang "lihat-lihat/iseng/belum pasti". Dampingi santai,
               jangan push. recommended_car_ids WAJIB [].

━━━ MENTION PRODUK (PRIORITAS TERTINGGI) ━━━
Jika customer menyebut nama model secara eksplisit — mis. "xpander", "xforce",
"pajero", "triton", "eclipse", "destinator", "outlander":
→ Masukkan model itu ke recommended_car_ids SEGERA, di tahap apapun kecuali WINDOW_SHOPPER.
→ hint_text: "Customer sebut [model] — tampilkan spec."
→ Tetap tanyakan MENGAPA tertarik, bukan fitur teknis apa yang diinginkan.
   Contoh: "Sudah pernah coba atau baru lihat-lihat Xpander?"

━━━ DEFLEKSI CUSTOMER ━━━
Jika customer menghindari topik — jawaban samar ("nanti aja", "terserah", "ga penting"),
sangat pendek tanpa info, atau ganti topik setelah ditanya sesuatu personal:
→ Isi "blocked_dimension" dengan nama dimensi yang dihindari:
   "keluarga" | "mobilitas" | "finansial" | "urgency" | "" (kosong jika tidak ada defleksi)
→ JANGAN tanyakan dimensi itu lagi di suggested_question atau probe_topics.
→ Alihkan ke dimensi lain yang belum tergali, atau topik netral.
Dimensi yang dihindari customer akan dikirim kembali di prompt berikutnya —
INGAT dan PATUHI sepanjang sesi, sampai customer sendiri yang menyebut topik itu.

━━━ LARANGAN ━━━
Jangan tanya: fitur, spesifikasi, transmisi, suspensi, ground clearance, torsi,
cc, ADAS, ABS, CVT, sunroof, atau istilah teknis apapun.

━━━ REKOMENDASI ━━━
Dua jalur untuk mengisi recommended_car_ids:

JALUR 1 — SINYAL_KUAT (1 sinyal kuat sudah cukup):
  • "mudik / sering luar kota / perjalanan jauh"  → xpander-ultimate-cvt, destinator-ultimate-cvt
  • "hemat BBM / irit / boros bensin"              → xpander-ultimate-cvt, xforce-ultimate-diamond-sense
  • "7 penumpang / keluarga besar / rame-rame"     → xpander-ultimate-cvt, xpander-cross
  • "offroad / jalanan rusak / medan berat"        → pajero-sport-dakar-ultimate-4x4-at, xpander-cross
  • "angkut barang / niaga / pickup"               → triton-ultimate-4x4-at
  • "mewah / premium / bergengsi"                  → pajero-sport-dakar-ultimate-4x4-at, xforce-ultimate-diamond-sense
  Gunakan tahap SINYAL_KUAT, isi 1-2 car_ids, lanjutkan menggali sisanya.

JALUR 2 — REKOMENDASI PENUH (ketiga dimensi terpetakan):
  Semua tiga sudah diketahui dari cerita customer → tahap REKOMENDASI, isi lengkap.

recommended_car_ids WAJIB [] HANYA saat: PEMBUKA, RAPPORT, EKSPLORASI, PENDALAMAN, WINDOW_SHOPPER.

━━━ GROUNDING — suggested_question ━━━
Jika "Pola percakapan historis" tersedia dalam konteks:
• PRIORITAS UTAMA: parafrase "effective_next_question" dari pola dengan outcome
  "customer_engaged" atau "sale_progressed" yang paling cocok dengan tahap saat ini.
• ADAPTASI ke customer konkret — boleh ganti kata, tapi pertahankan inti pertanyaan.
• HANYA buat pertanyaan baru dari nol jika tidak ada pola yang relevan.
• Isi "question_source": "pattern" jika berangkat dari pola RAG, "generated" jika buat sendiri.

━━━ OUTPUT ━━━
JSON tanpa markdown:
{
  "tahap": "PEMBUKA|RAPPORT|EKSPLORASI|PENDALAMAN|SINYAL_KUAT|REKOMENDASI|WINDOW_SHOPPER",
  "hint_text": "insight singkat situasi customer, max 12 kata",
  "suggested_question": "satu pertanyaan natural tentang kehidupan/aktivitas, max 12 kata",
  "probe_topics": ["Gali X", "Tanyakan Y"],
  "detected_needs": ["kebutuhan tersirat dari cerita customer"],
  "recommended_car_ids": [],
  "recommendation_reason": "alasan berdasarkan cerita customer, bukan fitur",
  "blocked_dimension": "",
  "question_source": "pattern|generated"
}"""


def _extract_json(raw: str) -> dict:
    """Ekstrak JSON dari response LLM yang mungkin dibungkus markdown code block."""
    text = raw.strip()
    # Hapus markdown code block kalau ada (```json ... ``` atau ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # buang baris pertama (```json atau ```)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Ambil substring dari { pertama ke } terakhir
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


def _build_conversation_text(context: ConversationContext) -> str:
    lines = []
    for u in context.utterances[-10:]:  # Ambil 10 utterance terakhir
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


async def analyze_conversation(context: ConversationContext) -> tuple[AiHintPayload, CarRecommendPayload]:
    """
    Analisis percakapan → hasilkan hint + rekomendasi.
    Mode demo: return dummy yang relevan berdasarkan keyword sederhana.
    Mode production: kirim ke GPT.
    """
    if settings.app_mode == "demo":
        return await _demo_analyze(context)

    return await _openai_analyze(context)


async def _demo_analyze(context: ConversationContext) -> tuple[AiHintPayload, CarRecommendPayload]:
    """Analisis keyword sederhana tanpa API eksternal."""
    await asyncio.sleep(0.3)

    full_text = " ".join(u.text.lower() for u in context.utterances)

    # Enrich detected_needs with RAG if available
    rag_customers = await search_similar_customers(full_text[:2000], top_k=2)

    # Deteksi sinyal kuat + nama model yang disebut customer
    quick_ids, quick_labels = _detect_quick_signals(full_text)
    mention_ids, mention_labels = _detect_car_mentions(full_text)

    # Gabung: mention model selalu ditampilkan (prioritas), lalu sinyal kebutuhan
    needs = [f"Customer sebut: {m}" for m in mention_labels] + list(quick_labels)
    recommended_ids: list[str] = []
    for cid in mention_ids + quick_ids:
        if cid not in recommended_ids:
            recommended_ids.append(cid)

    if any(w in full_text for w in ["keluarga", "anak", "6 orang", "5 orang"]):
        if "Butuh kapasitas besar" not in needs:
            needs.append("Keluarga besar — butuh kapasitas kursi banyak")
        if "xpander-ultimate-cvt" not in recommended_ids:
            recommended_ids.append("xpander-ultimate-cvt")
        if "xpander-cross" not in recommended_ids:
            recommended_ids.append("xpander-cross")

    if any(w in full_text for w in ["luar kota", "jalanan jelek", "medan berat", "offroad"]):
        if "Butuh medan berat / offroad" not in needs:
            needs.append("Sering luar kota / medan bervariasi")
        if "xpander-cross" not in recommended_ids:
            recommended_ids.append("xpander-cross")
        if "pajero-sport-dakar-ultimate-4x4-at" not in recommended_ids:
            recommended_ids.append("pajero-sport-dakar-ultimate-4x4-at")

    if any(w in full_text for w in ["250", "300", "350", "budget"]):
        needs.append("Budget 250–350 juta")
        if "xpander-ultimate-cvt" not in recommended_ids:
            recommended_ids.append("xpander-ultimate-cvt")

    if any(w in full_text for w in ["mewah", "premium", "nyaman", "fitur"]):
        if "Preferensi premium" not in needs:
            needs.append("Preferensi fitur premium")
        if "xforce-ultimate-diamond-sense" not in recommended_ids:
            recommended_ids.append("xforce-ultimate-diamond-sense")

    # Enrich needs from RAG matches
    for rc in rag_customers:
        faktor = rc.get("faktor_utama", "")
        if faktor:
            needs.append(f"[Histori serupa] {faktor[:80]}")

    # Default jika belum ada kebutuhan terdeteksi
    if not needs:
        needs = ["Kebutuhan belum teridentifikasi"]
        recommended_ids = []

    # Deteksi defleksi — cek apakah customer baru menghindar dari topik terakhir
    new_blocked_dim: str = ""
    if context.asked_questions:
        customer_uttrs = [u for u in context.utterances if u.speaker == "customer"]
        if customer_uttrs and is_deflection(customer_uttrs[-1].text):
            dim = question_to_dimension(context.asked_questions[-1])
            if dim and dim not in context.blocked_dimensions:
                new_blocked_dim = dim
                logger.info(f"[AI] Defleksi terdeteksi — dimensi '{dim}' di-block")

    # Tentukan dimensi yang belum tergali, kecuali yang customer hindari
    missing = compute_missing_dimensions(full_text, context.blocked_dimensions)
    n = len(context.utterances)

    if mention_ids:
        # Customer sebut nama model — tampilkan spec langsung, apapun tahapnya
        model_str = " & ".join(mention_labels[:2])
        q_dict = get_next_question(missing, context.asked_questions)
        if q_dict:
            hint = f"Customer sebut {model_str} — tampilkan spec. Gali alasannya."
            question = q_dict["question"]
        else:
            hint = f"Customer sebut {model_str} — tampilkan spec."
            question = "Sudah pernah coba atau baru lihat-lihat?"
        tahap = "SINYAL_KUAT"
    elif n <= 2 and len(missing) == 4:
        hint = "Customer baru tiba — sambut hangat, jangan buru-buru gali kebutuhan."
        question = "Silakan mampir dulu Pak/Bu, bisa saya temenin lihat-lihat?"
        tahap = "PEMBUKA"
        recommended_ids = []
    elif quick_ids:
        # Ada sinyal kuat — rekomendasikan langsung, tapi terus gali sisanya
        q_dict = get_next_question(missing, context.asked_questions)
        if q_dict:
            reveals = q_dict.get("reveals", [])
            hint = f"Sinyal kuat: {', '.join(quick_labels[:2])}. Gali {' & '.join(reveals)}."
            question = q_dict["question"]
        else:
            hint = f"Sinyal kuat: {', '.join(quick_labels[:2])}. Profil hampir lengkap."
            question = "Boleh saya tunjukkan pilihan yang paling cocok untuk situasi Bapak/Ibu?"
        tahap = "SINYAL_KUAT"
    else:
        q_dict = get_next_question(missing, context.asked_questions)
        if q_dict:
            reveals = q_dict.get("reveals", [])
            hint = f"Gali {' & '.join(reveals)} lebih dalam."
            question = q_dict["question"]
            tahap = "EKSPLORASI" if len(missing) > 2 else "PENDALAMAN"
        else:
            hint = "Profil customer cukup tergambar, siap arahkan ke rekomendasi."
            question = "Boleh saya tunjukkan dua pilihan yang paling cocok untuk situasi Bapak/Ibu?"
            tahap = "REKOMENDASI"

    hint_payload = AiHintPayload(
        hint_text=hint,
        suggested_question=question,
        detected_needs=needs,
        tahap=tahap,
        blocked_dimension=new_blocked_dim,
    )

    cars = [get_car_by_id(cid) for cid in recommended_ids if get_car_by_id(cid)]
    for car in cars:
        rekomendasi_total.labels(merek_mobil=car.brand).inc()
    reason = f"Berdasarkan: {', '.join(needs)}"
    car_payload = CarRecommendPayload(cars=cars, reason=reason)

    return hint_payload, car_payload


async def _openai_analyze(context: ConversationContext) -> tuple[AiHintPayload, CarRecommendPayload]:
    """Analisis menggunakan OpenAI GPT-4.1 — dipakai di production mode."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
    )

    conversation = _build_conversation_text(context)
    cars_summary = _build_cars_summary()

    # RAG: parallel retrieve — customer profiles + conversation patterns
    query = " ".join(u.text for u in context.utterances[-6:])
    rag_customers, rag_patterns = await asyncio.gather(
        search_similar_customers(query, top_k=3),
        search_conversation_patterns(query, top_k=3),
    )
    rag_section         = format_rag_context(rag_customers)
    rag_pattern_section = format_pattern_context(rag_patterns)

    asked = context.asked_questions
    asked_section = (
        "Pertanyaan yang SUDAH pernah ditanyakan (JANGAN ulangi atau parafrase):\n"
        + "\n".join(f"- {q}" for q in asked)
        if asked else ""
    )

    full_text_for_elicitation = " ".join(u.text for u in context.utterances)
    missing = compute_missing_dimensions(full_text_for_elicitation, context.blocked_dimensions)
    elicitation_section = build_elicitation_prompt_section(missing, context.blocked_dimensions)

    user_content = f"""Percakapan:
{conversation}

Daftar mobil tersedia (gunakan id persis dari sini):
{cars_summary}

{rag_section}

{rag_pattern_section}

{asked_section}

{elicitation_section}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    trace = langfuse.trace(name="analyze_conversation", input={"utterances": len(context.utterances)})
    generation = trace.generation(
        name="openai_llm",
        model="gpt-4.1",
        input=user_content,
    )

    try:
        async with track_latency(llm_latency):
            response = await client.chat.completions.create(
                model="gpt-4.1",
                max_tokens=600,
                temperature=0.4,
                messages=messages,
            )
        llm_requests.labels(status="sukses").inc()
    except Exception as e:
        llm_requests.labels(status="gagal").inc()
        error_total.labels(komponen="llm", tipe_error=type(e).__name__).inc()
        raise

    token_per_request.labels(tipe="input").observe(response.usage.prompt_tokens)
    token_per_request.labels(tipe="output").observe(response.usage.completion_tokens)

    raw = response.choices[0].message.content.strip()
    generation.end(
        output=raw,
        usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
    )
    logger.debug(f"[AI] Raw LLM response: {raw[:500]}")
    data = _extract_json(raw)
    tahap = data.get("tahap", "")
    logger.info(
        f"[AI] tahap={tahap} | "
        f"suggested_question='{data.get('suggested_question', '')[:80]}' | "
        f"car_ids={data.get('recommended_car_ids', [])}"
    )

    question_source = data.get("question_source", "generated")
    logger.info(f"[AI] question_source={question_source}")

    hint_payload = AiHintPayload(
        hint_text=data.get("hint_text", ""),
        suggested_question=data.get("suggested_question", ""),
        probe_topics=data.get("probe_topics", []),
        detected_needs=data.get("detected_needs", []),
        tahap=tahap,
        blocked_dimension=data.get("blocked_dimension", ""),
        question_source=question_source,
    )
    car_ids = data.get("recommended_car_ids", [])

    # Reinforcement — inject car mentions detected by regex so LLM can't miss them
    full_text_all = " ".join(u.text for u in context.utterances)
    mention_ids_prod, _ = _detect_car_mentions(full_text_all)
    for mid in mention_ids_prod:
        if mid not in car_ids:
            car_ids.insert(0, mid)

    cars = [get_car_by_id(cid) for cid in car_ids if get_car_by_id(cid)]
    for car in cars:
        rekomendasi_total.labels(merek_mobil=car.brand).inc()
    car_payload = CarRecommendPayload(
        cars=cars,
        reason=data.get("recommendation_reason", ""),
    )
    trace.update(output={"hint": data.get("hint_text", ""), "tahap": tahap, "recommended": car_ids})
    return hint_payload, car_payload


async def _claude_analyze(context: ConversationContext) -> tuple[AiHintPayload, CarRecommendPayload]:
    """Analisis menggunakan Claude API — aktifkan di production."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    conversation = _build_conversation_text(context)
    cars_summary = _build_cars_summary()

    prompt = f"""Kamu adalah AI asisten untuk sales showroom mobil.

Percakapan yang berlangsung:
{conversation}

Daftar mobil tersedia:
{cars_summary}

Berikan analisis dalam format JSON berikut (tanpa markdown):
{{
  "hint_text": "insight singkat untuk sales (max 20 kata)",
  "suggested_question": "pertanyaan lanjutan yang natural untuk sales tanyakan",
  "detected_needs": ["kebutuhan 1", "kebutuhan 2"],
  "recommended_car_ids": ["id-mobil-1", "id-mobil-2"],
  "recommendation_reason": "alasan singkat rekomendasi"
}}"""

    trace = langfuse.trace(name="analyze_conversation", input={"utterances": len(context.utterances)})
    generation = trace.generation(name="claude_llm", model="claude-sonnet-4-20250514", input=prompt)

    try:
        async with track_latency(llm_latency):
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
        llm_requests.labels(status="sukses").inc()
    except Exception as e:
        llm_requests.labels(status="gagal").inc()
        error_total.labels(komponen="llm", tipe_error=type(e).__name__).inc()
        raise

    token_per_request.labels(tipe="input").observe(response.usage.input_tokens)
    token_per_request.labels(tipe="output").observe(response.usage.output_tokens)

    raw = response.content[0].text
    generation.end(
        output=raw,
        usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    )
    data = json.loads(raw)

    hint_payload = AiHintPayload(
        hint_text=data["hint_text"],
        suggested_question=data["suggested_question"],
        detected_needs=data["detected_needs"],
    )
    cars = [get_car_by_id(cid) for cid in data["recommended_car_ids"] if get_car_by_id(cid)]
    for car in cars:
        rekomendasi_total.labels(merek_mobil=car.brand).inc()
    car_payload = CarRecommendPayload(
        cars=cars,
        reason=data["recommendation_reason"],
    )
    trace.update(output={"hint": data["hint_text"], "recommended": data["recommended_car_ids"]})
    return hint_payload, car_payload
