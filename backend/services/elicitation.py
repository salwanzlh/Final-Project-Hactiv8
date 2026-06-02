"""
backend/services/elicitation.py
=================================
Indirect elicitation strategy untuk sales showroom Mitsubishi.

Filosofi:
  Jangan tanya langsung tentang hal sensitif (budget, keluarga, finansial).
  Tanya tentang KEHIDUPAN — jawaban customer akan naturally reveal konteks itu.

  Teknik yang dipakai:
  - Situational questions  — reveal keluarga & rutinitas
  - Behavioral questions   — reveal mobilitas & kebiasaan
  - Comparative questions  — reveal finansial & ekspektasi
  - Future-pacing          — reveal urgency & kesiapan
"""

import re


# --- Signal patterns yang diperluas ----------------------------------------
# Bukan hanya keyword langsung, tapi juga sinyal tidak langsung
# yang muncul dari jawaban umpan pertanyaan

SIGNAL_PATTERNS = {

    # -- Keluarga -------------------------------------------------------------
    "keluarga": {
        "direct": [
            r"\b(anak|istri|suami|keluarga|orang tua|mertua|cucu|adik|kakak)\b",
            r"\b(\d+\s*(orang|anak|anggota|penumpang))\b",
        ],
        "indirect": [
            r"\b(bareng|sama-sama|rame-rame|sekeluarga|berlima|berenam|bertujuh)\b",
            r"\b(antar (anak|jemput)|les|sekolah|arisan|kondangan)\b",
            r"\b(kontrakan|kost|rumah (sendiri|orang tua|mertua))\b",
        ],
        "weight": 15,
    },

    # -- Mobilitas ------------------------------------------------------------
    "mobilitas": {
        "direct": [
            r"\b(luar kota|dalam kota|mudik|jalan tol|jalanan jelek|offroad)\b",
            r"\b(km|kilometer|jarak|menit|jam)\b",
            r"\b(tiap hari|setiap minggu|sebulan sekali)\b",
        ],
        "indirect": [
            r"\b(naik (tol|kereta|busway)|macet|bypass|ring road)\b",
            r"\b(bekasi|bogor|depok|tangerang|serpong|cikarang|karawang|bandung)\b",
            r"\b(\d+\s*(km|menit|jam)\s*(dari|ke|perjalanan))\b",
            r"\b(proyek|klien|supplier|cabang|meeting luar)\b",
            r"\b(puncak|pantai|kampung|hometown|pulang kampung)\b",
            r"\b(muat|mepet|sempit|penuh|ngangkut|muatan)\b",
        ],
        "weight": 20,
    },

    # -- Finansial — paling sensitif, paling indirect -------------------------
    "finansial": {
        "direct": [
            r"\b(\d{2,3})\s*(juta|jt)\b",
            r"\b(dp|down payment|uang muka)\b",
            r"\b(cicilan|angsuran|kredit|per bulan)\b",
        ],
        "indirect": [
            r"\b(fortuner|pajero|alphard|crv|hrv|xpander|innova|avanza|brio|agya|ayla)\b",
            r"\b(2-3 tahun|hampir (baru|lama)|baru (beli|ganti)|sudah (lama|tua))\b",
            r"\b(sudah nabung|sudah siapin|siap|lagi nunggu|bonus|thr|proyek selesai)\b",
            r"\b(sudah ke (dealer|showroom)|sudah (test drive|coba)|sudah (riset|baca))\b",
        ],
        "weight": 25,
    },

    # -- Urgency --------------------------------------------------------------
    "urgency": {
        "direct": [
            r"\b(bulan ini|tahun ini|minggu ini|secepatnya|urgent)\b",
            r"\b(mau beli|rencana beli|lagi cari|pengen ganti)\b",
        ],
        "indirect": [
            r"\b(sering (mogok|servis|rusak)|sudah (tua|butut)|banyak (masalah|trouble))\b",
            r"\b(istri (butuh|minta|perlu)|anak (minta|perlu|butuh)|kantor (butuh|perlu))\b",
            r"\b(sudah (lama|lama banget|dari dulu|dari tahun)|terlalu lama|akhirnya)\b",
            r"\b(nimbang-nimbang|nimbang nimbang|mikir-mikir|pikir-pikir lama)\b",
        ],
        "weight": 20,
    },
}

# Urutan prioritas: tanya yang paling tidak sensitif dulu
DIMENSION_PRIORITY = ["mobilitas", "keluarga", "urgency", "finansial"]


# --- Question bank — umpan per dimensi -------------------------------------

INDIRECT_QUESTIONS = {

    "keluarga": [
        {
            "question": "Biasanya kalau pergi weekend, sendirian atau ada yang ikut?",
            "reveals":  ["keluarga", "mobilitas"],
            "natural_because": "tanya rutinitas, jawaban naturally reveal siapa saja yang biasa ikut",
        },
        {
            "question": "Mobilnya nanti lebih sering buat aktivitas harian atau sesekali?",
            "reveals":  ["keluarga", "mobilitas"],
            "natural_because": "tanya use case, bukan jumlah orang",
        },
        {
            "question": "Tinggalnya sekarang sendiri atau masih sama keluarga?",
            "reveals":  ["keluarga", "finansial"],
            "natural_because": "casual question tentang situasi hidup",
        },
    ],

    "mobilitas": [
        {
            "question": "Kantornya jauh dari rumah atau masih deket-deket?",
            "reveals":  ["mobilitas", "rutinitas"],
            "natural_because": "obrolan ringan keseharian, tidak terasa seperti gali data",
        },
        {
            "question": "Hari-harinya lebih sering di kota atau sering keluar juga?",
            "reveals":  ["mobilitas"],
            "natural_because": "tanya lifestyle bukan kebutuhan mobil",
        },
        {
            "question": "Jalan yang biasa dilalui kondisinya gimana, mulus atau sering ketemu medan yang lumayan?",
            "reveals":  ["mobilitas"],
            "natural_because": "tanya kondisi jalan, bukan fitur mobil",
        },
        {
            "question": "Weekend biasanya ngapain, di rumah atau sering jalan?",
            "reveals":  ["mobilitas", "keluarga"],
            "natural_because": "tanya aktivitas, bukan kebutuhan kendaraan",
        },
    ],

    "finansial": [
        {
            "question": "Sekarang masih pakai kendaraan apa?",
            "reveals":  ["finansial", "urgency"],
            "natural_because": "kendaraan saat ini adalah proxy kuat untuk purchasing power",
            "note": "Kendaraan lama reveal segmen finansial tanpa perlu tanya angka",
        },
        {
            "question": "Sudah lama pakai yang sekarang?",
            "reveals":  ["finansial", "urgency"],
            "natural_because": "reveal kapan terakhir beli, proxy kemampuan finansial",
        },
        {
            "question": "Ini sudah lama nimbang-nimbang atau baru mulai lihat-lihat?",
            "reveals":  ["urgency", "finansial"],
            "natural_because": "reveal kesiapan finansial tanpa tanya angka",
        },
        {
            "question": "Sebelum ke sini sudah sempat lihat-lihat di tempat lain?",
            "reveals":  ["finansial", "urgency"],
            "natural_because": "customer yang sudah riset banyak biasanya lebih siap secara finansial",
        },
    ],

    "urgency": [
        {
            "question": "Kendaraan yang sekarang ada kendala atau memang lagi pengen upgrade saja?",
            "reveals":  ["urgency"],
            "natural_because": "reveal apakah ini kebutuhan mendesak atau pilihan",
        },
        {
            "question": "Ini buat siapa yang mainly pakai nantinya?",
            "reveals":  ["urgency", "keluarga"],
            "natural_because": "siapa yang pakai reveal tingkat kebutuhan dan urgency",
        },
    ],

    "universal": [
        {
            "question": "Sehari-hari aktivitasnya gimana Pak/Bu, lebih banyak di luar atau di dalam?",
            "reveals":  ["mobilitas", "keluarga", "urgency"],
        },
        {
            "question": "Selama ini kendaraannya gimana, sudah pas atau ada yang kurang?",
            "reveals":  ["urgency", "finansial"],
        },
        {
            "question": "Bapak/Ibu dari sini atau dari daerah lain?",
            "reveals":  ["mobilitas"],
            "natural_because": "obrolan paling ringan, reveal mobilitas jika dari luar kota",
        },
    ],
}


# --- Deflection detection ---------------------------------------------------

# Sinyal customer menghindar dari topik yang ditanyakan
DEFLECTION_PATTERNS = [
    r"\b(ga tau|gak tau|tidak tau|nggak tau)\b",
    r"\b(ga (penting|masalah|mau|perlu)|gak (penting|masalah|mau|perlu)|nggak (penting|masalah|mau|perlu))\b",
    r"\b(nanti (aja|saja|deh)|belakangan|lain kali|lain waktu)\b",
    r"\b(terserah (aja|saja|deh)?|bebas (aja|saja|deh)?|apapun (oke|ok|deh|boleh)|asal (ada|oke|ok))\b",
    r"\b(ga (mau|bisa) (cerita|bahas)|gak (mau|bisa) (cerita|bahas)|skip|lewat aja|next aja)\b",
    r"\b(private|sensitif|rahasia|ga usah|gausah|tidak perlu dibahas)\b",
    r"\b(udah|sudah|oke|ok|ya ya|iya iya)\s*$",  # jawaban sangat singkat menutup topik
]


def is_deflection(text: str) -> bool:
    """Apakah teks mengindikasikan customer menghindar dari topik."""
    cleaned = text.strip().lower()
    # Teks sangat pendek (≤3 kata) dan tidak mengandung info konkret juga dianggap defleksi
    words = cleaned.split()
    if len(words) <= 2 and not any(
        re.search(p, cleaned, re.IGNORECASE)
        for p in [r"\d", r"\b(ya|iya|betul|bener|oke|ok)\b"]
    ):
        return True
    return any(re.search(p, cleaned, re.IGNORECASE) for p in DEFLECTION_PATTERNS)


def question_to_dimension(question: str) -> str | None:
    """Temukan dimensi dari teks pertanyaan yang pernah disuggest ke sales."""
    for dim, questions in INDIRECT_QUESTIONS.items():
        if dim == "universal":
            continue
        for q_data in questions:
            if _similar(question, q_data["question"]):
                return dim
    return None


# --- compute_missing_dimensions ---------------------------------------------

def compute_missing_dimensions(
    conversation_text: str,
    blocked_dimensions: list[str] | None = None,
) -> list[str]:
    """
    Return list dimensi yang belum terungkap dari teks percakapan,
    diurutkan sesuai DIMENSION_PRIORITY (paling tidak sensitif dulu).
    Dimensi yang ada di blocked_dimensions dilewati — customer sudah menghindarinya.
    """
    found: set[str] = set()
    for dim, data in SIGNAL_PATTERNS.items():
        patterns = data.get("direct", []) + data.get("indirect", [])
        for pattern in patterns:
            if re.search(pattern, conversation_text, re.IGNORECASE):
                found.add(dim)
                break
    blocked = set(blocked_dimensions or [])
    return [d for d in DIMENSION_PRIORITY if d not in found and d not in blocked]


# --- get_next_question -------------------------------------------------------

def get_next_question(
    missing_dimensions: list[str],
    asked_questions: list[str],
) -> dict | None:
    """
    Pilih umpan pertanyaan terbaik berdasarkan dimensi yang belum tergali.
    Skip pertanyaan yang sudah pernah ditanyakan (word-overlap check).
    """
    for dim in missing_dimensions:
        for q in INDIRECT_QUESTIONS.get(dim, []):
            if not any(_similar(q["question"], asked) for asked in asked_questions):
                return q

    # Fallback universal
    for q in INDIRECT_QUESTIONS["universal"]:
        if not any(_similar(q["question"], asked) for asked in asked_questions):
            return q

    return None


def _similar(q1: str, q2: str) -> bool:
    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())
    if not words1 or not words2:
        return False
    overlap = len(words1 & words2) / min(len(words1), len(words2))
    return overlap > 0.4


# --- build_elicitation_prompt_section ----------------------------------------

def build_elicitation_prompt_section(
    missing_dimensions: list[str],
    blocked_dimensions: list[str] | None = None,
) -> str:
    """
    Bangun section prompt yang memberi tahu LLM dimensi mana yang masih kosong
    dan teknik umpan yang harus dipakai — bukan pertanyaan langsung.
    """
    if not missing_dimensions and not blocked_dimensions:
        return ""

    dim_notes = {
        "keluarga": (
            "Keluarga/kapasitas belum terungkap — tanya aktivitas bersama, "
            "bukan 'berapa orang'. Contoh: 'Weekend biasanya ngapain?'"
        ),
        "mobilitas": (
            "Rutinitas perjalanan belum terungkap — tanya keseharian, "
            "bukan 'sering keluar kota tidak'. Contoh: 'Kantornya jauh dari rumah?'"
        ),
        "finansial": (
            "Gambaran finansial belum terungkap — JANGAN tanya budget langsung. "
            "Tanya kendaraan sekarang atau sudah berapa lama nimbang-nimbang. "
            "Kendaraan lama adalah proxy kuat untuk purchasing power."
        ),
        "urgency": (
            "Urgensi belum terungkap — tanya kondisi kendaraan sekarang atau "
            "untuk siapa yang akan pakai, bukan 'kapan mau beli'."
        ),
    }

    lines = []

    if missing_dimensions:
        lines.append("DIMENSI YANG BELUM TERGALI — gunakan umpan, bukan pertanyaan langsung:")
        for dim in missing_dimensions:
            if dim in dim_notes:
                lines.append(f"* {dim_notes[dim]}")
        lines.append("")
        lines.append(
            "PRINSIP UMPAN: Tanya tentang kehidupan customer, bukan tentang kebutuhan mobil. "
            "Jawaban customer akan secara natural reveal apa yang dibutuhkan."
        )

    if blocked_dimensions:
        lines.append("")
        lines.append(
            "DIMENSI YANG CUSTOMER HINDARI — JANGAN tanyakan lagi sampai customer sendiri "
            "yang menyinggung topik ini:"
        )
        dim_names = {
            "keluarga": "keluarga / jumlah penumpang",
            "mobilitas": "rutinitas perjalanan / mobilitas",
            "finansial": "finansial / budget / kendaraan lama",
            "urgency": "urgensi pembelian / kapan beli",
        }
        for dim in blocked_dimensions:
            lines.append(f"* {dim_names.get(dim, dim)} — SKIP, ganti ke topik lain")

    return "\n".join(lines)
