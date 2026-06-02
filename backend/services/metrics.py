"""
Prometheus metrics untuk Showroom AI.

Diimpor oleh service lain untuk mencatat latency, counter, dan gauge.
"""
import time
import contextlib
from prometheus_client import Counter, Histogram, Gauge, Info

app_info = Info("showroom_ai", "Metadata aplikasi Showroom AI")
app_info.info({
    "version": "1.0.0",
    "model": "google/gemma-4-E4B-it",
    "environment": "production",
})

_latency_buckets = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

# ── Counters ──────────────────────────────────────────────────────────────────

llm_requests = Counter(
    "showroom_ai_llm_requests_total",
    "Total panggilan LLM",
    ["status"],          # sukses / gagal
)

stt_requests = Counter(
    "showroom_ai_stt_requests_total",
    "Total audio chunk yang diproses STT",
    ["status"],
)

tts_requests = Counter(
    "showroom_ai_tts_requests_total",
    "Total permintaan TTS synthesis",
    ["status"],
)

rekomendasi_total = Counter(
    "showroom_ai_rekomendasi_total",
    "Total rekomendasi mobil yang diberikan",
    ["merek_mobil"],     # Toyota / Mitsubishi / dll
)

error_total = Counter(
    "showroom_ai_error_total",
    "Total error per komponen",
    ["komponen", "tipe_error"],   # komponen: stt / llm / tts
)

sesi_selesai = Counter(
    "showroom_ai_sesi_selesai_total",
    "Total sesi percakapan yang selesai",
    ["outcome"],         # rekomendasi_diberikan / tidak_ada_minat
)

# ── Histograms ────────────────────────────────────────────────────────────────

stt_latency = Histogram(
    "showroom_ai_stt_latency_seconds",
    "Durasi STT per audio chunk",
    buckets=_latency_buckets,
)

llm_latency = Histogram(
    "showroom_ai_llm_latency_seconds",
    "Durasi LLM response",
    buckets=_latency_buckets,
)

tts_latency = Histogram(
    "showroom_ai_tts_latency_seconds",
    "Durasi TTS synthesis",
    buckets=_latency_buckets,
)

pipeline_latency = Histogram(
    "showroom_ai_pipeline_latency_seconds",
    "Durasi total satu siklus STT → LLM per audio chunk",
    buckets=_latency_buckets,
)

token_per_request = Histogram(
    "showroom_ai_token_per_request",
    "Token LLM per request",
    ["tipe"],            # input / output
    buckets=[50, 100, 200, 300, 500, 800, 1000, 1500],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

sesi_aktif = Gauge(
    "showroom_ai_sesi_aktif",
    "Jumlah sesi WebSocket yang sedang berlangsung",
)

# ── RAG / Pinecone ────────────────────────────────────────────────────────────

rag_latency = Histogram(
    "showroom_ai_rag_latency_seconds",
    "Durasi RAG retrieval dari Pinecone",
    ["namespace"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0],
)

rag_fallback_total = Counter(
    "showroom_ai_rag_fallback_total",
    "Jumlah kali RAG gagal dan kembali ke fallback",
    ["namespace"],
)

rag_results_returned = Histogram(
    "showroom_ai_rag_results_returned",
    "Jumlah dokumen yang dikembalikan per query RAG",
    ["namespace"],
    buckets=[0, 1, 2, 3, 5, 8],
)

# ── Speaker Classifier ────────────────────────────────────────────────────────

classifier_latency = Histogram(
    "showroom_ai_classifier_latency_seconds",
    "Durasi speaker classification per utterance",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

classifier_confidence = Histogram(
    "showroom_ai_classifier_confidence",
    "Confidence score speaker classifier",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

classifier_lowconf_total = Counter(
    "showroom_ai_classifier_lowconf_total",
    "Klasifikasi dengan confidence rendah (< 0.6)",
)

# ── WebSocket ─────────────────────────────────────────────────────────────────

ws_connect_total = Counter(
    "showroom_ai_ws_connect_total",
    "Total koneksi WebSocket yang masuk",
)

ws_disconnect_total = Counter(
    "showroom_ai_ws_disconnect_total",
    "Total WebSocket yang terputus",
    ["reason"],    # clean / unexpected
)

# ── Full Roundtrip ────────────────────────────────────────────────────────────

roundtrip_latency = Histogram(
    "showroom_ai_roundtrip_latency_seconds",
    "Durasi end-to-end audio_in sampai TTS_out terkirim",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
)

# ── Conversation Quality ──────────────────────────────────────────────────────

conversation_stage_total = Counter(
    "showroom_ai_conversation_stage_total",
    "Distribusi tahap percakapan yang dihasilkan AI",
    ["tahap"],
)

ai_busy_skip_total = Counter(
    "showroom_ai_ai_busy_skip_total",
    "Jumlah audio chunk yang di-skip karena AI masih sibuk",
)

silence_watchdog_total = Counter(
    "showroom_ai_silence_watchdog_total",
    "Jumlah kali silence watchdog terpicu (customer diam terlalu lama)",
)

deflection_total = Counter(
    "showroom_ai_deflection_total",
    "Customer menghindari pertanyaan per dimensi",
    ["dimension"],
)

filler_filtered_total = Counter(
    "showroom_ai_filler_filtered_total",
    "Utterance yang dibuang karena hanya berisi filler words",
)


# ── Utility ───────────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def track_latency(histogram: Histogram):
    """Async context manager: ukur dan catat durasi ke histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)
