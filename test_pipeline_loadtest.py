"""
Pipeline Load Test: STT + Speaker Classifier
=============================================
Membandingkan dua jalur pipeline end-to-end:
  A. OpenAI Whisper (Batch)          + classify_speaker (GPT-4o-mini)
  B. ElevenLabs Scribe Realtime (WS) + classify_speaker (GPT-4o-mini)

Metrik per run (diukur dengan time.perf_counter):
  stt_latency     : waktu dari audio dikirim → transkrip diterima
  cls_latency     : waktu GPT-4o-mini classify_speaker
                    (0.0 jika heuristic path: ≤2 kata)
  total_latency   : stt_latency + cls_latency
  prompt_tokens   : dari response.usage  (0 jika heuristic)
  completion_toks : dari response.usage  (0 jika heuristic)
  cost_usd        : estimasi biaya STT + classifier

Skenario audio (di-generate via OpenAI TTS):
  SHORT (~3 dtk)    : kalimat pendek berisi keyword otomotif
  LONG  (~9 dtk)    : kalimat panjang full-context showroom
  HEURISTIC (~1 dtk): 1-2 kata → menguji jalur bypass LLM

Usage:
    python test_pipeline_loadtest.py
    python test_pipeline_loadtest.py --runs 3
    python test_pipeline_loadtest.py --runs 5 --no-elevenlabs
    python test_pipeline_loadtest.py --runs 5 --no-whisper
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import time
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Literal

# ── Pricing constants (USD) ───────────────────────────────────────────────────
# Whisper-1: $0.006 / minute audio (rounded to nearest second)
WHISPER_COST_PER_SEC = 0.006 / 60

# ElevenLabs Scribe v2 Realtime: $0.40 / hour audio
EL_COST_PER_SEC = 0.40 / 3600

# GPT-4o-mini (2024-07-18): $0.15 / 1M input tokens, $0.60 / 1M output tokens
GPT4O_MINI_IN_PER_TOK  = 0.15  / 1_000_000
GPT4O_MINI_OUT_PER_TOK = 0.60  / 1_000_000

# Heuristic threshold (mirip production speaker_classifier.py)
_SHORT_TEXT_WORDS = 2

CLASSIFIER_MODEL = "gpt-4o-mini"   # production pakai gpt-4.1-mini; test pakai gpt-4o-mini

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    pipeline:      str          # "Whisper" | "ElevenLabs"
    scenario:      str          # "SHORT" | "LONG" | "HEURISTIC"
    run_index:     int
    transcript:    str
    speaker:       str
    confidence:    float
    stt_latency:   float        # detik
    cls_latency:   float        # detik (0 jika heuristic)
    cls_path:      str          # "heuristic" | "llm"
    prompt_tokens: int
    completion_tokens: int
    audio_duration: float       # detik audio yang dikirim
    cost_usd:      float
    error:         str = ""

    @property
    def total_latency(self) -> float:
        return self.stt_latency + self.cls_latency

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ScenarioAudio:
    label:     str
    text:      str
    mp3_bytes: bytes = field(default_factory=bytes, repr=False)
    pcm_bytes: bytes = field(default_factory=bytes, repr=False)

    @property
    def duration_secs(self) -> float:
        # PCM: 16kHz, 16-bit mono → 32000 bytes/detik
        return len(self.pcm_bytes) / 32_000 if self.pcm_bytes else 0.0


# ── Audio helpers ─────────────────────────────────────────────────────────────

async def generate_tts(text: str, api_key: str, voice: str = "nova") -> bytes:
    """Generate MP3 audio via OpenAI TTS (tts-1, kecepatan normal)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3",
        speed=1.0,
    )
    return resp.content


def to_pcm16k(audio_bytes: bytes) -> bytes:
    """Konversi MP3/WAV → raw PCM s16le 16kHz mono via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0",
         "-f", "s16le", "-ar", "16000", "-ac", "1",
         "pipe:1", "-loglevel", "quiet"],
        input=audio_bytes,
        capture_output=True,
    )
    return result.stdout


def silence_pcm(duration_secs: float = 0.8) -> bytes:
    """PCM silence murni — untuk flush VAD setelah audio selesai."""
    return bytes(int(16_000 * duration_secs) * 2)


# ── STT: Whisper Batch ────────────────────────────────────────────────────────

async def stt_whisper(mp3_bytes: bytes, api_key: str) -> tuple[str, float]:
    """
    Kirim MP3 ke Whisper-1 (batch).
    Returns (transcript, stt_latency_secs).
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    t0 = time.perf_counter()
    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.mp3", io.BytesIO(mp3_bytes), "audio/mpeg"),
        language="id",
        response_format="verbose_json",
        prompt=(
            "Percakapan di showroom mobil Mitsubishi Indonesia. "
            "Xpander, Pajero, Xforce, Outlander, harga, kredit, DP, cicilan, keluarga."
        ),
    )
    latency = time.perf_counter() - t0
    return result.text.strip(), latency


# ── STT: ElevenLabs Scribe Realtime ──────────────────────────────────────────

async def stt_elevenlabs_realtime(
    pcm_bytes: bytes,
    api_key: str,
    debug: bool = False,
) -> tuple[str, float]:
    """
    Stream PCM ke ElevenLabs Scribe Realtime via WebSocket (MANUAL commit).
    Returns (transcript, stt_latency_secs).
    Waktu dihitung dari mulai streaming sampai committed_transcript diterima.
    """
    from elevenlabs.realtime.scribe import (
        ScribeRealtime, AudioFormat, CommitStrategy,
    )
    from elevenlabs.realtime.connection import RealtimeEvents

    committed: list[str] = []
    partials:  list[str] = []
    done = asyncio.Event()

    scribe     = ScribeRealtime(api_key=api_key)
    connection = await scribe.connect({
        "model_id":               "scribe_v2_realtime",
        "audio_format":           AudioFormat.PCM_16000,
        "sample_rate":            16000,
        "language_code":          "id",
        "commit_strategy":        CommitStrategy.MANUAL,
        "min_speech_duration_ms": 100,
        "keyterms": ["Xpander", "Pajero", "Xforce", "Mitsubishi", "kredit", "DP", "cicilan"],
    })

    def on_partial(data):
        t = data.get("text", "") if isinstance(data, dict) else ""
        if t:
            partials.append(t)
            if debug:
                print(f"      [EL partial] {t!r}")

    def on_committed(data):
        t = data.get("text", "") if isinstance(data, dict) else ""
        if t:
            committed.append(t)
        if debug:
            print(f"      [EL committed] {t!r}")
        done.set()

    def on_no_activity(_):
        done.set()

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT,          on_partial)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT,        on_committed)
    connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, on_no_activity)
    connection.on(RealtimeEvents.AUTH_ERROR,                  lambda _: done.set())
    connection.on(RealtimeEvents.QUOTA_EXCEEDED,              lambda _: done.set())

    t0 = time.perf_counter()

    # Stream audio dalam chunk 4096 bytes (~128ms per chunk di 16kHz s16le)
    chunk_size = 4096
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i : i + chunk_size]
        await connection.send({"audio_base_64": base64.b64encode(chunk).decode()})
        await asyncio.sleep(0.02)   # simulasi pacing mikrofon nyata

    # Commit eksplisit setelah semua audio terkirim
    await connection.commit()

    try:
        await asyncio.wait_for(done.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        if debug:
            print("      [EL] Timeout — partials:", partials)

    latency = time.perf_counter() - t0
    await connection.close()

    text = (committed[-1] if committed else partials[-1] if partials else "").strip()
    return text, latency


# ── Classifier (dengan token tracking) ───────────────────────────────────────

async def classify_speaker(
    text: str,
    last_speaker: str,
    api_key: str,
) -> tuple[str, float, str, int, int, float]:
    """
    Klasifikasi speaker + catat metrics.

    Returns:
        speaker          : 'sales' | 'customer' | 'unknown'
        confidence       : 0.0 – 1.0
        path             : 'heuristic' | 'llm'
        prompt_tokens    : 0 jika heuristic
        completion_tokens: 0 jika heuristic
        cls_latency_secs : 0.0 jika heuristic
    """
    clean = text.strip()
    if not clean:
        return last_speaker or "unknown", 0.0, "heuristic", 0, 0, 0.0

    # ── Heuristic path: 1-2 kata ──────────────────────────────────────
    # Reaksi sangat pendek hampir selalu datang dari lawan bicara.
    if len(clean.split()) <= _SHORT_TEXT_WORDS:
        opposite = (
            "customer" if last_speaker == "sales"
            else "sales" if last_speaker == "customer"
            else "unknown"
        )
        return opposite, 0.35, "heuristic", 0, 0, 0.0

    # ── LLM path: GPT-4o-mini ─────────────────────────────────────────
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    last_label = (
        "SALES"    if last_speaker == "sales"
        else "CUSTOMER" if last_speaker == "customer"
        else "tidak diketahui"
    )

    user_prompt = (
        f"Pembicara terakhir yang diketahui: {last_label}\n\n"
        f"Kalimat baru yang harus diklasifikasikan:\n\"{clean}\"\n\n"
        "Berikan confidence RENDAH (< 0.5) jika kalimat ambigu."
    )

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        max_tokens=30,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Klasifikasikan pembicara di showroom mobil Mitsubishi Indonesia: SALES atau CUSTOMER.\n"
                    "SALES: menawarkan produk, menjelaskan fitur/harga, menyebut 'kami', bertanya kebutuhan.\n"
                    "CUSTOMER: bercerita situasi pribadi, bertanya harga/stok, menyebut budget/DP/cicilan.\n"
                    'Jawab HANYA JSON: {"speaker":"sales","confidence":0.85}'
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    cls_latency = time.perf_counter() - t0

    usage = response.usage
    prompt_tokens     = usage.prompt_tokens     if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    raw  = response.choices[0].message.content.strip()
    data = json.loads(raw)

    speaker    = data.get("speaker", "unknown")
    confidence = float(data.get("confidence", 0.5))
    if speaker not in ("sales", "customer"):
        speaker = "unknown"

    return speaker, confidence, "llm", prompt_tokens, completion_tokens, cls_latency


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(
    pipeline:          Literal["Whisper", "ElevenLabs"],
    audio_duration:    float,
    prompt_tokens:     int,
    completion_tokens: int,
) -> float:
    """Hitung estimasi biaya USD untuk satu run."""
    stt_cost = (
        audio_duration * WHISPER_COST_PER_SEC
        if pipeline == "Whisper"
        else audio_duration * EL_COST_PER_SEC
    )
    cls_cost = (
        prompt_tokens     * GPT4O_MINI_IN_PER_TOK
        + completion_tokens * GPT4O_MINI_OUT_PER_TOK
    )
    return stt_cost + cls_cost


# ── Pipeline runners ──────────────────────────────────────────────────────────

async def run_whisper_pipeline(
    scenario: ScenarioAudio,
    run_index: int,
    oai_key: str,
    last_speaker: str = "unknown",
) -> RunResult:
    try:
        transcript, stt_lat = await stt_whisper(scenario.mp3_bytes, oai_key)

        speaker, conf, path, p_tok, c_tok, cls_lat = await classify_speaker(
            transcript, last_speaker, oai_key
        )

        cost = estimate_cost("Whisper", scenario.duration_secs, p_tok, c_tok)

        return RunResult(
            pipeline="Whisper",
            scenario=scenario.label,
            run_index=run_index,
            transcript=transcript,
            speaker=speaker,
            confidence=conf,
            stt_latency=stt_lat,
            cls_latency=cls_lat,
            cls_path=path,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            audio_duration=scenario.duration_secs,
            cost_usd=cost,
        )
    except Exception as e:
        return RunResult(
            pipeline="Whisper", scenario=scenario.label, run_index=run_index,
            transcript="", speaker="unknown", confidence=0.0,
            stt_latency=0.0, cls_latency=0.0, cls_path="error",
            prompt_tokens=0, completion_tokens=0,
            audio_duration=scenario.duration_secs, cost_usd=0.0,
            error=str(e),
        )


async def run_elevenlabs_pipeline(
    scenario: ScenarioAudio,
    run_index: int,
    oai_key: str,
    el_key: str,
    last_speaker: str = "unknown",
    debug: bool = False,
) -> RunResult:
    try:
        transcript, stt_lat = await stt_elevenlabs_realtime(
            scenario.pcm_bytes, el_key, debug=debug
        )

        speaker, conf, path, p_tok, c_tok, cls_lat = await classify_speaker(
            transcript, last_speaker, oai_key
        )

        cost = estimate_cost("ElevenLabs", scenario.duration_secs, p_tok, c_tok)

        return RunResult(
            pipeline="ElevenLabs",
            scenario=scenario.label,
            run_index=run_index,
            transcript=transcript,
            speaker=speaker,
            confidence=conf,
            stt_latency=stt_lat,
            cls_latency=cls_lat,
            cls_path=path,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            audio_duration=scenario.duration_secs,
            cost_usd=cost,
        )
    except Exception as e:
        return RunResult(
            pipeline="ElevenLabs", scenario=scenario.label, run_index=run_index,
            transcript="", speaker="unknown", confidence=0.0,
            stt_latency=0.0, cls_latency=0.0, cls_path="error",
            prompt_tokens=0, completion_tokens=0,
            audio_duration=scenario.duration_secs, cost_usd=0.0,
            error=str(e),
        )


# ── Report printer ────────────────────────────────────────────────────────────

W = 100

def _bar(value: float, max_val: float = 3.0, width: int = 20) -> str:
    filled = min(int(value / max_val * width), width)
    return "█" * filled + "░" * (width - filled)


def print_run(r: RunResult) -> None:
    if r.error:
        print(f"  ✗ [{r.pipeline:<11}] run {r.run_index}  ERROR: {r.error}")
        return

    t = (r.transcript[:60] + "…") if len(r.transcript) > 60 else r.transcript or "(empty)"
    ok = "✓" if r.transcript else "✗"
    path_tag = f"[{r.cls_path}]" if r.cls_path == "heuristic" else ""

    print(
        f"  {ok} [{r.pipeline:<11}] run {r.run_index}"
        f"  stt={r.stt_latency:.2f}s  cls={r.cls_latency:.2f}s  total={r.total_latency:.2f}s"
        f"  tok={r.total_tokens:>3}  ${r.cost_usd:.6f}"
        f"  → {r.speaker:<8} conf={r.confidence:.2f} {path_tag}"
    )
    print(f"     \"{t}\"")


def print_summary(results: list[RunResult], pipelines: list[str]) -> None:
    from itertools import product as iproduct

    scenarios = ["SHORT", "LONG", "HEURISTIC"]
    metrics   = ["stt_latency", "cls_latency", "total_latency", "cost_usd", "total_tokens"]
    labels    = ["STT lat", "CLS lat", "Total lat", "Cost USD", "Tokens"]

    print(f"\n{'═' * W}")
    print("  RINGKASAN STATISTIK  (hanya run sukses)")
    print("═" * W)

    header = f"  {'Pipeline':<12} {'Scenario':<12}" + "".join(f"  {l:>11}" for l in labels)
    print(header)
    print("  " + "─" * (len(header) - 2))

    for pipe, scen in iproduct(pipelines, scenarios):
        runs = [
            r for r in results
            if r.pipeline == pipe and r.scenario == scen and not r.error
        ]
        if not runs:
            continue

        row = f"  {pipe:<12} {scen:<12}"
        for m in metrics:
            vals = [getattr(r, m) for r in runs]
            avg  = mean(vals)
            if m == "cost_usd":
                row += f"  ${avg:>9.6f}"
            elif m == "total_tokens":
                row += f"  {avg:>10.1f}"
            else:
                row += f"  {avg:>9.2f}s"
        print(row)

        # p50 / p95 untuk latency total (jika N ≥ 3)
        if len(runs) >= 3:
            lats = sorted(r.total_latency for r in runs)
            p50  = median(lats)
            p95  = lats[int(len(lats) * 0.95)]
            sd   = stdev(lats) if len(lats) > 1 else 0.0
            bar  = _bar(mean(lats))
            print(f"    {'':24} p50={p50:.2f}s  p95={p95:.2f}s  σ={sd:.2f}s  [{bar}]")

    print()
    # Tabel heuristic vs LLM path
    print("  Jalur classifier:")
    print(f"  {'Pipeline':<12} {'Scenario':<12} {'LLM runs':>10} {'Heuristic':>10} {'Avg LLM tok':>12}")
    for pipe, scen in iproduct(pipelines, scenarios):
        runs = [r for r in results if r.pipeline == pipe and r.scenario == scen and not r.error]
        if not runs:
            continue
        llm_runs = [r for r in runs if r.cls_path == "llm"]
        heur_runs = [r for r in runs if r.cls_path == "heuristic"]
        avg_tok = mean(r.total_tokens for r in llm_runs) if llm_runs else 0.0
        print(
            f"  {pipe:<12} {scen:<12} {len(llm_runs):>10} {len(heur_runs):>10} {avg_tok:>12.1f}"
        )

    print(f"\n{'═' * W}")
    print("  Pricing reference:")
    print(f"  Whisper-1       : ${WHISPER_COST_PER_SEC:.7f}/dtk audio  (${WHISPER_COST_PER_SEC*60:.4f}/menit)")
    print(f"  ElevenLabs Scribe Realtime: ${EL_COST_PER_SEC:.7f}/dtk audio  (${EL_COST_PER_SEC*3600:.2f}/jam)")
    print(f"  {CLASSIFIER_MODEL} : ${GPT4O_MINI_IN_PER_TOK*1e6:.2f}/1M input tokens  ${GPT4O_MINI_OUT_PER_TOK*1e6:.2f}/1M output tokens")
    print("═" * W)


# ── Skenario audio ────────────────────────────────────────────────────────────

AUDIO_SCENARIOS = [
    ScenarioAudio(
        label="SHORT",
        text=(
            "Harga Xpander berapa? Bisa kasih info soal DP dan cicilan?"
        ),
        # Target ~3 dtk di kecepatan TTS normal
    ),
    ScenarioAudio(
        label="LONG",
        text=(
            "Selamat pagi, saya sedang mencari mobil untuk keluarga. "
            "Saya tertarik dengan Mitsubishi Xpander. "
            "Bisa tolong jelaskan berapa DP minimal dan estimasi cicilan per bulan "
            "untuk tenor lima tahun? Apakah ada promo atau diskon bulan ini?"
        ),
        # Target ~9 dtk di kecepatan TTS normal
    ),
    ScenarioAudio(
        label="HEURISTIC",
        text="Iya, Xpander.",
        # 1-2 kata → heuristic path, bypass LLM classifier
    ),
]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Load Test: STT + Classifier")
    parser.add_argument("--runs",           type=int,  default=3,     help="Jumlah run per skenario (default: 3)")
    parser.add_argument("--no-whisper",     action="store_true",      help="Skip pipeline Whisper")
    parser.add_argument("--no-elevenlabs",  action="store_true",      help="Skip pipeline ElevenLabs")
    parser.add_argument("--debug",          action="store_true",      help="Tampilkan raw WebSocket events")
    args = parser.parse_args()

    use_whisper = not args.no_whisper
    use_eleven  = not args.no_elevenlabs

    from backend.config import settings
    oai_key = settings.openai_api_key
    el_key  = settings.elevenlabs_api_key

    pipelines = []
    if use_whisper:  pipelines.append("Whisper")
    if use_eleven:   pipelines.append("ElevenLabs")

    if not pipelines:
        print("ERROR: Setidaknya satu pipeline harus aktif.")
        return

    print("=" * W)
    print("  Pipeline Load Test: STT + Speaker Classifier")
    print(f"  Pipeline   : {' | '.join(pipelines)}")
    print(f"  Classifier : {CLASSIFIER_MODEL}")
    print(f"  Runs       : {args.runs} per skenario")
    print("=" * W)

    # ── 1. Generate audio mock (sekali, di-cache) ─────────────────────
    print(f"\n  Generating mock audio via OpenAI TTS...")
    for sc in AUDIO_SCENARIOS:
        sc.mp3_bytes = await generate_tts(sc.text, oai_key)
        sc.pcm_bytes = to_pcm16k(sc.mp3_bytes)
        print(
            f"  [{sc.label:<10}] {len(sc.mp3_bytes):>7,} B MP3  "
            f"{len(sc.pcm_bytes):>9,} B PCM  "
            f"≈ {sc.duration_secs:.1f}s  \"{sc.text[:55]}{'…' if len(sc.text) > 55 else ''}\""
        )

    all_results: list[RunResult] = []

    # ── 2. Load test loop ─────────────────────────────────────────────
    for sc in AUDIO_SCENARIOS:
        print(f"\n{'─' * W}")
        print(f"  SKENARIO: {sc.label}  │  Audio: {sc.duration_secs:.1f}s  │  Text: \"{sc.text[:70]}\"")
        print("─" * W)

        for run_idx in range(1, args.runs + 1):
            print(f"\n  ── Run {run_idx}/{args.runs} ──────────────────────────────────────────")

            tasks = []
            if use_whisper:
                tasks.append(run_whisper_pipeline(sc, run_idx, oai_key))
            if use_eleven:
                tasks.append(run_elevenlabs_pipeline(sc, run_idx, oai_key, el_key, debug=args.debug))

            # Jalankan pipeline secara concurrent dalam satu run
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    print(f"  ✗ Unexpected error: {res}")
                    continue
                all_results.append(res)
                print_run(res)

            # Jeda kecil antar run untuk menghindari rate limit
            if run_idx < args.runs:
                await asyncio.sleep(0.5)

    # ── 3. Summary ────────────────────────────────────────────────────
    print_summary(all_results, pipelines)

    # ── 4. Export CSV (opsional — untuk analisis lanjut) ─────────────
    csv_path = "pipeline_loadtest_results.csv"
    with open(csv_path, "w") as f:
        f.write(
            "pipeline,scenario,run,transcript,speaker,confidence,path,"
            "stt_latency,cls_latency,total_latency,"
            "prompt_tokens,completion_tokens,total_tokens,"
            "audio_duration_secs,cost_usd,error\n"
        )
        for r in all_results:
            transcript_esc = r.transcript.replace('"', '""')
            f.write(
                f"{r.pipeline},{r.scenario},{r.run_index},"
                f'"{transcript_esc}",{r.speaker},{r.confidence:.3f},{r.cls_path},'
                f"{r.stt_latency:.4f},{r.cls_latency:.4f},{r.total_latency:.4f},"
                f"{r.prompt_tokens},{r.completion_tokens},{r.total_tokens},"
                f"{r.audio_duration:.2f},{r.cost_usd:.8f},{r.error}\n"
            )
    print(f"\n  Hasil CSV tersimpan: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
