"""
STT Comparison: Whisper-1  vs  ElevenLabs Scribe (batch)  vs  ElevenLabs Scribe Realtime

How the test works:
  1. Generate test audio with OpenAI TTS (Indonesian phrases + silence)
  2. Send the SAME audio to all three providers
  3. Print: transcript, latency, hallucination behaviour

Test scenarios:
  A. Normal sales conversation phrase
  B. Car product names (Mitsubishi vocabulary)
  C. Fast mixed utterance (two people close together)
  D. Near-silence (tests hallucination resistance)

Usage:
    python test_stt_compare.py
"""

import asyncio
import base64
import io
import subprocess
import time
import wave
import struct

# ── Audio generation helpers ─────────────────────────────────────────

async def generate_tts_audio(text: str, openai_api_key: str) -> bytes:
    """Generate MP3 audio from text using OpenAI TTS."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=openai_api_key)
    response = await client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="mp3",
    )
    return response.content


def generate_silence_wav(duration_secs: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate near-silence WAV (very low amplitude noise, like an idle microphone)."""
    import random
    buf = io.BytesIO()
    n_samples = int(sample_rate * duration_secs)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Very low amplitude random noise (amplitude ~30 out of 32768)
        samples = [random.randint(-30, 30) for _ in range(n_samples)]
        wf.writeframes(struct.pack(f'<{n_samples}h', *samples))
    return buf.getvalue()


def mp3_to_pcm16k(mp3_bytes: bytes) -> bytes:
    """Convert MP3 bytes → raw PCM 16kHz mono 16-bit via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0",
         "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
         "-loglevel", "quiet"],
        input=mp3_bytes,
        capture_output=True,
    )
    return result.stdout


def wav_to_pcm16k(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes → raw PCM 16kHz mono 16-bit via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0",
         "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
         "-loglevel", "quiet"],
        input=wav_bytes,
        capture_output=True,
    )
    return result.stdout


# ── STT callers ──────────────────────────────────────────────────────

async def transcribe_whisper(audio_bytes: bytes, openai_api_key: str, label: str = "mp3") -> tuple[str, float]:
    from openai import AsyncOpenAI
    import io
    client = AsyncOpenAI(api_key=openai_api_key)
    t0 = time.perf_counter()
    result = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(f"audio.{label}", io.BytesIO(audio_bytes), f"audio/{label}"),
        language="id",
        response_format="verbose_json",
        prompt="Percakapan di showroom mobil Mitsubishi Indonesia. Sales dan customer berbicara dalam bahasa Indonesia. Xpander, Pajero, Xforce, Outlander, harga, kredit, DP, cicilan, keluarga.",
    )
    latency = time.perf_counter() - t0

    no_speech_prob = getattr(result, "no_speech_prob", None)
    if no_speech_prob is None and hasattr(result, "segments") and result.segments:
        no_speech_prob = getattr(result.segments[0], "no_speech_prob", None)

    return result.text.strip(), latency, no_speech_prob


async def transcribe_scribe_batch(audio_bytes: bytes, elevenlabs_api_key: str, fmt: str = "mp3") -> tuple[str, float]:
    from elevenlabs.client import AsyncElevenLabs
    client = AsyncElevenLabs(api_key=elevenlabs_api_key)
    mime = "audio/mpeg" if fmt == "mp3" else "audio/wav"
    t0 = time.perf_counter()
    result = await client.speech_to_text.convert(
        model_id="scribe_v2",
        file=(f"audio.{fmt}", io.BytesIO(audio_bytes), mime),
        language_code="id",
        keyterms=["Xpander", "Pajero", "Xforce", "Outlander", "Mitsubishi", "showroom", "kredit", "DP"],
        tag_audio_events=False,
    )
    latency = time.perf_counter() - t0
    return result.text.strip(), latency


async def transcribe_scribe_realtime(pcm_bytes: bytes, elevenlabs_api_key: str) -> tuple[str, float]:
    """
    Stream PCM audio to ElevenLabs Scribe Realtime WebSocket.
    Uses VAD commit strategy — ElevenLabs detects speech end automatically.
    """
    from elevenlabs.realtime.scribe import ScribeRealtime, AudioFormat, CommitStrategy, RealtimeAudioOptions
    from elevenlabs.realtime.connection import RealtimeEvents

    committed_text: list[str] = []
    partial_text:   list[str] = []
    done_event = asyncio.Event()
    t0 = time.perf_counter()

    scribe = ScribeRealtime(api_key=elevenlabs_api_key)
    options: RealtimeAudioOptions = {
        "model_id": "scribe_v2_realtime",
        "audio_format": AudioFormat.PCM_16000,
        "sample_rate": 16000,
        "language_code": "id",
        "commit_strategy": CommitStrategy.VAD,
        "vad_silence_threshold_secs": 0.6,
        "min_speech_duration_ms": 200,
        "keyterms": ["Xpander", "Pajero", "Xforce", "Outlander", "Mitsubishi", "kredit"],
    }

    connection = await scribe.connect(options)

    def on_partial(data):
        text = data.get("text", "") if isinstance(data, dict) else ""
        if text:
            partial_text.append(text)

    def on_committed(data):
        text = data.get("text", "") if isinstance(data, dict) else ""
        if text:
            committed_text.append(text)
        done_event.set()

    def on_error(data):
        done_event.set()

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT,   on_partial)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
    connection.on(RealtimeEvents.ERROR,                on_error)
    connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, lambda _: done_event.set())

    # Stream PCM in 4096-byte chunks (~128ms at 16kHz 16-bit mono)
    chunk_size = 4096
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        await connection.send({"audio_base_64": base64.b64encode(chunk).decode("utf-8")})
        await asyncio.sleep(0.02)  # realistic mic pacing

    # Wait for VAD to commit (up to 6s)
    try:
        await asyncio.wait_for(done_event.wait(), timeout=6.0)
    except asyncio.TimeoutError:
        pass

    latency = time.perf_counter() - t0
    await connection.close()

    return (committed_text[-1] if committed_text else partial_text[-1] if partial_text else "").strip(), latency


# ── Print helper ─────────────────────────────────────────────────────

def row(label: str, text: str, latency: float, extra: str = ""):
    marker = "✓" if text else "✗"
    truncated = (text[:80] + "…") if len(text) > 80 else text
    print(f"  {marker} [{label:<22}] {latency:5.2f}s  |  \"{truncated}\"  {extra}")


# ── Main ─────────────────────────────────────────────────────────────

SCENARIOS = {
    "A. Normal conversation": "Selamat pagi, saya lagi cari mobil untuk keluarga. Ada rekomendasi tidak?",
    "B. Product names":       "Harga Pajero Sport Dakar dan Xforce Ultimate berapa ya, bisa kredit DP berapa?",
    "C. Fast mixed speech":   "Halo selamat datang ada yang bisa saya bantu? Iya saya mau lihat Xpander dong.",
}


async def main():
    from backend.config import settings

    openai_key    = settings.openai_api_key
    elevenlabs_key = settings.elevenlabs_api_key

    print("=" * 80)
    print("  STT Comparison: Whisper-1  |  ElevenLabs Scribe (batch)  |  Scribe Realtime")
    print("=" * 80)

    for label, text in SCENARIOS.items():
        print(f"\n{'─' * 80}")
        print(f"  SCENARIO {label}")
        print(f"  Expected: \"{text}\"")
        print("─" * 80)

        # Generate audio once, reuse for all three providers
        print("  Generating TTS audio...", end=" ", flush=True)
        mp3_bytes = await generate_tts_audio(text, openai_key)
        pcm_bytes = mp3_to_pcm16k(mp3_bytes)
        print(f"{len(mp3_bytes):,} bytes MP3 / {len(pcm_bytes):,} bytes PCM")

        # Run Whisper and Scribe batch concurrently, then Realtime after
        whisper_task = asyncio.create_task(transcribe_whisper(mp3_bytes, openai_key, "mp3"))
        scribe_task  = asyncio.create_task(transcribe_scribe_batch(mp3_bytes, elevenlabs_key))
        results = await asyncio.gather(whisper_task, scribe_task, return_exceptions=True)

        print()
        if isinstance(results[0], Exception):
            print(f"  ✗ [Whisper-1              ] ERROR: {results[0]}")
        else:
            text_out, latency, nsp = results[0]
            row("Whisper-1", text_out, latency, f"(no_speech_prob={nsp:.2f})" if nsp else "")

        if isinstance(results[1], Exception):
            print(f"  ✗ [Scribe batch           ] ERROR: {results[1]}")
        else:
            text_out, latency = results[1]
            row("Scribe batch", text_out, latency)

        # Realtime runs after (sequential — needs its own connection)
        try:
            text_out, latency = await transcribe_scribe_realtime(pcm_bytes, elevenlabs_key)
            row("Scribe realtime", text_out, latency)
        except Exception as e:
            print(f"  ✗ [Scribe realtime        ] ERROR: {e}")

    # ── Hallucination test (silence) ─────────────────────────────────
    print(f"\n{'─' * 80}")
    print("  SCENARIO D. Near-silence (hallucination resistance)")
    print("─" * 80)

    silence_wav = generate_silence_wav(duration_secs=2.0)
    silence_pcm = wav_to_pcm16k(silence_wav)

    # Whisper needs WAV for silence (MP3 encoder adds artifacts)
    whisper_task  = asyncio.create_task(transcribe_whisper(silence_wav, openai_key, "wav"))
    scribe_task   = asyncio.create_task(transcribe_scribe_batch(silence_wav, elevenlabs_key, fmt="wav"))
    results = await asyncio.gather(whisper_task, scribe_task, return_exceptions=True)

    print()
    if isinstance(results[0], Exception):
        print(f"  ✗ [Whisper-1              ] ERROR: {results[0]}")
    else:
        text_out, latency, nsp = results[0]
        hallucinated = "HALLUCINATION" if text_out else "clean (empty)"
        row("Whisper-1", text_out or "(empty)", latency, f"← {hallucinated}  nsp={nsp:.2f}" if nsp else f"← {hallucinated}")

    if isinstance(results[1], Exception):
        print(f"  ✗ [Scribe batch           ] ERROR: {results[1]}")
    else:
        text_out, latency = results[1]
        hallucinated = "HALLUCINATION" if text_out else "clean (empty)"
        row("Scribe batch", text_out or "(empty)", latency, f"← {hallucinated}")

    try:
        text_out, latency = await transcribe_scribe_realtime(silence_pcm, elevenlabs_key)
        hallucinated = "HALLUCINATION" if text_out else "clean (empty)"
        row("Scribe realtime", text_out or "(empty)", latency, f"← {hallucinated}")
    except Exception as e:
        print(f"  ✗ [Scribe realtime        ] ERROR: {e}")

    print(f"\n{'=' * 80}")
    print("  Key differences:")
    print("  Scribe batch    → drop-in Whisper replacement, same architecture")
    print("  Scribe realtime → WebSocket stream, built-in VAD, needs PCM audio")
    print("  Whisper         → current, good but hallucinates on silence")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
