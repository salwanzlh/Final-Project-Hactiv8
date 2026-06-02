"""
STT Service — Speech-to-Text.
Mode demo: langsung return dummy transkrip.
Mode production: kirim audio ke Whisper API / ElevenLabs Scribe Realtime.
"""
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from backend.config import settings
from backend.models.schemas import Utterance
from backend.services.langfuse_client import langfuse
from backend.services.metrics import stt_requests, stt_latency, error_total, track_latency

logger = logging.getLogger(__name__)

# # Exact-match hallucinations (very short / punctuation-only)
# _HALLUCINATIONS_EXACT = {
#     "you", ".", ",", "-", "...", "[musik]", "[music]",
#     "musik", "music", "subscribe", "subscribed", "",
# }

# # Prefix-based hallucinations — Whisper generates endless variations of these
# # Any text that STARTS WITH one of these prefixes (after lowercasing) is dropped
# _HALLUCINATION_PREFIXES = (
#     "terima kasih",          # "terima kasih atas dukungan Anda", "terima kasih telah ..."
#     "thank you",
#     "sampai jumpa",          # "sampai jumpa di video ...", "sampai jumpa lagi"
#     "selamat menikmati",
#     "subtitle by",
#     "subtitles by",
#     "like dan subscribe",
#     "jangan lupa subscribe",
#     "jangan lupa like",
#     "don't forget to",
#     "please subscribe",
#     "selamat datang di channel",
# )

_NO_SPEECH_PROB_THRESHOLD = 0.5  # turunkan dari 0.6 → tangkap lebih banyak silence


# def _is_hallucination(text: str) -> bool:
#     normalized = text.lower().strip()
#     if normalized in _HALLUCINATIONS_EXACT:
#         return True
#     return any(normalized.startswith(p) for p in _HALLUCINATION_PREFIXES)


def _check_audio_energy(audio_bytes: bytes) -> bool:
    """Return True if audio has enough energy to be real speech (rough RMS check on raw bytes)."""
    # import struct
    # Ambil sample setelah header WebM (skip 200 bytes pertama)
    raw = audio_bytes[200:]
    if len(raw) < 200:
        return False
    # Hitung rata-rata nilai absolut byte sebagai proxy RMS
    avg = sum(abs(b - 128) for b in raw[:2000]) / 2000
    return avg > 3.0  # threshold empiris; < 3 = silence/noise


# # Dummy transkrip untuk mode demo — disimulasikan bergantian
# _DEMO_SCRIPT = [
#     ("customer", "Selamat pagi, saya lagi nyari mobil untuk keluarga."),
#     ("sales",    "Selamat pagi Pak! Boleh saya tahu, biasanya bepergian berapa orang?"),
#     ("customer", "Biasanya 5 sampai 6 orang, ada anak kecil juga."),
#     ("sales",    "Baik, budget yang disiapkan kira-kira berapa Pak?"),
#     ("customer", "Sekitar 250 sampai 350 juta kalau bisa."),
#     ("customer", "Sering juga ke luar kota, jalanannya kadang tidak bagus."),
# ]
# _demo_index = 0


async def transcribe_audio(audio_bytes: bytes, session_id: str) -> Utterance | None:
    """
    Konversi audio bytes → Utterance.
    Swap ke production: ganti isi blok `if production`.
    """
    # global _demo_index

    # if settings.app_mode == "demo":
    #     async with track_latency(stt_latency):
    #         await asyncio.sleep(0.4)  # Simulasi latency STT
    #     if _demo_index >= len(_DEMO_SCRIPT):
    #         _demo_index = 0
    #     speaker, text = _DEMO_SCRIPT[_demo_index]
    #     _demo_index += 1
    #     stt_requests.labels(status="sukses").inc()
    #     return Utterance(
    #         id=str(uuid.uuid4()),
    #         speaker=speaker,
    #         text=text,
    #         timestamp=datetime.now(timezone.utc),
    #         confidence=0.95,
    #     )

    # ── PRODUCTION ────────────────────────────────
    if len(audio_bytes) < 1000:
        logger.warning(f"[STT] Audio terlalu kecil ({len(audio_bytes)} bytes), skip.")
        return None

    if not _check_audio_energy(audio_bytes):
        logger.info("[STT] Audio energy terlalu rendah (silence/noise), skip.")
        return None

    import io

    if settings.stt_provider == "elevenlabs":
        import base64
        from elevenlabs.realtime.scribe import ScribeRealtime, AudioFormat, CommitStrategy
        from elevenlabs.realtime.connection import RealtimeEvents

        # WebM → PCM 16kHz mono s16le (ffmpeg) — pakai async subprocess agar event loop tidak blok
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1", "-loglevel", "quiet",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        pcm_bytes, _ = await proc.communicate(input=audio_bytes)
        if not pcm_bytes:
            logger.warning("[STT] ffmpeg: gagal konversi WebM → PCM, skip.")
            return None

        logger.info(f"[STT] Streaming {len(pcm_bytes)} bytes PCM ke ElevenLabs Scribe Realtime...")
        trace = langfuse.trace(name="stt", input={"session_id": session_id, "audio_bytes": len(audio_bytes)})
        span  = trace.span(name="scribe_v2_realtime", input={"pcm_bytes": len(pcm_bytes)})

        committed: list[str] = []
        done = asyncio.Event()

        scribe     = ScribeRealtime(api_key=settings.elevenlabs_api_key)
        connection = await scribe.connect({
            "model_id":               "scribe_v2_realtime",
            "audio_format":           AudioFormat.PCM_16000,
            "sample_rate":            16000,
            "language_code":          settings.language,
            "commit_strategy":        CommitStrategy.MANUAL,
            "min_speech_duration_ms": 100,
            "keyterms": ["Xpander", "Pajero", "Xforce", "Outlander", "Mitsubishi", "kredit", "DP", "cicilan",
                         "mudik", "BBM", "bepergian", "diesel", 'Triton', 'Eclipse', 'Destinator'],
        })

        def on_committed(data):
            t = data.get("text", "") if isinstance(data, dict) else ""
            logger.info(f"[STT] ElevenLabs committed: '{t}'")
            if t:
                committed.append(t)
            done.set()

        def on_insufficient(_):
            logger.info("[STT] ElevenLabs: insufficient audio activity")
            done.set()

        def on_auth_error(_):
            logger.error("[STT] ElevenLabs: AUTH_ERROR — cek ELEVENLABS_API_KEY")
            done.set()

        def on_quota(_):
            logger.error("[STT] ElevenLabs: QUOTA_EXCEEDED")
            done.set()

        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT,        on_committed)
        connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, on_insufficient)
        connection.on(RealtimeEvents.AUTH_ERROR,                  on_auth_error)
        connection.on(RealtimeEvents.QUOTA_EXCEEDED,              on_quota)

        logger.info(f"[STT] Mengirim {len(pcm_bytes)} bytes PCM ke ElevenLabs Scribe Realtime...")
        try:
            async with track_latency(stt_latency):
                chunk_size = 4096  # ~128ms at 16kHz s16le
                for i in range(0, len(pcm_bytes), chunk_size):
                    await connection.send({"audio_base_64": base64.b64encode(pcm_bytes[i:i+chunk_size]).decode()})
                await connection.commit()
                logger.info("[STT] Audio di-commit, menunggu transkrip...")
                try:
                    await asyncio.wait_for(done.wait(), timeout=8.0)
                except asyncio.TimeoutError:
                    logger.warning("[STT] ElevenLabs Realtime timeout 8s — tidak ada respons dari server")
        except Exception as e:
            stt_requests.labels(status="gagal").inc()
            error_total.labels(komponen="stt", tipe_error=type(e).__name__).inc()
            raise
        finally:
            await connection.close()

        text = (committed[-1] if committed else "").strip()
        if not text:
            logger.info("[STT] ElevenLabs: tidak ada transkrip yang di-commit, skip.")
            span.end(output={"discarded": True})
            trace.update(output={"discarded": True})
            return None
        span.end(output={"text": text, "chars": len(text)})
        trace.update(output={"text": text})

    elif settings.stt_provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info(f"[STT] Mengirim {len(audio_bytes)} bytes ke Whisper...")

        trace = langfuse.trace(name="stt", input={"session_id": session_id, "audio_bytes": len(audio_bytes)})
        span  = trace.span(name="whisper-1", input={"audio_bytes": len(audio_bytes), "model": "whisper-1"})

        try:
            async with track_latency(stt_latency):
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("audio.webm", io.BytesIO(audio_bytes), "audio/webm"),
                    language=settings.language,
                    response_format="verbose_json",
                    prompt="Percakapan di showroom mobil Mitsubishi Indonesia. Sales dan customer berbicara dalam bahasa Indonesia. Xpander, Pajero, Xforce, Outlander, harga, kredit, DP, cicilan, keluarga.",
                )
        except Exception as e:
            stt_requests.labels(status="gagal").inc()
            error_total.labels(komponen="stt", tipe_error=type(e).__name__).inc()
            raise

        text = result.text.strip()

        no_speech_prob = getattr(result, "no_speech_prob", None)
        if no_speech_prob is None and hasattr(result, "segments") and result.segments:
            no_speech_prob = getattr(result.segments[0], "no_speech_prob", 0)

        if no_speech_prob is not None and no_speech_prob > _NO_SPEECH_PROB_THRESHOLD:
            logger.info(f"[STT] no_speech_prob={no_speech_prob:.2f} terlalu tinggi, buang: '{text}'")
            span.end(output={"discarded": True, "no_speech_prob": no_speech_prob})
            trace.update(output={"discarded": True})
            return None

        span.end(output={"text": text, "chars": len(text), "no_speech_prob": no_speech_prob})
        trace.update(output={"text": text})

    else:
        logger.warning(f"[STT] Provider '{settings.stt_provider}' tidak didukung.")
        return None

    # if _is_hallucination(text):
    #     logger.info(f"[STT] Hallucination dibuang: '{text}'")
    #     return None

    stt_requests.labels(status="sukses").inc()
    return Utterance(
        id=str(uuid.uuid4()),
        speaker="unknown",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


