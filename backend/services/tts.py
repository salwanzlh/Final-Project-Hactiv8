"""
TTS Service — Text-to-Speech via OpenAI.

Dipanggil setiap kali AI menghasilkan suggested_question baru.
Return: bytes audio (mp3) yang langsung dikirim ke frontend via WebSocket.
"""
import logging
import langdetect
from backend.config import settings
from backend.services.langfuse_client import langfuse
from backend.services.metrics import tts_requests, tts_latency, error_total, track_latency

logger = logging.getLogger(__name__)


async def synthesize(text: str) -> bytes | None:
    """
    Konversi teks ke audio bytes (mp3).
    Mode demo: return None — frontend akan skip playback.
    Mode production: panggil OpenAI TTS API.
    """
    if settings.app_mode == "demo":
        # Di demo mode, frontend pakai browser SpeechSynthesis sebagai fallback
        return None

    if not settings.openai_api_key:
        logger.warning("[TTS] OPENAI_API_KEY belum diset, skip TTS.")
        return None

    try:
        detected_lang = langdetect.detect(text)
        if detected_lang not in ("id", "en"):
            logger.warning(f"[TTS] Bahasa tidak didukung '{detected_lang}', skip TTS.")
            return None
        if detected_lang != settings.language:
            logger.warning(f"[TTS] Bahasa '{detected_lang}' tidak sesuai config '{settings.language}', skip TTS.")
            return None

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        lang_instruction = "Speak in Indonesian." if detected_lang == "id" else "Speak in English."
        model = settings.tts_model

        trace = langfuse.trace(name="tts", input={"text": text, "chars": len(text), "language": detected_lang})
        span = trace.span(name=model, input={"text": text, "chars": len(text), "model": model, "language": detected_lang})

        create_kwargs = dict(
            model=model,
            voice=settings.tts_voice,
            input=text,
            response_format="mp3",
            speed=settings.tts_speed,
        )
        if "gpt-4o" in model:
            create_kwargs["instructions"] = lang_instruction

        async with track_latency(tts_latency):
            response = await client.audio.speech.create(**create_kwargs)

        audio_bytes = response.content
        span.end(output={"audio_bytes": len(audio_bytes), "model": model})
        trace.update(output={"audio_bytes": len(audio_bytes)})
        tts_requests.labels(status="sukses").inc()
        logger.info(f"[TTS] Berhasil generate {len(audio_bytes)} bytes untuk: '{text[:50]}...'")
        return audio_bytes

    except Exception as e:
        tts_requests.labels(status="gagal").inc()
        error_total.labels(komponen="tts", tipe_error=type(e).__name__).inc()
        logger.error(f"[TTS] Gagal generate audio: {e}")
        return None
