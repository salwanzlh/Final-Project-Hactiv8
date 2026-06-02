"""
Mic Audio Diagnostic
Capture 5s from mic → check amplitude → test with ElevenLabs batch API.
This tells us: (1) is audio loud enough? (2) does the content transcribe at all?

Usage:
    uv run python test_mic_audio_check.py
"""
import asyncio
import io
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000


async def main():
    from backend.config import settings

    print("=" * 60)
    print("  Mic Diagnostic")
    print("=" * 60)
    print("\n  Recording 5 seconds — speak now...\n")

    audio = sd.rec(5 * SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    audio_flat = audio.flatten()
    rms = np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2))
    peak = np.abs(audio_flat).max()
    print(f"  Samples : {len(audio_flat):,}")
    print(f"  RMS     : {rms:.1f}  (good speech ≥ 500)")
    print(f"  Peak    : {peak:,}  (max 32767)")
    print(f"  Volume  : {'OK ✓' if rms >= 500 else 'TOO QUIET ✗ — move closer to mic'}")

    # Save to WAV
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_flat.tobytes())
    wav_bytes = wav_buf.getvalue()

    with open("/tmp/mic_test.wav", "wb") as f:
        f.write(wav_bytes)
    print(f"\n  Saved: /tmp/mic_test.wav  ({len(wav_bytes):,} bytes)")
    print("  Play back: afplay /tmp/mic_test.wav")

    # Test with ElevenLabs batch (baseline)
    print("\n  Sending to ElevenLabs batch (scribe_v2)...")
    from elevenlabs.client import AsyncElevenLabs
    client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
    result = await client.speech_to_text.convert(
        model_id="scribe_v2",
        file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
        language_code="id",
    )
    batch_text = (result.text or "").strip()
    print(f"  Batch result: {batch_text!r}")
    if batch_text:
        print("  → Audio content is transcribable ✓  (issue is in realtime streaming)")
    else:
        print("  → Batch also empty ✗  (issue is in audio content/volume)")

    # Test with ElevenLabs realtime (manual commit)
    print("\n  Sending same WAV to ElevenLabs realtime (manual commit)...")
    import base64
    from elevenlabs.realtime.scribe import ScribeRealtime, AudioFormat, CommitStrategy
    from elevenlabs.realtime.connection import RealtimeEvents

    raw_pcm = audio_flat.tobytes()
    scribe = ScribeRealtime(api_key=settings.elevenlabs_api_key)
    connection = await scribe.connect({
        "model_id":        "scribe_v2_realtime",
        "audio_format":    AudioFormat.PCM_16000,
        "sample_rate":     SAMPLE_RATE,
        "commit_strategy": CommitStrategy.MANUAL,
        "language_code":   "id",
    })

    done = asyncio.Event()
    results: list[str] = []

    def on_any(data, evt_name):
        msg = data.get("transcript", data) if isinstance(data, dict) else data
        print(f"  [realtime evt={evt_name}] {str(msg)[:100]}")
        if evt_name in ("committed_transcript", "insufficient_audio_activity"):
            if isinstance(data, dict) and data.get("transcript"):
                results.append(data["transcript"])
            done.set()

    for evt in RealtimeEvents:
        connection.on(evt, lambda d, e=evt: on_any(d, e.value))

    # Send PCM in 3200-byte chunks (100ms each)
    chunk_size = 3200
    n = 0
    for i in range(0, len(raw_pcm), chunk_size):
        await connection.send({"audio_base_64": base64.b64encode(raw_pcm[i:i+chunk_size]).decode()})
        n += 1
    print(f"  Sent {n} chunks ({len(raw_pcm):,} bytes PCM)")

    await connection.commit()
    print("  commit() sent — waiting…")

    try:
        await asyncio.wait_for(done.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        print("  TIMEOUT — no response after 10s")

    await connection.close()

    rt_text = results[-1] if results else ""
    print(f"\n  Realtime result: {rt_text!r}")

    print("\n" + "=" * 60)
    print(f"  Batch   : {batch_text!r}")
    print(f"  Realtime: {rt_text!r}")
    if batch_text and not rt_text:
        print("  → Realtime API issue (audio is fine, batch works)")
    elif batch_text and rt_text:
        print("  → Both work ✓")
    elif not batch_text and not rt_text:
        print("  → Audio issue (too quiet or wrong format)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
