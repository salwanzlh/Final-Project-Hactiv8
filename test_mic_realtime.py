"""
Live Mic Realtime Transcription Test
=====================================
Rekam dari mic langsung → stream ke STT provider → tampilkan transcript real-time.

Modes:
  --openai     OpenAI Realtime (gpt-4o-mini-transcribe)
  --elevenlabs ElevenLabs Scribe Realtime
  --both       Jalankan kedua provider bersamaan (default)

Controls:
  SPACE  mulai/stop rekaman
  Q      quit

Usage:
    uv run python test_mic_realtime.py
    uv run python test_mic_realtime.py --openai
    uv run python test_mic_realtime.py --elevenlabs
    uv run python test_mic_realtime.py --duration 10   # rekam 10 detik otomatis
"""

import asyncio
import argparse
import base64
import json
import sys
import time
import threading
import queue

import sounddevice as sd
import numpy as np

SAMPLE_RATE    = 16000
CHANNELS       = 1
DTYPE          = "int16"
CHUNK_DURATION = 0.1   # 100ms per chunk → 1600 samples


# ── Audio capture ──────────────────────────────────────────────────────────────

class MicRecorder:
    """Captures mic audio into a queue as raw PCM bytes (int16 16kHz)."""

    def __init__(self):
        self._q: queue.Queue[bytes] = queue.Queue()
        self._stream = None
        self._recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"  [mic] {status}", file=sys.stderr)
        if self._recording:
            self._q.put(indata.copy().tobytes())

    def start(self, device=None):
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def drain(self) -> bytes:
        """Return all buffered PCM bytes."""
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get_nowait())
        return b"".join(chunks)

    def get_chunk(self, timeout: float = 0.2) -> bytes | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ── OpenAI Realtime ────────────────────────────────────────────────────────────

async def run_openai_realtime(
    api_key: str,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    result_callback,
    debug: bool = False,
):
    """
    Connects to OpenAI Realtime transcription WebSocket,
    streams audio from audio_queue, prints transcript as it arrives.
    """
    import websockets

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta":   "realtime=v2",
    }

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Configure session
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model":    "gpt-4o-mini-transcribe",
                        "language": "id",
                        "prompt":   "Percakapan di showroom mobil Mitsubishi Indonesia. Xpander, Pajero, Xforce, kredit, DP.",
                    },
                    "turn_detection": {
                        "type":                 "server_vad",
                        "silence_duration_ms":  600,
                        "threshold":            0.5,
                        "prefix_padding_ms":    200,
                    },
                },
            }))

            partial_buf = ""

            async def receive_loop():
                nonlocal partial_buf
                async for raw in ws:
                    evt   = json.loads(raw)
                    etype = evt.get("type", "")

                    if etype == "conversation.item.input_audio_transcription.delta":
                        delta = evt.get("delta", "")
                        partial_buf += delta
                        result_callback("openai", "partial", partial_buf)

                    elif etype == "conversation.item.input_audio_transcription.completed":
                        text = evt.get("transcript", "").strip()
                        partial_buf = ""
                        if text:
                            result_callback("openai", "final", text)

                    elif etype == "input_audio_buffer.speech_started":
                        result_callback("openai", "status", "🎤 speech detected")

                    elif etype == "input_audio_buffer.speech_stopped":
                        result_callback("openai", "status", "🔇 speech stopped, transcribing…")

                    elif etype == "error":
                        result_callback("openai", "error", str(evt.get("error", evt)))

            recv_task = asyncio.create_task(receive_loop())

            # Stream audio chunks until stop
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.3)
                    await ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }))
                except asyncio.TimeoutError:
                    continue

            # Flush remaining buffer
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.sleep(2.0)   # wait for final transcript

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        result_callback("openai", "error", f"{type(e).__name__}: {e}")


# ── ElevenLabs Scribe Realtime ────────────────────────────────────────────────

async def run_elevenlabs_realtime(
    api_key: str,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    result_callback,
    debug: bool = False,
):
    """
    Connects to ElevenLabs Scribe Realtime WebSocket.
    Uses CommitStrategy.MANUAL — explicit commit() after audio ends.
    This avoids VAD timing issues where speech end is not detected.
    """
    from elevenlabs.realtime.scribe import (
        ScribeRealtime, AudioFormat, CommitStrategy, RealtimeAudioOptions,
    )
    from elevenlabs.realtime.connection import RealtimeEvents

    scribe  = ScribeRealtime(api_key=api_key)
    options: RealtimeAudioOptions = {
        "model_id":               "scribe_v2_realtime",
        "audio_format":           AudioFormat.PCM_16000,
        "sample_rate":            16000,
        "language_code":          "id",
        "commit_strategy":        CommitStrategy.MANUAL,   # explicit commit after mic stops
        "min_speech_duration_ms": 100,
        "keyterms": ["Xpander", "Pajero", "Xforce", "Outlander", "Mitsubishi", "kredit", "DP"],
    }

    connection = await scribe.connect(options)
    done_event = asyncio.Event()

    def on_partial(data):
        t = data.get("text", "") if isinstance(data, dict) else str(data)
        if debug:
            result_callback("elevenlabs", "status", f"⏳ partial: {t!r}")
        if t:
            result_callback("elevenlabs", "partial", t)

    def on_committed(data):
        t = data.get("text", "") if isinstance(data, dict) else str(data)
        result_callback("elevenlabs", "status", f"→ committed_transcript: {t!r}")
        if t:
            result_callback("elevenlabs", "final", t)
        done_event.set()

    def on_insufficient(_):
        result_callback("elevenlabs", "status", "→ insufficient_audio_activity (no speech detected)")
        done_event.set()

    def on_hard_error(data):
        msg = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        # Only hard errors (auth, quota, rate limit) reach here after filtering
        result_callback("elevenlabs", "error", msg)
        done_event.set()

    if debug:
        # Register ALL events so we can see everything that comes back
        for evt in RealtimeEvents:
            connection.on(evt, lambda d, e=evt: result_callback("elevenlabs", "status", f"[evt] {e.value}: {str(d)[:80]}"))

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT,          on_partial)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT,        on_committed)
    connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, on_insufficient)
    connection.on(RealtimeEvents.AUTH_ERROR,                  on_hard_error)
    connection.on(RealtimeEvents.QUOTA_EXCEEDED,              on_hard_error)
    connection.on(RealtimeEvents.RATE_LIMITED,                on_hard_error)

    # Stream audio until stop_event
    chunks_sent = 0
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.3)
            await connection.send({"audio_base_64": base64.b64encode(chunk).decode("utf-8")})
            chunks_sent += 1
        except asyncio.TimeoutError:
            continue

    result_callback("elevenlabs", "status", f"→ streamed {chunks_sent} chunks, committing…")

    # Drain remaining queue chunks
    while not audio_queue.empty():
        try:
            chunk = audio_queue.get_nowait()
            await connection.send({"audio_base_64": base64.b64encode(chunk).decode("utf-8")})
        except asyncio.QueueEmpty:
            break

    # Explicit commit — triggers COMMITTED_TRANSCRIPT on server side
    await connection.commit()

    result_callback("elevenlabs", "status", "→ waiting for server response (up to 8s)…")
    try:
        await asyncio.wait_for(done_event.wait(), timeout=8.0)
    except asyncio.TimeoutError:
        result_callback("elevenlabs", "status", "⏱ TIMEOUT — no response after 8s (check API key / audio)")

    await connection.close()


# ── Display ────────────────────────────────────────────────────────────────────

class Display:
    """Thread-safe terminal display for real-time transcription."""

    def __init__(self, providers: list[str]):
        self._lock = threading.Lock()
        self._providers = providers
        self._partials: dict[str, str] = {p: "" for p in providers}
        self._finals:   list[tuple[str, str]] = []

    def update(self, provider: str, kind: str, text: str):
        with self._lock:
            label = "OpenAI  " if provider == "openai" else "ElevenLabs"

            if kind == "partial":
                self._partials[provider] = text
                print(f"\r  [{label}] ⏳ {text[:70]:<70}", end="", flush=True)

            elif kind == "final":
                self._partials[provider] = ""
                self._finals.append((provider, text))
                print(f"\r  [{label}] ✓  {text}")
                print(f"  {'─'*72}")

            elif kind == "status":
                print(f"\r  [{label}] {text:<74}")

            elif kind == "error":
                print(f"\r  [{label}] ✗ ERROR: {text}")

    def summary(self):
        print(f"\n\n  {'═'*72}")
        print("  TRANSCRIPTION SUMMARY")
        print(f"  {'═'*72}")
        if not self._finals:
            print("  (no transcripts received)")
        for provider, text in self._finals:
            label = "OpenAI  " if provider == "openai" else "ElevenLabs"
            print(f"  [{label}] {text}")
        print(f"  {'═'*72}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

async def run_live_test(
    api_key_openai: str,
    api_key_elevenlabs: str,
    use_openai: bool,
    use_elevenlabs: bool,
    duration: float | None,
    device: int | None,
    debug: bool = False,
):
    active_providers = []
    if use_openai:     active_providers.append("openai")
    if use_elevenlabs: active_providers.append("elevenlabs")

    display = Display(active_providers)

    # Separate queues so each provider gets all audio independently
    queues: dict[str, asyncio.Queue] = {p: asyncio.Queue() for p in active_providers}
    stop_event = asyncio.Event()

    # Mic recorder runs in a thread, pushes to asyncio queues
    recorder = MicRecorder()
    loop     = asyncio.get_running_loop()

    def mic_feeder():
        while not stop_event.is_set():
            chunk = recorder.get_chunk(timeout=0.2)
            if chunk:
                for q in queues.values():
                    loop.call_soon_threadsafe(q.put_nowait, chunk)
        # Drain leftovers
        remaining = recorder.drain()
        if remaining:
            for i in range(0, len(remaining), 3200):
                c = remaining[i:i+3200]
                for q in queues.values():
                    loop.call_soon_threadsafe(q.put_nowait, c)

    # ── Print instructions ─────────────────────────────────────────────
    print("=" * 74)
    print("  Live Mic → Realtime STT Test")
    print(f"  Providers: {', '.join(active_providers)}")
    print(f"  Sample rate: {SAMPLE_RATE}Hz  |  Chunk: {int(CHUNK_DURATION*1000)}ms")
    print("=" * 74)

    if duration:
        print(f"\n  Recording for {duration}s automatically...\n")
        print("  Speak now!\n")
        print(f"  {'─'*72}")
    else:
        print("\n  Press ENTER to start recording, ENTER again to stop.\n")
        input("  → Press ENTER to start...")
        print()
        print(f"  {'─'*72}")
        print("  🎤 Recording... press ENTER to stop\n")

    # ── Start providers ────────────────────────────────────────────────
    provider_tasks = []
    if use_openai:
        provider_tasks.append(asyncio.create_task(
            run_openai_realtime(api_key_openai, queues["openai"], stop_event,
                                lambda p, k, t: display.update(p, k, t), debug=debug)
        ))
    if use_elevenlabs:
        provider_tasks.append(asyncio.create_task(
            run_elevenlabs_realtime(api_key_elevenlabs, queues["elevenlabs"], stop_event,
                                    lambda p, k, t: display.update(p, k, t), debug=debug)
        ))

    # ── Start mic ──────────────────────────────────────────────────────
    recorder.start(device=device)
    feeder_thread = threading.Thread(target=mic_feeder, daemon=True)
    feeder_thread.start()

    # ── Wait for stop ──────────────────────────────────────────────────
    if duration:
        await asyncio.sleep(duration)
    else:
        # Non-blocking ENTER wait
        await asyncio.get_event_loop().run_in_executor(None, input)
        print("\n  🔇 Stopped recording, flushing...\n")

    # ── Stop ───────────────────────────────────────────────────────────
    recorder.stop()
    stop_event.set()
    feeder_thread.join(timeout=2)

    # Wait for providers to flush + finish
    await asyncio.gather(*provider_tasks, return_exceptions=True)

    display.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai",      action="store_true", help="Use OpenAI Realtime only")
    parser.add_argument("--elevenlabs",  action="store_true", help="Use ElevenLabs Realtime only")
    parser.add_argument("--duration",    type=float, default=None,
                        help="Auto-record for N seconds (default: manual ENTER)")
    parser.add_argument("--device",      type=int, default=None,
                        help="Mic device index (run with --list-devices to see options)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--debug",        action="store_true", help="Print all raw WebSocket events")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable input devices:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  {i}: {d['name']}  (ch={d['max_input_channels']}, sr={int(d['default_samplerate'])}Hz)")
        return

    use_openai     = args.openai or (not args.openai and not args.elevenlabs)
    use_elevenlabs = args.elevenlabs or (not args.openai and not args.elevenlabs)

    from backend.config import settings
    oai_key = settings.openai_api_key
    el_key  = settings.elevenlabs_api_key

    asyncio.run(run_live_test(
        api_key_openai=oai_key,
        api_key_elevenlabs=el_key,
        use_openai=use_openai,
        use_elevenlabs=use_elevenlabs,
        duration=args.duration,
        device=args.device,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
