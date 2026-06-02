"""
Realtime STT + Speaker Classifier Pipeline Test
================================================
Compares:
  1. OpenAI Realtime Transcription  (gpt-4o-mini-transcribe, WebSocket)
  2. ElevenLabs Scribe Realtime     (scribe_v2_realtime,     WebSocket)

After each STT result, runs the transcript through the speaker classifier
(gpt-4o-mini) to test the full realtime pipeline end-to-end.

Parts:
  PART 1 — Single utterance: STT accuracy + speaker classification
  PART 2 — Full 5-turn conversation: does accuracy improve with history?
  PART 3 — Edge cases: silence, overlap, short reactions

Usage:
    python test_realtime_stt_classifier.py
    python test_realtime_stt_classifier.py --openai-only    # skip ElevenLabs
    python test_realtime_stt_classifier.py --eleven-only    # skip OpenAI
    python test_realtime_stt_classifier.py --no-classifier  # skip classifier
"""

import asyncio
import argparse
import base64
import io
import json
import subprocess
import time
import wave
import struct

# ── Audio helpers ──────────────────────────────────────────────────────────────

async def generate_tts(text: str, api_key: str) -> bytes:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.audio.speech.create(
        model="tts-1", voice="nova", input=text, response_format="mp3",
    )
    return resp.content


def to_pcm16k(audio_bytes: bytes, fmt: str = "mp3") -> bytes:
    """Convert MP3 or WAV → raw PCM s16le 16kHz mono via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0",
         "-f", "s16le", "-ar", "16000", "-ac", "1",
         "pipe:1", "-loglevel", "quiet"],
        input=audio_bytes, capture_output=True,
    )
    return result.stdout


def silence_pcm(duration_secs: float = 0.6) -> bytes:
    """Generate raw PCM silence (used to flush VAD)."""
    n_samples = int(16000 * duration_secs)
    return bytes(n_samples * 2)  # 16-bit = 2 bytes per sample


def near_silence_wav(duration_secs: float = 2.0) -> bytes:
    """WAV with very low amplitude noise — tests hallucination resistance."""
    import random
    buf = io.BytesIO()
    n = int(16000 * duration_secs)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n}h", *[random.randint(-30, 30) for _ in range(n)]))
    return buf.getvalue()


# ── OpenAI Realtime Transcription ─────────────────────────────────────────────

async def transcribe_openai_realtime(
    pcm_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o-mini-transcribe",
    debug: bool = False,
) -> tuple[str, float]:
    """
    Stream PCM audio to OpenAI Realtime transcription API (WebSocket).
    wss://api.openai.com/v1/realtime?intent=transcription
    """
    import websockets

    url = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v2",
    }

    transcript = ""
    t0 = time.perf_counter()
    done_event = asyncio.Event()

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:

            # Step 1: configure session
            await ws.send(json.dumps({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": model,
                        "language": "id",
                        "prompt": (
                            "Percakapan di showroom mobil Mitsubishi Indonesia. "
                            "Xpander, Pajero, Xforce, Outlander, kredit, DP, cicilan."
                        ),
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": 600,
                        "threshold": 0.5,
                    },
                },
            }))

            # Step 2: receive task (runs concurrently with streaming)
            async def receive_loop():
                nonlocal transcript
                try:
                    async for raw in ws:
                        evt = json.loads(raw)
                        etype = evt.get("type", "")
                        if debug:
                            print(f"        [OpenAI RT evt] {etype}")

                        if etype == "conversation.item.input_audio_transcription.completed":
                            transcript = evt.get("transcript", "")
                            done_event.set()
                            return
                        elif etype == "conversation.item.input_audio_transcription.delta":
                            transcript += evt.get("delta", "")
                        elif etype == "error":
                            if debug:
                                print(f"        [OpenAI RT error] {evt.get('error')}")
                            done_event.set()
                            return
                except Exception:
                    done_event.set()

            recv_task = asyncio.create_task(receive_loop())

            # Step 3: stream audio chunks (simulate mic pace)
            chunk_size = 4096  # ~128ms at 16kHz 16-bit
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("utf-8"),
                }))
                await asyncio.sleep(0.02)

            # Send trailing silence to trigger VAD speech_stop
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(silence_pcm(0.8)).decode("utf-8"),
            }))

            # Commit the buffer explicitly
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # Step 4: wait for transcript (up to 10s)
            try:
                await asyncio.wait_for(done_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                if debug:
                    print("        [OpenAI RT] Timeout — partial:", repr(transcript))

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"    [OpenAI RT] Connection error: {type(e).__name__}: {e}")

    return transcript.strip(), time.perf_counter() - t0


# ── ElevenLabs Scribe Realtime ─────────────────────────────────────────────────

async def transcribe_elevenlabs_realtime(
    pcm_bytes: bytes,
    api_key: str,
    debug: bool = False,
) -> tuple[str, float]:
    """
    Stream PCM audio to ElevenLabs Scribe Realtime WebSocket.
    Sends 600ms of silence after the audio to help VAD commit.
    """
    from elevenlabs.realtime.scribe import (
        ScribeRealtime, AudioFormat, CommitStrategy, RealtimeAudioOptions,
    )
    from elevenlabs.realtime.connection import RealtimeEvents

    committed: list[str] = []
    partials:  list[str] = []
    done_event = asyncio.Event()
    t0 = time.perf_counter()

    scribe = ScribeRealtime(api_key=api_key)
    options: RealtimeAudioOptions = {
        "model_id":               "scribe_v2_realtime",
        "audio_format":           AudioFormat.PCM_16000,
        "sample_rate":            16000,
        "language_code":          "id",
        "commit_strategy":        CommitStrategy.MANUAL,   # explicit commit — no VAD timing issues
        "min_speech_duration_ms": 100,
        "keyterms": ["Xpander", "Pajero", "Xforce", "Outlander", "Mitsubishi", "kredit", "DP"],
    }

    connection = await scribe.connect(options)

    def on_partial(data):
        t = data.get("text", "") if isinstance(data, dict) else ""
        if t:
            partials.append(t)
            if debug:
                print(f"        [EL partial] {t!r}")

    def on_committed(data):
        t = data.get("text", "") if isinstance(data, dict) else ""
        if t:
            committed.append(t)
        if debug:
            print(f"        [EL committed] {t!r}")
        done_event.set()

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT,          on_partial)
    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT,        on_committed)
    connection.on(RealtimeEvents.INSUFFICIENT_AUDIO_ACTIVITY, lambda _: done_event.set())

    # Stream audio in 4096-byte chunks (~128ms each)
    chunk_size = 4096
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i + chunk_size]
        await connection.send({"audio_base_64": base64.b64encode(chunk).decode("utf-8")})
        await asyncio.sleep(0.02)

    # Explicit commit after all audio sent
    await connection.commit()

    try:
        await asyncio.wait_for(done_event.wait(), timeout=8.0)
    except asyncio.TimeoutError:
        if debug:
            print("        [EL] Timeout — partials so far:", partials)

    await connection.close()

    # Prefer committed, fall back to last partial
    text = (committed[-1] if committed else partials[-1] if partials else "").strip()
    return text, time.perf_counter() - t0


# ── Speaker Classifier ─────────────────────────────────────────────────────────

async def classify_speaker(
    text: str,
    history: list[dict],   # [{"speaker": "sales"|"customer", "text": "..."}]
    last_speaker: str,
    api_key: str,
) -> tuple[str, float, float]:
    """Returns (speaker, confidence, latency_secs)."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    clean = text.strip()
    # Short-text heuristic (mirror of production classifier)
    if len(clean.split()) <= 4:
        return last_speaker or "unknown", 0.4, 0.0

    history_text = "\n".join(
        f"{'SALES' if u['speaker'] == 'sales' else 'CUSTOMER'}: {u['text']}"
        for u in history
    ) or "(belum ada percakapan)"

    last_label = {"sales": "SALES", "customer": "CUSTOMER"}.get(last_speaker, "tidak diketahui")

    user_prompt = f"""Riwayat percakapan terakhir:
{history_text}

Pembicara terakhir yang diketahui: {last_label}

Kalimat baru yang harus diklasifikasikan:
"{clean}"

Berikan confidence RENDAH (< 0.5) jika kalimat ambigu, kemungkinan overlap, atau konteks belum cukup."""

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=30,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu menganalisis percakapan di showroom mobil Mitsubishi Indonesia. "
                    "Dua pembicara: SALES (staf dealer) dan CUSTOMER (calon pembeli).\n"
                    "SALES: menjelaskan produk/fitur/harga, bertanya kebutuhan, menawarkan, menyebut 'kami'.\n"
                    "CUSTOMER: bercerita kebutuhan/keluarga/situasi, menjawab pertanyaan, menanyakan produk/harga.\n"
                    "PENTING: Jika teks mengandung ucapan DUA orang sekaligus, klasifikasikan berdasarkan BAGIAN TERAKHIR. "
                    "Confidence rendah (< 0.5) untuk kasus overlap.\n"
                    'Jawab JSON: {"speaker":"sales","confidence":0.85}'
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    lat = time.perf_counter() - t0
    data = json.loads(resp.choices[0].message.content.strip())
    spk  = data.get("speaker", "unknown")
    conf = float(data.get("confidence", 0.5))
    if spk not in ("sales", "customer"):
        spk = "unknown"
    return spk, conf, lat


# ── Print helpers ──────────────────────────────────────────────────────────────

W = 80

def hdr(title: str):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print("─" * W)

def stt_line(provider: str, transcript: str, stt_lat: float):
    status = "✓" if transcript else "✗ (empty)"
    t = (transcript[:65] + "…") if len(transcript) > 65 else transcript or "(no transcript)"
    print(f"  {status}  [{provider:<28}] {stt_lat:5.2f}s  │  \"{t}\"")

def cls_line(expected: str, got: str, conf: float, cls_lat: float):
    bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
    ok  = "✓ correct" if got == expected else f"✗ WRONG (expected {expected.upper()})"
    print(f"            → classifier: {got.upper():<8} conf={conf:.2f} [{bar}] +{cls_lat:.2f}s  {ok}")


# ── Test data ──────────────────────────────────────────────────────────────────

SINGLE = [
    ("sales",    "A. Sales greeting",
     "Selamat pagi, selamat datang di Mitsubishi! Ada yang bisa saya bantu?"),
    ("customer", "B. Customer query",
     "Halo, saya mau tanya harga Pajero Sport Dakar warna hitam ada tidak?"),
    ("sales",    "C. Sales offering",
     "Untuk Xforce Ultimate, kami bisa bantu proses kreditnya dengan DP mulai 30 juta."),
    ("customer", "D. Customer situation",
     "Saya biasanya bawa keluarga 5 orang, sering ke luar kota juga jalan kurang bagus."),
]

# 6-turn conversation — tests classifier accuracy as history grows
CONVERSATION = [
    ("sales",    "Selamat datang, ada yang bisa saya bantu?"),
    ("customer", "Halo, saya mau lihat Xpander. Ada unit warna putih?"),
    ("sales",    "Ada Pak, Xpander Ultimate tersedia warna white pearl. Mau kredit atau cash?"),
    ("customer", "Kredit. Kira-kira DP berapa untuk Xpander Ultimate tenor 5 tahun?"),
    ("sales",    "DP mulai 35 juta, cicilan sekitar 4,2 juta per bulan sudah termasuk asuransi."),
    ("customer", "Boleh test drive dulu? Saya bawa istri sekalian."),
]

EDGE_CASES = [
    ("customer", "E. Short reaction (≤4 words)",
     "Oh iya, berapa?"),
    ("customer", "F. Overlap/merge",
     "Berapa harganya? Sekitar 600 juta Pak untuk varian Dakar."),
    (None,       "G. Near-silence (hallucination test)",
     None),  # uses generated silence WAV
]


# ── Main ────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai-only",    action="store_true")
    parser.add_argument("--eleven-only",    action="store_true")
    parser.add_argument("--no-classifier",  action="store_true")
    parser.add_argument("--debug",          action="store_true", help="Show raw WebSocket events")
    args = parser.parse_args()

    use_openai  = not args.eleven_only
    use_eleven  = not args.openai_only
    use_cls     = not args.no_classifier

    from backend.config import settings
    oai_key = settings.openai_api_key
    el_key  = settings.elevenlabs_api_key

    providers = []
    if use_openai: providers.append(("OpenAI Realtime",    lambda p: transcribe_openai_realtime(p, oai_key, debug=args.debug)))
    if use_eleven: providers.append(("ElevenLabs Realtime", lambda p: transcribe_elevenlabs_realtime(p, el_key, debug=args.debug)))

    print("=" * W)
    print("  Realtime STT + Speaker Classifier — Pipeline Test")
    print(f"  Providers: {', '.join(n for n,_ in providers)}")
    print(f"  Classifier: {'gpt-4o-mini' if use_cls else 'disabled'}")
    print("=" * W)

    # ════════════════════════════════════════════════════════════════════
    print("\n\n══ PART 1: Single Utterances — STT Accuracy + Classification ══════════════")

    for expected_spk, label, text in SINGLE:
        hdr(f"{label}  │  expected speaker: {expected_spk.upper()}")
        print(f"  Text: \"{text}\"\n")

        mp3 = await generate_tts(text, oai_key)
        pcm = to_pcm16k(mp3)
        print(f"  Audio: {len(mp3):,}B MP3 / {len(pcm):,}B PCM\n")

        tasks = [asyncio.create_task(fn(pcm)) for _, fn in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (pname, _), res in zip(providers, results):
            if isinstance(res, Exception):
                print(f"  ✗  [{pname:<28}] ERROR: {res}")
                continue
            transcript, stt_lat = res
            stt_line(pname, transcript, stt_lat)
            if use_cls and transcript:
                spk, conf, cls_lat = await classify_speaker(transcript, [], "unknown", oai_key)
                cls_line(expected_spk, spk, conf, cls_lat)

    # ════════════════════════════════════════════════════════════════════
    print(f"\n\n══ PART 2: Full Conversation (6 turns) — History Effect ════════════════════")
    print("  Tests whether classifier accuracy improves as conversation history grows\n")

    for pname, fn in providers:
        print(f"\n  ── {pname} ──────────────────────────────────────────────")
        history: list[dict] = []
        last_spk = "unknown"
        correct = 0
        total   = 0

        for i, (expected_spk, text) in enumerate(CONVERSATION, 1):
            mp3 = await generate_tts(text, oai_key)
            pcm = to_pcm16k(mp3)

            transcript, stt_lat = await fn(pcm)
            t_short = (transcript[:52] + "…") if len(transcript) > 52 else transcript

            if not transcript:
                print(f"  {i}.  ✗ (empty)   expected={expected_spk.upper()}  stt={stt_lat:.2f}s")
                total += 1
                continue

            if use_cls:
                spk, conf, cls_lat = await classify_speaker(
                    transcript, history[-4:], last_spk, oai_key
                )
                ok = "✓" if spk == expected_spk else "✗"
                if spk == expected_spk:
                    correct += 1
                total += 1
                bar  = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                print(f"  {i}. [{ok}] {expected_spk.upper():<8} → {spk.upper():<8} conf={conf:.2f} [{bar}]  stt={stt_lat:.2f}s +cls={cls_lat:.2f}s")
                print(f"       \"{t_short}\"")
                history.append({"speaker": spk, "text": transcript})
                last_spk = spk
            else:
                print(f"  {i}.  stt={stt_lat:.2f}s  \"{t_short}\"")

        if use_cls and total:
            pct = correct / total * 100
            bar = "█" * correct + "░" * (total - correct)
            print(f"\n  Accuracy: {correct}/{total} ({pct:.0f}%)  [{bar}]")

    # ════════════════════════════════════════════════════════════════════
    print(f"\n\n══ PART 3: Edge Cases ══════════════════════════════════════════════════════")

    for expected_spk, label, text in EDGE_CASES:
        hdr(label)

        if text is None:
            # Silence / hallucination test
            print("  Source: generated near-silence WAV (2s, amplitude ≤ 30)\n")
            silence_wav = near_silence_wav(2.0)
            pcm = to_pcm16k(silence_wav, fmt="wav")
        else:
            print(f"  Text: \"{text}\"\n")
            mp3 = await generate_tts(text, oai_key)
            pcm = to_pcm16k(mp3)

        tasks = [asyncio.create_task(fn(pcm)) for _, fn in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (pname, _), res in zip(providers, results):
            if isinstance(res, Exception):
                print(f"  ✗  [{pname:<28}] ERROR: {res}")
                continue
            transcript, stt_lat = res

            if text is None:
                # Hallucination check
                verdict = "HALLUCINATION ✗" if transcript else "clean (empty) ✓"
                stt_line(pname, transcript or "(empty)", stt_lat)
                print(f"            → {verdict}")
            else:
                stt_line(pname, transcript, stt_lat)
                if use_cls and transcript and expected_spk:
                    spk, conf, cls_lat = await classify_speaker(transcript, [], "unknown", oai_key)
                    cls_line(expected_spk, spk, conf, cls_lat)

    # ════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * W}")
    print("  KEY METRICS TO COMPARE:")
    print("  STT latency        — time from audio start to transcript received")
    print("  Accuracy           — does the transcript match the expected phrase?")
    print("  Classifier correct — does speaker label match expected?")
    print("  Edge: silence      — empty output = good (no hallucination)")
    print("  Edge: overlap      — low confidence + correct last-speaker fallback")
    print("=" * W)


if __name__ == "__main__":
    asyncio.run(main())
