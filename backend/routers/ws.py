"""
WebSocket Router — /ws/session/{session_id}

Flow per pesan:
  audio_chunk  → STT → Speaker classify → AI analyze → kirim hint + rekomendasi
  session_end  → simpan log sesi
"""
import time
import json
import base64
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.session import session_manager
from backend.services.stt import transcribe_audio
from backend.services.ai import analyze_conversation
from backend.services.tts import synthesize
from backend.services.speaker_classifier import classify_speaker
from backend.services.rag import search_conversation_patterns
from backend.services.elicitation import compute_missing_dimensions, get_next_question
from backend.services.topic_patterns import (
    detect_topic,
    search_topic_transitions,
    pick_transition_question,
    TOPIC_TO_ELICITATION_DIM,
)
from backend.models.schemas import WsMessageType
from backend.services.metrics import (
    pipeline_latency, sesi_aktif, sesi_selesai,
    ws_connect_total, ws_disconnect_total,
    roundtrip_latency,
    conversation_stage_total, ai_busy_skip_total,
    silence_watchdog_total, deflection_total, filler_filtered_total,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Per-session flag: True kalau AI sedang dianalisis — skip (jangan queue)
_ai_busy: dict[str, bool] = {}

# Per-session silence watchdog task
_silence_task: dict[str, asyncio.Task | None] = {}
SILENCE_THRESHOLD_SEC = 8

FILLER_ONLY = {
    "mm", "mmm", "eh", "ehm", "uh", "uhm",
    "hmm", "hm", "ah", "oh", "huh",
    "eee", "aaa", "umm",
}


def _cancel_silence_timer(session_id: str) -> None:
    task = _silence_task.pop(session_id, None)
    if task and not task.done():
        task.cancel()


def _reset_silence_timer(session_id: str, context) -> None:
    _cancel_silence_timer(session_id)
    _silence_task[session_id] = asyncio.create_task(
        _silence_watchdog(session_id, context)
    )


async def _silence_watchdog(session_id: str, context) -> None:
    """Trigger fallback hint jika tidak ada utterance baru selama SILENCE_THRESHOLD_SEC."""
    try:
        await asyncio.sleep(SILENCE_THRESHOLD_SEC)
        if _ai_busy.get(session_id):
            return
        logger.info(f"[WS] Silence {SILENCE_THRESHOLD_SEC}s terdeteksi pada {session_id} — trigger fallback hint")
        silence_watchdog_total.inc()
        await _rag_fallback_hint(session_id, context)
    except asyncio.CancelledError:
        pass


@router.websocket("/ws/session/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    await session_manager.connect(websocket, session_id)
    _ai_busy[session_id] = False
    _silence_task[session_id] = None
    sesi_aktif.inc()
    ws_connect_total.inc()
    _session_ended_cleanly = False

    await session_manager.send(session_id, "connected", {
        "session_id": session_id,
        "message": "Sesi dimulai. Sistem siap mendengarkan.",
    })

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")
            payload  = msg.get("payload", {})

            if msg_type == WsMessageType.AUDIO_CHUNK:
                # Background task — WS tidak block menunggu STT/AI/TTS selesai
                asyncio.create_task(_handle_audio(session_id, payload))

            elif msg_type == WsMessageType.SESSION_END:
                await _handle_session_end(session_id)
                _session_ended_cleanly = True
                break

            elif msg_type == WsMessageType.PING:
                await session_manager.send(session_id, "pong", {})

            else:
                logger.warning(f"[WS] Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: {session_id}")
        ws_disconnect_total.labels(reason="unexpected").inc()
    except Exception as e:
        logger.error(f"[WS] Error on session {session_id}: {e}", exc_info=True)
        await session_manager.send(session_id, WsMessageType.ERROR, {"message": str(e)})
        ws_disconnect_total.labels(reason="unexpected").inc()
    finally:
        ws_disconnect_total.labels(reason="clean").inc() if _session_ended_cleanly else None
        _ai_busy.pop(session_id, None)
        _cancel_silence_timer(session_id)
        sesi_aktif.dec()
        session_manager.disconnect(session_id)


async def _handle_audio(session_id: str, payload: dict):
    context = session_manager.get_context(session_id)
    if context is None:
        return

    audio_b64   = payload.get("audio", "")
    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

    # 1. STT — harus selesai dulu sebelum langkah berikutnya
    utterance = await transcribe_audio(audio_bytes, session_id)
    if not utterance:
        return

    # 2. Parallel: classify speaker + analyze conversation (pakai context SEBELUM utterance baru)
    #    analyze_conversation tidak perlu tahu siapa yang ngomong terbaru —
    #    dia pakai RAG conversation patterns untuk bridge the gap.
    ai_skipped = _ai_busy.get(session_id)
    if not ai_skipped:
        _ai_busy[session_id] = True

    pipeline_start = time.perf_counter()

    classify_task = asyncio.create_task(
        classify_speaker(utterance.text, context.utterances[-5:], context.last_speaker)
    )
    analyze_task = (
        asyncio.create_task(analyze_conversation(context))
        if not ai_skipped else None
    )

    # Tunggu classify selesai (~200-400ms), analyze jalan di background
    speaker, confidence = await classify_task
    utterance.speaker    = speaker
    utterance.confidence = confidence
    if confidence >= 0.6 and speaker in ("sales", "customer"):
        context.last_speaker = speaker

    # 3. Kirim transcript ke frontend segera setelah classify selesai
    words = utterance.text.lower().split()
    is_filler_only = bool(words) and all(w in FILLER_ONLY for w in words)
    is_overlap_fragment = is_filler_only or (confidence < 0.4 and len(words) <= 4)
    context.utterances.append(utterance)
    await session_manager.send(session_id, WsMessageType.TRANSCRIPT, {
        "utterance": utterance.model_dump(mode="json")
    })
    snapshot_count = len(context.utterances)

    if is_overlap_fragment:
        reason = "filler-only" if is_filler_only else "low-confidence fragment"
        logger.info(f"[WS] Overlap dideteksi ({reason}), skip AI: '{utterance.text}'")
        if is_filler_only:
            filler_filtered_total.inc()
        if analyze_task:
            analyze_task.cancel()
            _ai_busy[session_id] = False
        return

    if utterance.speaker == "customer":
        _reset_silence_timer(session_id, context)

    if ai_skipped:
        logger.info(f"[WS] AI masih sibuk, fallback ke RAG untuk utterance {snapshot_count}")
        ai_busy_skip_total.inc()
        asyncio.create_task(_rag_fallback_hint(session_id, context))
        return

    # 4. Tunggu analyze selesai (sudah jalan paralel sejak STT selesai)
    try:
        hint_payload, car_payload = await analyze_task
        pipeline_latency.observe(time.perf_counter() - pipeline_start)
        conversation_stage_total.labels(tahap=hint_payload.tahap).inc()

        context.detected_needs      = hint_payload.detected_needs
        context.recommended_car_ids = [c.id for c in car_payload.cars]
        context.asked_questions.append(hint_payload.suggested_question)

        # Catat dimensi yang baru saja customer hindari agar tidak ditanya lagi
        if hint_payload.blocked_dimension and hint_payload.blocked_dimension not in context.blocked_dimensions:
            context.blocked_dimensions.append(hint_payload.blocked_dimension)
            deflection_total.labels(dimension=hint_payload.blocked_dimension).inc()
            logger.info(f"[WS] Dimensi '{hint_payload.blocked_dimension}' di-block untuk sesi {session_id}")

        # 4. Mulai TTS di background, paralel dengan send hint + car recs
        tts_task = asyncio.create_task(synthesize(hint_payload.suggested_question))

        await session_manager.send(session_id, WsMessageType.AI_HINT, {
            "hint_text":          hint_payload.hint_text,
            "suggested_question": hint_payload.suggested_question,
            "probe_topics":       hint_payload.probe_topics,
            "detected_needs":     hint_payload.detected_needs,
            "question_source":    hint_payload.question_source,
        })
        await session_manager.send(session_id, WsMessageType.CAR_RECOMMEND, {
            "cars":   [c.model_dump(mode="json") for c in car_payload.cars],
            "reason": car_payload.reason,
        })

        # 5. Tunggu TTS — lalu cek apakah context masih relevan
        tts_bytes     = await tts_task
        current_count = len(context.utterances)

        # Jangan putar TTS kalau percakapan sudah maju ≥ 2 utterance
        if tts_bytes and current_count - snapshot_count < 2:
            await session_manager.send(session_id, WsMessageType.TTS_AUDIO, {
                "audio":  base64.b64encode(tts_bytes).decode("utf-8"),
                "format": "mp3",
                "text":   hint_payload.suggested_question,
            })
            roundtrip_latency.observe(time.perf_counter() - pipeline_start)
        elif tts_bytes:
            logger.info(
                f"[WS] TTS dibuang — context sudah maju "
                f"{current_count - snapshot_count} utterance sejak analisis"
            )

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"[WS] AI/TTS error on {session_id}: {e}", exc_info=True)
    finally:
        _ai_busy[session_id] = False


async def _rag_fallback_hint(session_id: str, context) -> None:
    """
    Kirim hint saat LLM sedang sibuk.

    Prioritas:
      1. Topic transitions — query historis "dari topik ini, lanjut ke mana?"
      2. RAG sequence patterns — sequence similarity pada conversation-patterns
      3. Elicitation — dimensi yang belum tergali, di-bias oleh topik saat ini
    """
    full_text   = " ".join(u.text for u in context.utterances)
    recent_text = " ".join(u.text for u in context.utterances[-5:])
    question    = ""
    source      = ""
    hint_text   = "Pertanyaan dari pola percakapan serupa."

    # 1. Topic transitions
    current_topic = detect_topic(" ".join(u.text for u in context.utterances[-3:]))
    if current_topic:
        transitions = await search_topic_transitions(current_topic, context.utterances)
        question, to_topic = pick_transition_question(transitions, context.asked_questions)
        if question:
            source = "topic_transition"
            dim = TOPIC_TO_ELICITATION_DIM.get(to_topic or "")
            hint_text = (
                f"Arahkan ke topik {to_topic}."
                + (f" Gali dimensi {dim}." if dim else "")
            )

    # 2. RAG sequence patterns
    if not question:
        patterns  = await search_conversation_patterns(recent_text, top_k=5)
        preferred = [
            p for p in patterns
            if p.get("outcome") != "question_deflected"
            and p.get("effective_next_question", "") not in context.asked_questions
        ]
        candidate = (
            preferred
            or [p for p in patterns if p.get("effective_next_question", "") not in context.asked_questions]
            or [None]
        )[0]
        question = candidate.get("effective_next_question", "") if candidate else ""
        if question:
            source = "rag"

    # 3. Elicitation — prioritaskan dimensi yang relevan dengan topik saat ini
    if not question:
        missing = compute_missing_dimensions(full_text)
        if current_topic:
            dim = TOPIC_TO_ELICITATION_DIM.get(current_topic)
            if dim and dim in missing:
                missing = [dim] + [d for d in missing if d != dim]
        q_dict = get_next_question(missing, context.asked_questions)
        if not q_dict:
            return
        question  = q_dict["question"]
        source    = "elicitation"
        hint_text = f"Gali {' & '.join(q_dict.get('reveals', ['informasi']))} dari customer."

    context.asked_questions.append(question)
    logger.info(f"[WS] Fallback hint ({source}): '{question[:80]}'")

    await session_manager.send(session_id, WsMessageType.AI_HINT, {
        "hint_text":          hint_text,
        "suggested_question": question,
        "probe_topics":       [],
        "detected_needs":     context.detected_needs,
        "tahap":              "RAG_FALLBACK",
    })


async def _handle_session_end(session_id: str):
    context = session_manager.get_context(session_id)
    total = len(context.utterances) if context else 0
    outcome = "rekomendasi_diberikan" if context and context.recommended_car_ids else "tidak_ada_minat"
    sesi_selesai.labels(outcome=outcome).inc()
    logger.info(f"[WS] Session ended: {session_id} | {total} utterances | {outcome}")
    await session_manager.send(session_id, "session_summary", {
        "total_utterances": total,
        "detected_needs":   context.detected_needs if context else [],
        "recommended_cars": context.recommended_car_ids if context else [],
    })
