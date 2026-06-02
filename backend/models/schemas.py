from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime, timezone


# ── Percakapan ──────────────────────────────────────────

class Utterance(BaseModel):
    """Satu segmen ucapan yang sudah di-transcribe."""
    id: str
    speaker: Literal["sales", "customer", "unknown"]
    text: str
    timestamp: datetime
    confidence: float = 1.0


class ConversationContext(BaseModel):
    """Akumulasi percakapan dalam satu sesi."""
    session_id: str
    utterances: list[Utterance] = []
    detected_needs: list[str] = []
    recommended_car_ids: list[str] = []
    asked_questions: list[str] = []       # pertanyaan yang sudah pernah disuggest ke sales
    blocked_dimensions: list[str] = []    # dimensi yang customer hindari — jangan tanya lagi
    last_speaker: Literal["sales", "customer", "unknown"] = "unknown"
    current_tahap: str = "PEMBUKA"        # tahap terakhir yang dideteksi LLM


# ── Mobil ────────────────────────────────────────────────

class CarColor(BaseModel):
    name: str
    hex: str


class CarSpec(BaseModel):
    id: str
    brand: str
    model: str
    variant: str
    year: int
    type: Literal["MPV", "SUV", "SUV MPV", "Sedan", "Hatchback", "Pickup Double Cabin"]
    seats: int
    price_otr_jakarta: int           # dalam rupiah
    engine_cc: int
    horsepower: int
    fuel_consumption_kml: float
    wheel_size_inch: int
    colors: list[CarColor]
    features: list[str]
    radar: dict[str, int]            # kenyamanan, performa, efisiensi, keamanan, kapasitas


# ── WebSocket messages ───────────────────────────────────

class WsMessageType(str):
    # Client → Server
    AUDIO_CHUNK   = "audio_chunk"
    SESSION_START = "session_start"
    SESSION_END   = "session_end"

    # Server → Client
    TRANSCRIPT    = "transcript"
    AI_HINT       = "ai_hint"
    CAR_RECOMMEND = "car_recommend"
    TTS_AUDIO     = "tts_audio"
    ERROR         = "error"
    PING          = "ping"


class WsIncoming(BaseModel):
    type: str
    payload: dict = {}


class TranscriptPayload(BaseModel):
    utterance: Utterance


class AiHintPayload(BaseModel):
    hint_text: str
    suggested_question: str       # pertanyaan utama pendek — untuk TTS
    probe_topics: list[str] = []  # topik singkat yang perlu digali sales (bullet points)
    detected_needs: list[str]
    tahap: str = ""               # tahap percakapan yang dideteksi LLM — untuk logging/debugging
    blocked_dimension: str = ""   # dimensi yang baru saja customer hindari (kosong = tidak ada)
    question_source: str = ""     # "pattern" = dari RAG, "generated" = LLM buat sendiri


class CarRecommendPayload(BaseModel):
    cars: list[CarSpec]
    reason: str


class WsOutgoing(BaseModel):
    type: str
    payload: dict
    session_id: str
    ts: datetime = datetime.now(timezone.utc)
