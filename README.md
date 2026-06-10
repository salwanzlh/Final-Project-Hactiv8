# Mitsubishi Showroom AI — Sales Assistant

Real-time AI assistant untuk sales mobil Mitsubishi. Sistem mendengarkan percakapan antara sales dan customer melalui mikrofon, melakukan transkripsi otomatis, mengklasifikasi siapa yang berbicara, lalu memberikan **hint** dan **rekomendasi mobil** secara langsung ke layar sales.

---

## Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)                                          │
│  • Rekam audio → kirim via WebSocket                              │
│  • Tampilkan transcript, AI hint, rekomendasi mobil               │
│  • TTS: putar suara pertanyaan yang disarankan AI                 │
└────────────────────────┬─────────────────────────────────────────┘
                         │ WebSocket  ws://backend:8000/ws/session/{id}
┌────────────────────────▼─────────────────────────────────────────┐
│  Backend (FastAPI + Python)                                       │
│                                                                   │
│  Audio Chunk → STT (ElevenLabs/OpenAI/Google)                     │
│             → Speaker Classifier (GPT-4.1 + heuristic)           │
│             → AI Analyze (LLM via Featherless/OpenAI)             │
│             → RAG Retrieval (Pinecone)                            │
│             → TTS (OpenAI)                                        │
│             → Hint + Car Recommendation → Frontend                │
└────────────────────┬──────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │  Pinecone Vector DB     │  Langfuse (tracing)
        │  • customers-data       │  Prometheus + Grafana
        │  • conversation-patterns│  (metrics & monitoring)
        └─────────────────────────┘
```

### Layanan Backend

| File | Fungsi |
|---|---|
| `backend/services/stt.py` | Speech-to-Text (ElevenLabs, OpenAI Whisper, Google) |
| `backend/services/speaker_classifier.py` | Klasifikasi speaker (Sales/Customer) via GPT-4.1 |
| `backend/services/ai.py` | Analisis percakapan + generate hint + rekomendasi |
| `backend/services/rag.py` | RAG retrieval dari Pinecone (customer profiles & conversation patterns) |
| `backend/services/elicitation.py` | Deteksi dimensi kebutuhan yang belum tergali |
| `backend/services/topic_patterns.py` | Deteksi topik & transisi percakapan |
| `backend/services/tts.py` | Text-to-Speech (OpenAI) |
| `backend/services/session.py` | Manajemen koneksi WebSocket aktif |
| `backend/services/metrics.py` | Prometheus metrics (latency, throughput, dll) |
| `backend/services/langfuse_client.py` | Tracing LLM calls ke Langfuse |

---

## Cara Menjalankan

### Requirements

- Python 3.12+
- Node.js 18+
- Docker & Docker Compose (opsional, untuk monitoring stack)
- API Keys: OpenAI, ElevenLabs, Pinecone, Langfuse (lihat `.env.example`)

### 1. Clone dan setup environment

```bash
cp .env.example .env
# Edit .env — isi semua API key yang dibutuhkan
```

### 2. Install dependencies backend

```bash
# Menggunakan uv (direkomendasikan)
uv sync

# Atau pip biasa
pip install -r requirements.txt
```

### 3. Seed data ke Pinecone (hanya sekali)

```bash
python scripts/seed_pinecone.py
python -m backend.scripts.seed_conversation_patterns
```

### 4. Jalankan backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Jalankan frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend berjalan di `http://localhost:5173`  
Backend berjalan di `http://localhost:8000`  
Swagger docs di `http://localhost:8000/docs`

---

### Menjalankan dengan Docker Compose

```bash
# Backend + Frontend
docker compose up --build

# Monitoring stack (Prometheus + Grafana)
cd monitoring-stack/prometheus
docker compose up -d

# Langfuse (tracing)
cd monitoring-stack/langfuse
docker compose up -d
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin/admin) |
| Langfuse | http://localhost:3000 |

---

## Konfigurasi `.env`

```env
# LLM — pilih salah satu
FEATHERLESS_API_KEY=          # Featherless.ai (open-source LLM)
FEATHERLESS_BASE_URL=https://api.featherless.ai/v1
OPENAI_API_KEY=               # OpenAI (GPT-4.1 untuk classifier)

# STT — pilih provider
STT_PROVIDER=elevenlabs       # elevenlabs | openai | google | local
ELEVENLABS_API_KEY=

# TTS
TTS_PROVIDER=openai
TTS_MODEL=gpt-4o-mini-tts
TTS_VOICE=nova
TTS_SPEED=1.5

# Vector DB
PINECONE_API_KEY=
PINECONE_INDEX_NAME=mitsubishi-customers
PINECONE_NAMESPACE=customers-data

# Observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000

# App
LANGUAGE=id                   # id | en
APP_MODE=production           # demo | production
PORT=8000
CORS_ORIGINS=http://localhost:80,http://localhost:5173
```

---

## WebSocket Protocol

### Koneksi

```
ws://localhost:8000/ws/session/{session_id}
```

Dapatkan `session_id` baru dari:
```
GET /api/new-session
```

### Pesan Client → Server

```jsonc
// Kirim chunk audio (base64 encoded)
{ "type": "audio_chunk", "payload": { "audio": "<base64>" } }

// Akhiri sesi
{ "type": "session_end", "payload": {} }

// Keepalive
{ "type": "ping", "payload": {} }
```

### Pesan Server → Client

```jsonc
// Transkrip real-time
{
  "type": "transcript",
  "payload": {
    "utterance": {
      "id": "uuid",
      "speaker": "sales | customer | unknown",
      "text": "Saya cari mobil untuk keluarga",
      "timestamp": "2024-01-01T10:00:00",
      "confidence": 0.95
    }
  }
}

// Hint untuk sales
{
  "type": "ai_hint",
  "payload": {
    "hint_text": "Customer butuh 7 kursi, budget 250 jt",
    "suggested_question": "Apakah Bapak sering ke luar kota?",
    "probe_topics": ["budget", "kapasitas"],
    "detected_needs": ["Keluarga besar", "Budget 250 juta"],
    "question_source": "llm | rag | elicitation | topic_transition"
  }
}

// Rekomendasi mobil
{
  "type": "car_recommend",
  "payload": {
    "cars": [ { ...CarSpec } ],
    "reason": "Cocok karena kapasitas 7 kursi dan harga sesuai budget"
  }
}

// Audio TTS (pertanyaan yang disarankan)
{
  "type": "tts_audio",
  "payload": {
    "audio": "<base64 mp3>",
    "format": "mp3",
    "text": "Apakah Bapak sering ke luar kota?"
  }
}
```

---

## Pipeline Audio Real-time

Setiap audio chunk yang diterima diproses secara paralel untuk meminimalkan latensi:

```
Audio Chunk
    │
    ▼
[1] STT — transkripsi audio → teks
    │
    ├──[2a] Speaker Classifier (paralel)
    │       • Heuristic: 1-2 kata → alternating
    │       • LLM (GPT-4.1): confidence-based
    │
    └──[2b] AI Analyze (paralel, skip jika AI masih sibuk)
            • LLM analisis percakapan
            • RAG: ambil pola percakapan serupa dari Pinecone
            • Elicitation: dimensi kebutuhan yang belum digali
            │
            ▼
        [3] Send Transcript + Hint + Car Recs + TTS ke Frontend
```

**Fallback saat AI sibuk** (urutan prioritas):
1. Topic transition patterns (Pinecone)
2. RAG conversation sequence patterns (Pinecone)
3. Elicitation — dimensi yang belum tergali

**Silence watchdog**: jika tidak ada utterance baru selama 8 detik, fallback hint dikirim otomatis.

---

## Menjalankan Test

```bash
# Semua test
pytest

# Test spesifik
pytest test_rag_pipeline_integration.py
pytest test_classifier_model_compare.py
pytest test_edge_cases_emotional_intelligence.py
```

---

## Struktur Folder

```
dealer/
├── main.py                          # Entry point FastAPI
├── docker-compose.yml               # Backend + Frontend
├── pyproject.toml / requirements.txt
│
├── backend/
│   ├── config.py                    # Settings dari .env
│   ├── routers/
│   │   ├── ws.py                    # WebSocket endpoint /ws/session/{id}
│   │   └── cars.py                  # REST endpoint /api/cars
│   ├── services/
│   │   ├── stt.py                   # Speech-to-Text
│   │   ├── tts.py                   # Text-to-Speech
│   │   ├── ai.py                    # Analisis percakapan + rekomendasi
│   │   ├── speaker_classifier.py    # Klasifikasi Sales/Customer
│   │   ├── rag.py                   # Pinecone retrieval
│   │   ├── elicitation.py           # Gali kebutuhan customer
│   │   ├── topic_patterns.py        # Deteksi & transisi topik
│   │   ├── session.py               # Manajemen WebSocket session
│   │   ├── metrics.py               # Prometheus metrics
│   │   └── langfuse_client.py       # LLM tracing
│   ├── models/
│   │   └── schemas.py               # Pydantic models
│   └── db/
│       ├── car_db.py                # Data inventori mobil
│       └── customer_db.py           # Data profil customer
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Root component
│   │   ├── components/              # ConversationPanel, AiHintPanel, CarDashboard
│   │   └── hooks/                   # useWebSocket, useAudioRecorder, useTts
│   └── Dockerfile
│
├── monitoring-stack/
│   ├── prometheus/                  # Prometheus + Grafana
│   └── langfuse/                    # Langfuse tracing server
│
├── scripts/
│   └── seed_pinecone.py             # Upload customer profiles ke Pinecone
│
└── test_*.py                        # Test suite
```
