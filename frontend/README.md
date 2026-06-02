# Showroom AI — Frontend

React + Vite PWA untuk tablet sales showroom.

## Struktur folder

```
frontend/
├── index.html
├── vite.config.js          # Proxy /api dan /ws ke backend :8000
├── package.json
└── src/
    ├── main.jsx            # Entry point React
    ├── App.jsx             # Root component — sambungkan semua
    ├── index.css           # Design system (CSS variables, font)
    │
    ├── hooks/
    │   ├── useAudioRecorder.js   # VAD + MediaRecorder + rolling buffer
    │   └── useWebSocket.js       # Koneksi WS + auto-reconnect + dispatch pesan
    │
    └── components/
        ├── ConversationPanel.jsx  # Transkrip real-time + volume bar
        ├── AiHintPanel.jsx        # Hint + pertanyaan + kebutuhan terdeteksi
        └── CarDashboard.jsx       # Spec mobil + radar chart + warna + fitur
```

## Cara menjalankan

### 1. Pastikan backend sudah jalan dulu
```bash
cd ../backend
uvicorn main:app --reload
```

### 2. Install dependencies frontend
```bash
cd frontend
npm install
```

### 3. Jalankan dev server
```bash
npm run dev
```

Buka di browser: `http://localhost:5173`

Untuk tablet di jaringan yang sama:
```bash
npm run dev -- --host
# Akses dari tablet: http://<IP-laptop>:5173
```

## Alur data

```
Tombol "Mulai Sesi"
  → GET /api/new-session → dapat session_id
  → useWebSocket connect ke ws://localhost:8000/ws/session/{id}
  → useAudioRecorder.startRecording() — minta izin mic

Rekaman berjalan:
  → MediaRecorder hasilkan chunk tiap 100ms → buffer
  → Web Audio API ukur volume tiap 100ms
  → Jeda > 1.5 detik ATAU buffer > 8 detik → flushBuffer()
  → base64 encode → useWebSocket.sendAudio()
  → backend terima → STT → AI → kirim balik

Pesan dari backend:
  → "transcript"    → tambah ke utterances → ConversationPanel update
  → "ai_hint"       → AiHintPanel update
  → "car_recommend" → CarDashboard update
```
