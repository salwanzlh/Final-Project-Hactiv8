import { useState, useCallback, useEffect } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { useTts } from "./hooks/useTts";
import { ConversationPanel } from "./components/ConversationPanel";
import { AiHintPanel } from "./components/AiHintPanel";
import { CarDashboard } from "./components/CarDashboard";

function MitsubishiLogo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="13 19 74 65" fill="#e4001b">
      {/* Top diamond */}
      <polygon points="50,22 63,41 50,60 37,41" />
      {/* Bottom-right diamond */}
      <polygon points="84,80 61,81 50,60 73,59" />
      {/* Bottom-left diamond */}
      <polygon points="16,80 39,81 50,60 27,59" />
    </svg>
  );
}

// Status badge warna
const STATUS_COLOR = {
  connected: "#e4001b",
  connecting: "#ff6b35",
  disconnected: "#555555",
  error: "#ff1a30",
};

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [utterances, setUtterances] = useState([]);
  const [currentHint, setCurrentHint] = useState(null);
  const [carData, setCarData] = useState({ cars: [], reason: "" });
  const [sessionActive, setSessionActive] = useState(false);

  // ── TTS hook ───────────────────────────────────────────────────────
  const { handleTtsMessage, stop: stopTts } = useTts();

  // ── Handlers dari WebSocket ─────────────────────────────────────────
  const handleTranscript = useCallback((utterance) => {
    setUtterances((prev) => [...prev, utterance]);
  }, []);

  const handleAiHint = useCallback((payload) => {
    setCurrentHint(payload);
  }, []);

  const handleCarRecommend = useCallback((payload) => {
    if (payload.cars && payload.cars.length > 0) {
      setCarData({ cars: payload.cars, reason: payload.reason });
    }
  }, []);

  const [muted, setMuted] = useState(false);

  const toggleMute = () => {
    if (!muted) stopTts(); // stop audio yang sedang main
    setMuted((m) => !m);
  };

  // Kalau muted, jangan play TTS
  const handleTtsAudio = useCallback(
    (payload) => {
      if (!muted) handleTtsMessage(payload);
    },
    [handleTtsMessage, muted],
  );

  // ── WebSocket hook ──────────────────────────────────────────────────
  const { status, sendAudio, disconnect } = useWebSocket({
    sessionId,
    onTranscript: handleTranscript,
    onAiHint: handleAiHint,
    onCarRecommend: handleCarRecommend,
    onTtsAudio: handleTtsAudio,
  });

  // ── Audio recorder hook ─────────────────────────────────────────────
  // onAudioReady dipanggil useAudioRecorder setiap ada chunk siap kirim
  const {
    isRecording,
    isSpeaking,
    volumeLevel,
    startRecording,
    stopRecording,
  } = useAudioRecorder({ onAudioReady: sendAudio });

  // ── Mulai sesi baru ─────────────────────────────────────────────────
  const startSession = async () => {
    try {
      const res = await fetch("/api/new-session");
      const data = await res.json();
      setSessionId(data.session_id);
      setUtterances([]);
      setCurrentHint({ greeting: "Ayo sapa customers mu :)" });
      setCarData({ cars: [], reason: "" });
      setSessionActive(true);
      // Tunggu WebSocket connect dulu baru start recording
      // (useEffect di bawah yang handle ini)
    } catch (err) {
      alert("Gagal memulai sesi. Pastikan backend berjalan di port 8000.");
      console.error(err);
    }
  };

  // ── Auto-start recording setelah WebSocket connected ────────────────
  useEffect(() => {
    if (sessionActive && status === "connected" && !isRecording) {
      startRecording();
    }
  }, [sessionActive, status]);

  // ── Akhiri sesi ─────────────────────────────────────────────────────
  const endSession = () => {
    stopTts();
    stopRecording();
    disconnect();
    setSessionActive(false);
    setSessionId(null);
    setCurrentHint(null);
  };

  return (
    <div style={styles.root}>
      {/* ── TOP BAR ── */}
      <div style={styles.topBar}>
        <div style={styles.brand}>
          <MitsubishiLogo size={28} />
          <div style={styles.brandText}>
            <span style={styles.brandName}>MITSUBISHI</span>
            <span style={styles.brandSub}>Sales Assistant</span>
          </div>
        </div>

        <div style={styles.topCenter}>
          {/* Status indicator */}
          <div style={styles.statusRow}>
            <div
              style={{ ...styles.statusDot, background: STATUS_COLOR[status] }}
            />
            <span style={styles.statusText}>{status}</span>
          </div>
          {/* Session ID */}
          {sessionId && (
            <span style={styles.sessionId} className="mono">
              #{sessionId.slice(0, 8)}
            </span>
          )}
        </div>

        {/* Tombol aksi */}
        <div style={styles.topActions}>
          {sessionActive && (
            <button
              style={{
                ...styles.btnMute,
                ...(muted ? styles.btnMutedActive : {}),
              }}
              onClick={toggleMute}
              title={muted ? "Aktifkan suara" : "Matikan suara"}
            >
              {muted ? "🔇 Mute" : "🔊 Suara"}
            </button>
          )}
          {!sessionActive ? (
            <button style={styles.btnStart} onClick={startSession}>
              ▶ Mulai Sesi
            </button>
          ) : (
            <button style={styles.btnEnd} onClick={endSession}>
              ■ Akhiri Sesi
            </button>
          )}
        </div>
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div style={styles.mainGrid}>
        {/* Kolom kiri — percakapan + hint */}
        <div style={styles.leftCol}>
          <div style={styles.panelWrap}>
            <ConversationPanel
              utterances={utterances}
              isSpeaking={isSpeaking}
              volumeLevel={volumeLevel}
            />
          </div>
          <div style={styles.hintWrap}>
            <AiHintPanel hint={currentHint} />
          </div>
        </div>

        {/* Kolom kanan — car dashboard */}
        <div style={styles.rightCol}>
          <div style={styles.carPanelHeader}>
            <span style={styles.carPanelTitle}>Rekomendasi Mobil</span>
            <span style={styles.carCount}>
              {carData.cars.length > 0 ? `${carData.cars.length} mobil` : "—"}
            </span>
          </div>
          <div style={styles.carPanelBody}>
            <CarDashboard cars={carData.cars} reason={carData.reason} />
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100dvh",
    display: "flex",
    flexDirection: "column",
    background: "#0a0a0a",
    color: "#f5f5f5",
  },

  // ── Top bar
  topBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 20px",
    borderBottom: "2px solid #e4001b",
    background: "#111111",
    flexShrink: 0,
  },
  brand: { display: "flex", alignItems: "center", gap: 10 },
  brandText: { display: "flex", flexDirection: "column", lineHeight: 1.2 },
  brandName: {
    fontSize: 15,
    fontWeight: 700,
    color: "#ffffff",
    letterSpacing: "0.1em",
  },
  brandSub: {
    fontSize: 10,
    color: "#e4001b",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  topCenter: { display: "flex", alignItems: "center", gap: 14 },
  statusRow: { display: "flex", alignItems: "center", gap: 6 },
  statusDot: { width: 8, height: 8, borderRadius: "50%" },
  statusText: { fontSize: 12, color: "#a0a0a0", textTransform: "capitalize" },
  sessionId: { fontSize: 11, color: "#555555" },
  topActions: { display: "flex", gap: 8 },
  btnMute: {
    background: "rgba(255,255,255,0.06)",
    color: "#a0a0a0",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 8,
    padding: "8px 14px",
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    fontFamily: "DM Sans, sans-serif",
  },
  btnMutedActive: {
    background: "rgba(228,0,27,0.12)",
    color: "#e4001b",
    borderColor: "rgba(228,0,27,0.3)",
  },
  btnStart: {
    background: "#e4001b",
    color: "#ffffff",
    border: "none",
    borderRadius: 8,
    padding: "8px 18px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "DM Sans, sans-serif",
    letterSpacing: "0.03em",
  },
  btnEnd: {
    background: "rgba(228,0,27,0.12)",
    color: "#e4001b",
    border: "1px solid rgba(228,0,27,0.35)",
    borderRadius: 8,
    padding: "8px 18px",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "DM Sans, sans-serif",
  },

  // ── Main grid
  mainGrid: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "340px 1fr",
    gap: 12,
    padding: 12,
    overflow: "hidden",
    height: "calc(100dvh - 57px)",
  },

  // ── Kolom kiri
  leftCol: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
    overflow: "hidden",
  },
  panelWrap: { flex: "1 1 0", overflow: "hidden", minHeight: 0 },
  hintWrap: { flex: "0 0 220px" },

  // ── Kolom kanan
  rightCol: {
    background: "#141414",
    border: "1px solid rgba(228,0,27,0.2)",
    borderTop: "2px solid #e4001b",
    borderRadius: 14,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  carPanelHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 18px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    flexShrink: 0,
  },
  carPanelTitle: {
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    color: "#e4001b",
  },
  carCount: { fontSize: 12, color: "#555555" },
  carPanelBody: { flex: 1, overflowY: "auto", padding: 14 },
};
