/**
 * useWebSocket
 *
 * Mengelola koneksi WebSocket ke backend FastAPI.
 * - Auto-reconnect kalau koneksi putus
 * - Kirim audio chunk ke backend
 * - Terima dan dispatch pesan: transcript, ai_hint, car_recommend
 */

import { useRef, useState, useCallback, useEffect } from "react";

const WS_URL = (sessionId) => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/session/${sessionId}`;
};
const RECONNECT_DELAY = 2000; // ms sebelum coba reconnect
const MAX_RECONNECT = 5; // maksimal berapa kali coba reconnect

export function useWebSocket({
  sessionId,
  onTranscript,
  onAiHint,
  onCarRecommend,
  onTtsAudio,
}) {
  const [status, setStatus] = useState("disconnected"); // connected | connecting | disconnected | error
  const wsRef = useRef(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(false); // false = disconnect sengaja, tidak perlu reconnect

  // ── Dispatch pesan dari server ke handler yang tepat ───────────────
  const handleMessage = useCallback(
    (raw) => {
      let msg;
      try {
        msg = JSON.parse(raw);
      } catch {
        console.warn("[WS] Pesan tidak valid JSON:", raw);
        return;
      }

      switch (msg.type) {
        case "transcript":
          onTranscript?.(msg.payload.utterance);
          break;
        case "ai_hint":
          onAiHint?.(msg.payload);
          break;
        case "car_recommend":
          onCarRecommend?.(msg.payload);
          break;
        case "tts_audio":
          onTtsAudio?.(msg.payload);
          break;
        case "connected":
          console.log("[WS] Sesi dimulai:", msg.payload);
          break;
        case "pong":
          break; // keepalive, tidak perlu handling
        case "error":
          console.error("[WS] Error dari server:", msg.payload.message);
          break;
        default:
          console.log("[WS] Pesan tidak dikenal:", msg.type);
      }
    },
    [onTranscript, onAiHint, onCarRecommend, onTtsAudio],
  );

  // ── Buka koneksi WebSocket ──────────────────────────────────────────
  const connect = useCallback(() => {
    if (!sessionId) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(WS_URL(sessionId));

    ws.onopen = () => {
      console.log("[WS] Terhubung:", sessionId);
      setStatus("connected");
      reconnectCountRef.current = 0; // reset counter kalau berhasil connect
    };

    ws.onmessage = (event) => handleMessage(event.data);

    ws.onclose = (event) => {
      console.log("[WS] Koneksi tutup, code:", event.code);
      setStatus("disconnected");

      // Reconnect otomatis kalau bukan disconnect sengaja
      if (
        shouldReconnectRef.current &&
        reconnectCountRef.current < MAX_RECONNECT
      ) {
        reconnectCountRef.current++;
        console.log(
          `[WS] Reconnect ke-${reconnectCountRef.current} dalam ${RECONNECT_DELAY}ms...`,
        );
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
      setStatus("error");
    };

    wsRef.current = ws;
  }, [sessionId, handleMessage]);

  // ── Disconnect sengaja (akhiri sesi) ───────────────────────────────
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false; // tandai: ini disconnect sengaja
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }
    if (wsRef.current) {
      // Beritahu server sesi selesai sebelum tutup
      sendMessage("session_end", {});
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, []);

  // ── Kirim pesan ke server ───────────────────────────────────────────
  const sendMessage = useCallback((type, payload) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.warn("[WS] Tidak bisa kirim — koneksi belum terbuka");
      return;
    }
    wsRef.current.send(JSON.stringify({ type, payload }));
  }, []);

  // ── Kirim audio chunk (dipanggil dari useAudioRecorder) ────────────
  const sendAudio = useCallback(
    (base64Audio, speaker = "unknown") => {
      sendMessage("audio_chunk", { audio: base64Audio, speaker });
    },
    [sendMessage],
  );

  // ── Auto-connect saat sessionId tersedia ───────────────────────────
  useEffect(() => {
    if (!sessionId) return;
    shouldReconnectRef.current = true;
    connect();

    // Cleanup saat komponen unmount
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [sessionId, connect]);

  // ── Ping setiap 30 detik supaya koneksi tidak mati ─────────────────
  useEffect(() => {
    if (status !== "connected") return;
    const pingInterval = setInterval(() => sendMessage("ping", {}), 30_000);
    return () => clearInterval(pingInterval);
  }, [status, sendMessage]);

  return { status, sendAudio, disconnect };
}
