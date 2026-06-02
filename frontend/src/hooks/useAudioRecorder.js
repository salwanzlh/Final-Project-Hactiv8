/**
 * useAudioRecorder
 *
 * Logika:
 * - MediaRecorder start fresh setiap utterance — setiap blob adalah WebM yang valid
 * - Web Audio API ukur volume setiap 100ms
 * - Saat senyap > SILENCE_THRESHOLD ms: stop recorder, kirim blob, restart recorder
 * - Safety net: stop paksa kalau satu utterance > MAX_BUFFER_MS ms
 */

import { useRef, useState, useCallback } from "react";

const SILENCE_THRESHOLD_MS = 300;
const MAX_BUFFER_MS = 8000;
const VOLUME_CHECK_INTERVAL = 100;
const SILENCE_VOLUME_LIMIT = 0.015;

export function useAudioRecorder({ onAudioReady }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const chunksBufferRef = useRef([]);
  const silenceTimerRef = useRef(null);
  const bufferStartRef = useRef(null);
  const volumeIntervalRef = useRef(null);
  const isSpeakingRef = useRef(false);
  const isSessionRef = useRef(false); // apakah sesi masih aktif
  const hadSpeechRef = useRef(false); // ada suara bicara di recording ini?
  const speechDurationRef = useRef(0); // total ms suara terdeteksi di recording ini

  // ── Buat dan jalankan MediaRecorder baru ─────────────────────────────
  const startRecorder = useCallback(() => {
    if (!streamRef.current || !isSessionRef.current) return;

    const mr = new MediaRecorder(streamRef.current, {
      mimeType: "audio/webm;codecs=opus",
    });

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksBufferRef.current.push(e.data);
    };

    mr.onstop = () => {
      const chunks = chunksBufferRef.current.splice(0);

      // Hanya kirim ke STT kalau ada suara bicara ≥ 300ms (filter noise singkat)
      if (
        chunks.length > 0 &&
        hadSpeechRef.current &&
        speechDurationRef.current >= 300
      ) {
        const blob = new Blob(chunks, { type: "audio/webm;codecs=opus" });
        if (blob.size >= 1000 && isSessionRef.current) {
          const reader = new FileReader();
          reader.onloadend = () => onAudioReady(reader.result.split(",")[1]);
          reader.readAsDataURL(blob);
        }
      }

      if (isSessionRef.current) {
        startRecorder();
      }
    };

    hadSpeechRef.current = false; // reset untuk utterance baru
    speechDurationRef.current = 0;
    mr.start(100);
    mediaRecorderRef.current = mr;
    bufferStartRef.current = Date.now();
  }, [onAudioReady]);

  // ── Stop recorder saat ini → trigger onstop → kirim + restart ────────
  const stopCurrentRecorder = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // ── Loop cek volume setiap 100ms ─────────────────────────────────────
  const startVolumeCheck = useCallback(() => {
    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    volumeIntervalRef.current = setInterval(() => {
      analyser.getByteFrequencyData(dataArray);

      const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
      const volume = avg / 255;
      setVolumeLevel(volume);

      const speaking = volume > SILENCE_VOLUME_LIMIT;

      if (speaking) {
        isSpeakingRef.current = true;
        hadSpeechRef.current = true;
        speechDurationRef.current += VOLUME_CHECK_INTERVAL; // akumulasi durasi bicara
        setIsSpeaking(true);
        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
      } else if (isSpeakingRef.current) {
        if (!silenceTimerRef.current) {
          silenceTimerRef.current = setTimeout(() => {
            isSpeakingRef.current = false;
            silenceTimerRef.current = null;
            setIsSpeaking(false);
            stopCurrentRecorder(); // stop → onstop → kirim blob → restart
          }, SILENCE_THRESHOLD_MS);
        }
      }

      // Safety net: utterance terlalu panjang → paksa stop
      if (bufferStartRef.current) {
        const age = Date.now() - bufferStartRef.current;
        if (age >= MAX_BUFFER_MS) {
          if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
          }
          stopCurrentRecorder();
        }
      }
    }, VOLUME_CHECK_INTERVAL);
  }, [stopCurrentRecorder]);

  // ── Start session ─────────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true, // filter suara speaker dari mic (AEC)
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      isSessionRef.current = true;
      chunksBufferRef.current = [];

      startRecorder();
      startVolumeCheck();
      setIsRecording(true);
    } catch (err) {
      if (err.name === "NotAllowedError") {
        alert(
          "Izin mikrofon ditolak. Mohon izinkan akses mikrofon di browser.",
        );
      } else {
        console.error("Gagal start recording:", err);
      }
    }
  }, [startRecorder, startVolumeCheck]);

  // ── Stop session ──────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    isSessionRef.current = false; // cegah restart di onstop

    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current);
      volumeIntervalRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }

    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioContextRef.current?.close();

    mediaRecorderRef.current = null;
    audioContextRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;
    chunksBufferRef.current = [];
    bufferStartRef.current = null;
    isSpeakingRef.current = false;

    setIsRecording(false);
    setIsSpeaking(false);
    setVolumeLevel(0);
  }, []);

  return {
    isRecording,
    isSpeaking,
    volumeLevel,
    startRecording,
    stopRecording,
  };
}
