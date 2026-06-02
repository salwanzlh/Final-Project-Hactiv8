/**
 * useTts
 *
 * Dua mode playback:
 *
 * 1. PRODUCTION — backend kirim pesan "tts_audio" berisi mp3 base64.
 *    Hook decode → Blob → URL → HTMLAudioElement → play ke speaker tablet.
 *    Tidak ada latency tambahan karena audio sudah di-generate paralel
 *    dengan pengiriman hint.
 *
 * 2. DEMO — backend tidak kirim audio (tts_bytes = None).
 *    Hook pakai browser Web Speech API (SpeechSynthesis) sebagai fallback.
 *    Tidak butuh API key, jalan langsung di browser.
 *    Kualitas suara tergantung OS — di iPad cukup bagus (suara Siri).
 */

import { useRef, useCallback, useEffect } from 'react'

export function useTts() {
  const audioRef       = useRef(null)   // HTMLAudioElement aktif
  const objectUrlRef   = useRef(null)   // URL sementara untuk Blob audio
  const isSpeakingRef  = useRef(false)

  // Cleanup URL lama supaya tidak bocor memori
  const revokeCurrentUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
  }, [])

  // ── Mode 1: Play audio mp3 dari base64 (production) ────────────────
  const playFromBase64 = useCallback((base64Audio, format = 'mp3') => {
    // Hentikan audio yang sedang main dulu
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    revokeCurrentUrl()

    // Decode base64 → Uint8Array → Blob → Object URL
    const binary   = atob(base64Audio)
    const bytes    = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i)
    }
    const blob = new Blob([bytes], { type: `audio/${format}` })
    const url  = URL.createObjectURL(blob)
    objectUrlRef.current = url

    const audio = new Audio(url)
    audioRef.current = audio

    audio.onended = () => {
      isSpeakingRef.current = false
      revokeCurrentUrl()
    }
    audio.onerror = (e) => {
      console.error('[TTS] Gagal play audio:', e)
      isSpeakingRef.current = false
      revokeCurrentUrl()
    }

    isSpeakingRef.current = true
    audio.play().catch(err => {
      // Browser blokir autoplay sebelum ada interaksi user
      // Setelah user klik tombol "Mulai Sesi", ini tidak akan terjadi
      console.warn('[TTS] Autoplay diblokir browser:', err)
      isSpeakingRef.current = false
    })
  }, [revokeCurrentUrl])

  // ── Mode 2: Browser SpeechSynthesis (demo / fallback) ──────────────
  const playFromSpeech = useCallback((text) => {
    if (!('speechSynthesis' in window)) {
      console.warn('[TTS] Browser tidak support SpeechSynthesis')
      return
    }

    // Batalkan yang sedang main
    window.speechSynthesis.cancel()

    const utter  = new SpeechSynthesisUtterance(text)
    utter.lang   = 'id-ID'   // Bahasa Indonesia
    utter.rate   = 0.9        // sedikit lebih lambat dari default
    utter.pitch  = 1.0
    utter.volume = 1.0

    // Pilih suara Indonesia kalau tersedia, fallback ke default
    const voices = window.speechSynthesis.getVoices()
    const idVoice = voices.find(v => v.lang.startsWith('id'))
    if (idVoice) utter.voice = idVoice

    utter.onend   = () => { isSpeakingRef.current = false }
    utter.onerror = () => { isSpeakingRef.current = false }

    isSpeakingRef.current = true
    window.speechSynthesis.speak(utter)
  }, [])

  // ── Handler utama — dipanggil dari useWebSocket ─────────────────────
  //
  // Kalau pesan "tts_audio" ada → pakai mp3 dari backend (production)
  // Kalau tidak ada → pakai SpeechSynthesis dengan teks hint (demo)
  const handleTtsMessage = useCallback((payload) => {
    if (payload?.audio) {
      // Production: backend kirim mp3
      playFromBase64(payload.audio, payload.format ?? 'mp3')
    } else if (payload?.text) {
      // Demo: fallback ke browser TTS
      playFromSpeech(payload.text)
    }
  }, [playFromBase64, playFromSpeech])

  // ── Stop manual (misal: sales mute) ────────────────────────────────
  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    window.speechSynthesis?.cancel()
    revokeCurrentUrl()
    isSpeakingRef.current = false
  }, [revokeCurrentUrl])

  // Cleanup saat komponen unmount
  useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

  return { handleTtsMessage, stop }
}
