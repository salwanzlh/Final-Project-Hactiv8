import { useEffect, useRef } from 'react'

const SPEAKER_STYLE = {
  sales:    { label: 'Sales',    color: '#e4001b' },
  customer: { label: 'Customer', color: '#d0d0d0' },
  unknown:  { label: '...',      color: '#555555' },
}

export function ConversationPanel({ utterances, isSpeaking, volumeLevel }) {
  const bottomRef = useRef(null)

  // Auto-scroll ke bawah setiap ada utterance baru
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [utterances])

  return (
    <div style={styles.card}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.headerTitle}>Percakapan</span>
        <div style={styles.micStatus}>
          {/* Volume bar — 5 bar kecil */}
          <div style={styles.volBars}>
            {[0.15, 0.35, 0.55, 0.75, 1.0].map((threshold, i) => (
              <div
                key={i}
                style={{
                  ...styles.volBar,
                  height: `${8 + i * 4}px`,
                  background: volumeLevel >= threshold
                    ? '#e4001b'
                    : 'rgba(255,255,255,0.1)',
                  transition: 'background 0.1s',
                }}
              />
            ))}
          </div>
          <span style={{ ...styles.micLabel, color: isSpeaking ? '#e4001b' : '#555555' }}>
            {isSpeaking ? 'Mendengarkan...' : 'Senyap'}
          </span>
        </div>
      </div>

      {/* Daftar utterance */}
      <div style={styles.utteranceList}>
        {utterances.length === 0 ? (
          <div style={styles.empty}>Percakapan akan muncul di sini...</div>
        ) : (
          utterances.map((u) => {
            const sp = SPEAKER_STYLE[u.speaker] ?? SPEAKER_STYLE.unknown
            return (
              <div
                key={u.id}
                style={{
                  ...styles.utterance,
                  borderLeftColor: sp.color,
                }}
              >
                <div style={styles.utteranceMeta}>
                  <span style={{ ...styles.speakerBadge, color: sp.color }}>{sp.label}</span>
                  <span style={styles.timestamp}>
                    {new Date(u.timestamp).toLocaleTimeString('id-ID', {
                      hour: '2-digit', minute: '2-digit', second: '2-digit'
                    })}
                  </span>
                </div>
                <p style={styles.utteranceText}>{u.text}</p>
              </div>
            )
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

const styles = {
  card: {
    background: '#141414',
    border: '1px solid rgba(255,255,255,0.07)',
    borderTop: '2px solid #e4001b',
    borderRadius: 14,
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 18px',
    borderBottom: '1px solid rgba(255,255,255,0.06)',
    flexShrink: 0,
  },
  headerTitle: {
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    color: '#e4001b',
  },
  micStatus: { display: 'flex', alignItems: 'center', gap: 8 },
  volBars: { display: 'flex', alignItems: 'flex-end', gap: 3, height: 24 },
  volBar: { width: 3, borderRadius: 2 },
  micLabel: { fontSize: 12, fontWeight: 500 },
  utteranceList: {
    flex: 1,
    overflowY: 'auto',
    padding: '12px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  empty: { color: '#555555', fontSize: 14, textAlign: 'center', marginTop: 40 },
  utterance: {
    borderLeft: '2px solid',
    paddingLeft: 12,
    paddingTop: 2,
    paddingBottom: 2,
  },
  utteranceMeta: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 },
  speakerBadge: { fontSize: 12, fontWeight: 600, letterSpacing: '0.03em' },
  timestamp: { fontSize: 11, color: '#555555', fontFamily: 'DM Mono, monospace' },
  utteranceText: { fontSize: 14, color: '#e0e0e0', lineHeight: 1.5 },
}
