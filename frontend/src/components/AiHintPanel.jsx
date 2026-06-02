export function AiHintPanel({ hint }) {
  if (!hint) {
    return (
      <div style={styles.card}>
        <div style={styles.header}>
          <span style={styles.headerTitle}>AI Hint</span>
          <span style={styles.badge}>Standby</span>
        </div>
        <div style={styles.empty}>
          Mulai sesi untuk mendapatkan panduan real-time
        </div>
      </div>
    );
  }

  if (hint.greeting) {
    return (
      <div style={styles.card}>
        <div style={styles.header}>
          <span style={styles.headerTitle}>AI Hint</span>
          <span style={{ ...styles.badge, background: "rgba(228,0,27,0.15)", color: "#e4001b" }}>
            Live
          </span>
        </div>
        <div style={styles.greeting}>{hint.greeting}</div>
      </div>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>AI Hint</span>
        <span
          style={{
            ...styles.badge,
            background: "rgba(228,0,27,0.15)",
            color: "#e4001b",
          }}
        >
          Live
        </span>
      </div>

      <div style={styles.body}>
        {/* Probe topics — paling atas, paling prominent */}
        {hint.probe_topics?.length > 0 && (
          <div style={styles.probeBox}>
            <div style={styles.probeLabel}>GALI LEBIH LANJUT</div>
            <div style={styles.probeList}>
              {hint.probe_topics.map((topic, i) => (
                <div key={i} style={styles.probeItem}>
                  <span style={styles.probeArrow}>›</span>
                  <span style={styles.probeText}>{topic}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Suggested question — contoh kalimat lengkap */}
        <div style={styles.questionBox}>
          <div style={styles.questionLabel}>💬 Contoh pertanyaan</div>
          <p style={styles.questionText}>"{hint.suggested_question}"</p>
        </div>

        {/* Insight — teks kecil di bawah */}
        <div style={styles.insightRow}>
          <span style={styles.insightDot}>◆</span>
          <span style={styles.insightText}>{hint.hint_text}</span>
        </div>
      </div>
    </div>
  );
}

const styles = {
  card: {
    background: "#141414",
    border: "1px solid rgba(255,255,255,0.07)",
    borderTop: "2px solid #e4001b",
    borderRadius: 14,
    display: "flex",
    flexDirection: "column",
    height: "100%",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 18px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    flexShrink: 0,
  },
  headerTitle: {
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    color: "#e4001b",
  },
  badge: {
    fontSize: 11,
    fontWeight: 600,
    padding: "3px 10px",
    borderRadius: 20,
    background: "rgba(255,255,255,0.06)",
    color: "#555555",
  },
  empty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#555555",
    fontSize: 13,
    padding: 24,
    textAlign: "center",
  },
  greeting: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#ffffff",
    fontSize: 15,
    fontWeight: 600,
    padding: 24,
    textAlign: "center",
  },
  body: {
    padding: "12px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
    overflowY: "auto",
  },

  // ── Probe topics
  probeBox: {
    background: "rgba(228,0,27,0.07)",
    border: "1px solid rgba(228,0,27,0.25)",
    borderRadius: 10,
    padding: "10px 14px",
  },
  probeLabel: {
    fontSize: 10,
    fontWeight: 700,
    color: "#e4001b",
    letterSpacing: "0.08em",
    marginBottom: 8,
  },
  probeList: { display: "flex", flexDirection: "column", gap: 6 },
  probeItem: { display: "flex", alignItems: "baseline", gap: 8 },
  probeArrow: { color: "#e4001b", fontSize: 16, lineHeight: 1, flexShrink: 0 },
  probeText: {
    fontSize: 14,
    fontWeight: 500,
    color: "#f0d0d0",
    lineHeight: 1.4,
  },

  // ── Suggested question
  questionBox: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 10,
    padding: "10px 14px",
  },
  questionLabel: {
    fontSize: 11,
    fontWeight: 600,
    color: "#d0d0d0",
    marginBottom: 4,
  },
  questionText: {
    fontSize: 13,
    color: "#f5f5f5",
    lineHeight: 1.5,
    fontStyle: "italic",
  },

  // ── Insight row
  insightRow: {
    display: "flex",
    alignItems: "flex-start",
    gap: 7,
    padding: "0 2px",
  },
  insightDot: { color: "#e4001b", fontSize: 9, marginTop: 4, flexShrink: 0 },
  insightText: { fontSize: 12, color: "#888888", lineHeight: 1.5 },
};
