import { useState } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

function MitsubishiLogo({ size = 48, opacity = 0.5 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <polygon points="50,5 65,35 50,50 35,35" fill="#e60012" opacity={opacity} />
      <polygon points="11,73 45,71 50,50 30,45" fill="#e60012" opacity={opacity} />
      <polygon points="89,73 71,45 50,50 56,71" fill="#e60012" opacity={opacity} />
    </svg>
  );
}

const RADAR_KEYS = [
  "kenyamanan",
  "performa",
  "efisiensi",
  "keamanan",
  "kapasitas",
];

function formatRupiah(n) {
  return new Intl.NumberFormat("id-ID", {
    style: "currency",
    currency: "IDR",
    maximumFractionDigits: 0,
  }).format(n);
}

function CarImage({ carId, primaryColor }) {
  const [failed, setFailed] = useState(false);
  const src = `/cars/${carId}.jpg`;

  if (failed) {
    return (
      <div
        style={{
          ...styles.imagePlaceholder,
          background: `linear-gradient(135deg, ${primaryColor}22 0%, #0d1020 100%)`,
        }}
      >
        <MitsubishiLogo size={64} opacity={0.4} />
      </div>
    );
  }

  return (
    <div style={styles.imageWrap}>
      <img
        src={src}
        alt={carId}
        style={styles.carImg}
        onError={() => setFailed(true)}
      />
      <div style={{ ...styles.imageColorBar, background: primaryColor }} />
    </div>
  );
}

function CarCard({ car, isActive, onClick }) {
  const [colorIdx, setColorIdx] = useState(0);

  const radarData = RADAR_KEYS.map((key) => ({
    subject: key.charAt(0).toUpperCase() + key.slice(1),
    value: car.radar[key] ?? 0,
  }));

  return (
    <div
      style={{ ...styles.carCard, ...(isActive ? styles.carCardActive : {}) }}
      onClick={onClick}
    >
      {/* Car image */}
      <CarImage
        carId={car.id}
        primaryColor={car.colors[colorIdx]?.hex ?? "#4d7cfe"}
      />

      <div style={styles.carCardInner}>
        {/* Header */}
        <div style={styles.carHeader}>
          <div>
            <div style={styles.carBrand}>{car.brand}</div>
            <div style={styles.carModel}>
              {car.model} {car.variant}
            </div>
          </div>
          <div style={styles.carTypeBadge}>{car.type}</div>
        </div>

        <div style={styles.carPrice}>{formatRupiah(car.price_otr_jakarta)}</div>

        {/* Spec grid */}
        <div style={styles.specGrid}>
          <div style={styles.specItem}>
            <span style={styles.specVal}>{car.engine_cc}</span>
            <span style={styles.specLbl}>cc</span>
          </div>
          <div style={styles.specItem}>
            <span style={styles.specVal}>{car.horsepower}</span>
            <span style={styles.specLbl}>hp</span>
          </div>
          <div style={styles.specItem}>
            <span style={styles.specVal}>{car.seats}</span>
            <span style={styles.specLbl}>kursi</span>
          </div>
          <div style={styles.specItem}>
            <span style={styles.specVal}>{car.fuel_consumption_kml}</span>
            <span style={styles.specLbl}>km/L</span>
          </div>
        </div>

        {/* Radar chart */}
        <div style={{ height: 160, marginTop: 8 }}>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis
                dataKey="subject"
                tick={{ fill: "#8b91a8", fontSize: 11 }}
              />
              <Radar
                dataKey="value"
                stroke="#4d7cfe"
                fill="#4d7cfe"
                fillOpacity={0.15}
                strokeWidth={1.5}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Warna */}
        <div style={styles.colorsRow}>
          {car.colors.map((c, i) => (
            <button
              key={i}
              onClick={(e) => {
                e.stopPropagation();
                setColorIdx(i);
              }}
              title={c.name}
              style={{
                ...styles.colorDot,
                background: c.hex,
                outline:
                  i === colorIdx
                    ? `2px solid #4d7cfe`
                    : "2px solid transparent",
                outlineOffset: 2,
              }}
            />
          ))}
        </div>
        <div style={styles.colorName}>{car.colors[colorIdx]?.name}</div>

        {/* Velg */}
        <div style={styles.wheelInfo}>
          <span style={styles.wheelIcon}>◎</span>
          {car.wheel_size_inch}" Alloy
        </div>

        {/* Fitur */}
        <div style={styles.featureList}>
          {car.features.slice(0, 4).map((f, i) => (
            <div key={i} style={styles.featureItem}>
              <span style={styles.featureDot}>✓</span>
              <span style={styles.featureText}>{f}</span>
            </div>
          ))}
          {car.features.length > 4 && (
            <div style={styles.featureMore}>
              +{car.features.length - 4} fitur lainnya
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function CarDashboard({ cars, reason }) {
  const [activeCarId, setActiveCarId] = useState(null);

  if (!cars || cars.length === 0) {
    return (
      <div style={styles.emptyWrap}>
        <MitsubishiLogo size={52} opacity={0.5} />
        <div style={styles.emptyText}>
          Rekomendasi mobil akan muncul di sini
        </div>
        <div style={styles.emptySubtext}>
          AI akan menyarankan mobil sesuai kebutuhan customer
        </div>
      </div>
    );
  }

  return (
    <div style={styles.wrap}>
      {/* Reason banner */}
      {reason && (
        <div style={styles.reasonBanner}>
          <span style={styles.reasonIcon}>✦</span>
          {reason}
        </div>
      )}

      {/* Car cards */}
      <div style={styles.carGrid}>
        {cars.map((car) => (
          <CarCard
            key={car.id}
            car={car}
            isActive={activeCarId === car.id}
            onClick={() =>
              setActiveCarId(activeCarId === car.id ? null : car.id)
            }
          />
        ))}
      </div>
    </div>
  );
}

const styles = {
  wrap: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
    height: "100%",
    overflowY: "auto",
  },
  reasonBanner: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    background: "rgba(251,191,36,0.08)",
    border: "1px solid rgba(251,191,36,0.2)",
    borderRadius: 10,
    padding: "8px 14px",
    fontSize: 13,
    color: "#fbbf24",
  },
  reasonIcon: { fontSize: 10 },

  carGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: 12,
  },

  carCard: {
    background: "#181c27",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    cursor: "pointer",
    overflow: "hidden",
    transition: "border-color 0.2s",
    position: "relative",
  },
  carCardActive: { borderColor: "#4d7cfe" },
  carCardInner: {
    padding: "14px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },

  imageWrap: {
    position: "relative",
    width: "100%",
    height: 160,
    overflow: "hidden",
  },
  carImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
  },
  imageColorBar: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: 5,
  },
  imagePlaceholder: {
    width: "100%",
    height: 160,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  imagePlaceholderIcon: { fontSize: 48, opacity: 0.4 },

  carHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  carBrand: {
    fontSize: 11,
    color: "#8b91a8",
    fontWeight: 600,
    letterSpacing: "0.04em",
  },
  carModel: { fontSize: 15, fontWeight: 600, color: "#f0f2f8" },
  carTypeBadge: {
    fontSize: 10,
    fontWeight: 600,
    padding: "3px 8px",
    borderRadius: 20,
    background: "rgba(77,124,254,0.12)",
    color: "#4d7cfe",
    letterSpacing: "0.04em",
    whiteSpace: "nowrap",
  },
  carPrice: { fontSize: 16, fontWeight: 600, color: "#34d399" },

  specGrid: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6 },
  specItem: {
    background: "rgba(255,255,255,0.04)",
    borderRadius: 8,
    padding: "6px 8px",
    textAlign: "center",
  },
  specVal: {
    display: "block",
    fontSize: 14,
    fontWeight: 600,
    color: "#f0f2f8",
  },
  specLbl: { display: "block", fontSize: 10, color: "#555d78", marginTop: 1 },

  colorsRow: { display: "flex", gap: 6, flexWrap: "wrap" },
  colorDot: {
    width: 20,
    height: 20,
    borderRadius: "50%",
    border: "none",
    cursor: "pointer",
    boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.2)",
  },
  colorName: { fontSize: 11, color: "#8b91a8", marginTop: -4 },

  wheelInfo: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 12,
    color: "#8b91a8",
  },
  wheelIcon: { fontSize: 14, color: "#555d78" },

  featureList: { display: "flex", flexDirection: "column", gap: 4 },
  featureItem: { display: "flex", gap: 6, alignItems: "flex-start" },
  featureDot: { fontSize: 11, color: "#34d399", marginTop: 2, flexShrink: 0 },
  featureText: { fontSize: 12, color: "#8b91a8", lineHeight: 1.4 },
  featureMore: { fontSize: 11, color: "#555d78", marginTop: 2 },

  emptyWrap: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    padding: 40,
  },
  emptyIcon: { fontSize: 36 },
  emptyText: { fontSize: 15, color: "#8b91a8", fontWeight: 500 },
  emptySubtext: { fontSize: 13, color: "#555d78", textAlign: "center" },
};
