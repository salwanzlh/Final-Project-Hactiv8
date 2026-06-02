"""
Benchmark: classify_speaker — gpt-4o-mini vs gpt-4.1-mini
==========================================================
Membandingkan fungsi classify_speaker menggunakan dua model OpenAI:
  A. gpt-4o-mini  ($0.15 / $0.60 per 1M token  input/output)
  B. gpt-4.1-mini ($0.40 / $1.60 per 1M token  input/output)

4 skenario teks showroom Mitsubishi:
  1. HEURISTIC              : "Oh, oke siap."
  2. NORMAL_PANJANG         : kalimat panjang customer tentang MPV
  3. OVERLAP                : kalimat overlap sales–customer
  4. NORMAL_PANJANG_OVERLAP : kalimat panjang dengan konteks overlap

Metrik per run:
  latency_ms     : waktu panggilan API (time.perf_counter), ms
  prompt_tokens  : response.usage.prompt_tokens  (0 jika heuristic)
  completion_toks: response.usage.completion_tokens (0 jika heuristic)
  cost_usd       : estimasi biaya (harga pasar 2025)

Usage:
    python test_classifier_model_compare.py
    python test_classifier_model_compare.py --runs 5
    python test_classifier_model_compare.py --runs 3 --last-speaker customer
    python test_classifier_model_compare.py --csv hasil_benchmark.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Literal

# ── Pricing (USD per token) ────────────────────────────────────────────────────
PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {
        "input":  0.40  / 1_000_000,   # $0.40 / 1M input tokens
        "output": 1.60  / 1_000_000,   # $1.60 / 1M output tokens
    },
    "gpt-4.1": {
        "input":  2.00  / 1_000_000,   # $2.00 / 1M input tokens
        "output": 8.00  / 1_000_000,   # $8.00 / 1M output tokens
    },
}

# Heuristic threshold — sama dengan speaker_classifier.py
_SHORT_TEXT_WORDS = 2

# ── System prompt — identik dengan speaker_classifier.py ─────────────────────
_SYSTEM_PROMPT = """\
Klasifikasikan speaker percakapan showroom Mitsubishi Indonesia: SALES atau CUSTOMER.
━━━ SINYAL ━━━
SALES pasti  : "Ada yang bisa saya bantu", "Selamat datang", "Silakan", memanggil "Pak/Bu",
               menawarkan test drive/brosur/kredit, sebut "unit kami / promo bulan ini".
CUSTOMER pasti: memanggil "Mbak/Mas", "Saya lagi cari/mau/butuh", cerita keluarga/rutinitas pribadi,
               "Budget/DP/cicilan saya...", bertanya harga/stok/kredit sebagai respons.
SALES kuat   : menjelaskan spesifikasi/harga atas inisiatif sendiri, menggali kebutuhan customer.
CUSTOMER kuat: menyatakan preferensi pribadi, ungkapkan keberatan ("mahal ya", "pikir dulu").
LEMAH        : sapaan netral / jawaban pendek → assign ke speaker BERBEDA dari sebelumnya.
AMBIGU       : confidence 0.50, tetap pilih, jangan null.

━━━ EXAMPLES ━━━
"Halo Mbak selamat siang" | riwayat: - → {"speaker":"customer","confidence":0.95}
"Ada yang bisa saya bantu Pak?" | riwayat: [customer] → {"speaker":"sales","confidence":0.97}
"Ada 5 orang, saya istri sama 3 anak" | riwayat: [sales: tanya berapa orang] → {"speaker":"customer","confidence":0.95}
"Kalau gitu berarti kita butuh yang nyaman buat jarak jauh" | riwayat: [customer: sering mudik] → {"speaker":"sales","confidence":0.85}
"Wah lumayan ya, bisa dicicil tidak Mas?" | riwayat: [sales: sebut harga] → {"speaker":"customer","confidence":0.96}
"Iya" | riwayat: [sales: tanya keluar kota] → {"speaker":"customer","confidence":0.65}
"Selamat pagi" | riwayat: - → {"speaker":"sales","confidence":0.72}

Jawab HANYA JSON: {"speaker":"sales","confidence":0.85}"""

# ── 4 Skenario teks benchmark ──────────────────────────────────────────────────
SCENARIOS: list[dict] = [
    {
        "label":        "HEURISTIC",
        "text":         "Oh, oke siap.",
        "description":  "Respons pendek — menguji apakah heuristic path aktif",
    },
    {
        "label":        "NORMAL_PANJANG",
        "text":         (
            "Saya tuh lagi berencana ganti mobil lama ke MPV yang kabinnya agak luas Mas, "
            "soalnya kalau weekend anak-anak sama mertua sering ikut pergi bareng."
        ),
        "description":  "Kalimat panjang customer, konteks jelas",
    },
    {
        "label":        "OVERLAP",
        "text":         (
            "Silakan duduk Pak untuk tipe Pajero Sport harganya— "
            "ah iya terima kasih Mas brosurnya ditaruh situ aja."
        ),
        "description":  "Kalimat overlap sales–customer dalam satu utterance",
    },
    {
        "label":        "NORMAL_PANJANG_OVERLAP",
        "text":         (
            "Kemarin saya sempat riset sih di internet mengenai konsumsi bahan bakar "
            "Xpander Ultimate baru ini, nah betul sekali Pak irit banget, makanya saya mau "
            "mastiin apa benar diskon bulan ini bisa sampai dua puluh juta?"
        ),
        "description":  "Kalimat panjang dengan konteks overlap/konfirmasi",
    },
]

# ── Data class ─────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    scenario:          str
    model:             str
    run_index:         int
    text:              str
    speaker:           str
    confidence:        float
    cls_path:          str        # "heuristic" | "llm"
    latency_ms:        float      # 0.0 jika heuristic
    prompt_tokens:     int        # 0 jika heuristic
    completion_tokens: int        # 0 jika heuristic
    cost_usd:          float
    error:             str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── Core classifier ─────────────────────────────────────────────────────────────
async def classify_with_model(
    text: str,
    model: str,
    last_speaker: str,
    api_key: str,
) -> tuple[str, float, str, int, int, float]:
    """
    Jalankan classify_speaker dengan model tertentu.

    Returns:
        speaker, confidence, path, prompt_tokens, completion_tokens, latency_ms
    """
    clean = text.strip()
    if not clean:
        return last_speaker or "unknown", 0.0, "heuristic", 0, 0, 0.0

    # Heuristic path: ≤2 kata
    if len(clean.split()) <= _SHORT_TEXT_WORDS:
        opposite = (
            "customer" if last_speaker == "sales"
            else "sales" if last_speaker == "customer"
            else "unknown"
        )
        return opposite, 0.35, "heuristic", 0, 0, 0.0

    # LLM path
    last_label = (
        "SALES"    if last_speaker == "sales"
        else "CUSTOMER" if last_speaker == "customer"
        else "tidak diketahui"
    )

    user_prompt = (
        f"Riwayat percakapan terakhir:\n(belum ada percakapan)\n\n"
        f"Pembicara terakhir yang diketahui: {last_label}\n\n"
        f"Kalimat baru yang harus diklasifikasikan:\n\"{clean}\"\n\n"
        "Berikan confidence RENDAH (< 0.5) jika kalimat ambigu, "
        "kemungkinan overlap, atau konteks belum cukup."
    )

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=30,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    usage             = response.usage
    prompt_tokens     = usage.prompt_tokens     if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    raw  = response.choices[0].message.content.strip()
    data = json.loads(raw)

    speaker    = data.get("speaker", "unknown")
    confidence = float(data.get("confidence", 0.5))
    if speaker not in ("sales", "customer"):
        speaker = "unknown"

    return speaker, confidence, "llm", prompt_tokens, completion_tokens, latency_ms


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING[model]
    return prompt_tokens * p["input"] + completion_tokens * p["output"]


# ── Single run ─────────────────────────────────────────────────────────────────
async def run_single(
    scenario: dict,
    model: str,
    run_index: int,
    last_speaker: str,
    api_key: str,
) -> RunResult:
    try:
        speaker, conf, path, p_tok, c_tok, lat_ms = await classify_with_model(
            scenario["text"], model, last_speaker, api_key
        )
        cost = estimate_cost(model, p_tok, c_tok)
        return RunResult(
            scenario=scenario["label"],
            model=model,
            run_index=run_index,
            text=scenario["text"],
            speaker=speaker,
            confidence=conf,
            cls_path=path,
            latency_ms=lat_ms,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            cost_usd=cost,
        )
    except Exception as exc:
        return RunResult(
            scenario=scenario["label"],
            model=model,
            run_index=run_index,
            text=scenario["text"],
            speaker="unknown",
            confidence=0.0,
            cls_path="error",
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            error=str(exc),
        )


# ── Console output helpers ──────────────────────────────────────────────────────
W = 110

def _path_tag(path: str) -> str:
    return "[heuristic]" if path == "heuristic" else "[llm]      "


def print_run(r: RunResult) -> None:
    if r.error:
        print(f"    ✗ [{r.model:<14}] run {r.run_index:>2}  ERROR: {r.error}")
        return
    snippet = (r.text[:55] + "…") if len(r.text) > 55 else r.text
    ok = "✓"
    print(
        f"    {ok} [{r.model:<14}] run {r.run_index:>2}"
        f"  {_path_tag(r.cls_path)}"
        f"  lat={r.latency_ms:>7.1f}ms"
        f"  tok={r.total_tokens:>3}"
        f"  ${r.cost_usd:.7f}"
        f"  → {r.speaker:<8} conf={r.confidence:.2f}"
    )
    print(f"       \"{snippet}\"")


def _stat(values: list[float], unit: str = "") -> str:
    if not values:
        return "—"
    avg = mean(values)
    p50 = median(sorted(values))
    p95 = sorted(values)[max(0, int(len(values) * 0.95) - 1)]
    sd  = stdev(values) if len(values) > 1 else 0.0
    return f"avg={avg:>7.1f}{unit}  p50={p50:>7.1f}{unit}  p95={p95:>7.1f}{unit}  σ={sd:>5.1f}{unit}"


def print_summary(results: list[RunResult], models: list[str]) -> None:
    scenarios = [s["label"] for s in SCENARIOS]

    print(f"\n{'═' * W}")
    print("  RINGKASAN PERBANDINGAN — gpt-4.1-mini vs gpt-4.1")
    print(f"  {'Skenario':<26} {'Model':<15} {'Runs':>5} {'AvgLat(ms)':>11} "
          f"{'p95Lat(ms)':>11} {'AvgTokIn':>9} {'AvgTokOut':>10} {'AvgCost($)':>12}")
    print("  " + "─" * (W - 2))

    for scen in scenarios:
        for model in models:
            ok_runs = [
                r for r in results
                if r.scenario == scen and r.model == model and not r.error
            ]
            if not ok_runs:
                print(f"  {scen:<26} {model:<15} {'—':>5}")
                continue

            lats   = [r.latency_ms     for r in ok_runs]
            p_toks = [r.prompt_tokens  for r in ok_runs]
            c_toks = [r.completion_tokens for r in ok_runs]
            costs  = [r.cost_usd       for r in ok_runs]

            avg_lat  = mean(lats)
            p95_lat  = sorted(lats)[max(0, int(len(lats) * 0.95) - 1)]
            avg_ptok = mean(p_toks)
            avg_ctok = mean(c_toks)
            avg_cost = mean(costs)
            path_tag = ok_runs[0].cls_path  # heuristic runs all share same path

            path_note = " ← heuristic" if path_tag == "heuristic" else ""
            print(
                f"  {scen:<26} {model:<15} {len(ok_runs):>5}"
                f"  {avg_lat:>10.1f}  {p95_lat:>10.1f}"
                f"  {avg_ptok:>8.1f}  {avg_ctok:>9.1f}"
                f"  ${avg_cost:>10.7f}{path_note}"
            )

    # ── Selisih biaya & latency per skenario ───────────────────────────────────
    print(f"\n{'═' * W}")
    print("  DELTA gpt-4.1 vs gpt-4.1-mini  (positif = gpt-4.1 lebih mahal/lambat)")
    print(f"  {'Skenario':<26} {'ΔLat(ms)':>10} {'ΔTokIn':>8} {'ΔTokOut':>8} "
          f"{'ΔCost($)':>12} {'Cost ratio':>11}")
    print("  " + "─" * (W - 2))

    for scen in scenarios:
        for m_a, m_b in [("gpt-4.1-mini", "gpt-4.1")]:
            runs_a = [r for r in results if r.scenario == scen and r.model == m_a and not r.error]
            runs_b = [r for r in results if r.scenario == scen and r.model == m_b and not r.error]
            if not runs_a or not runs_b:
                continue

            lat_a  = mean(r.latency_ms         for r in runs_a)
            lat_b  = mean(r.latency_ms         for r in runs_b)
            tok_in_a  = mean(r.prompt_tokens     for r in runs_a)
            tok_in_b  = mean(r.prompt_tokens     for r in runs_b)
            tok_out_a = mean(r.completion_tokens for r in runs_a)
            tok_out_b = mean(r.completion_tokens for r in runs_b)
            cost_a = mean(r.cost_usd           for r in runs_a)
            cost_b = mean(r.cost_usd           for r in runs_b)

            d_lat  = lat_b  - lat_a
            d_in   = tok_in_b  - tok_in_a
            d_out  = tok_out_b - tok_out_a
            d_cost = cost_b - cost_a
            ratio  = (cost_b / cost_a) if cost_a > 0 else float("inf")

            ratio_str = f"{ratio:>10.2f}x" if cost_a > 0 else "      ∞ (heuristic)"
            print(
                f"  {scen:<26} {d_lat:>+10.1f} {d_in:>+8.1f} {d_out:>+8.1f}"
                f"  ${d_cost:>+10.7f}  {ratio_str}"
            )

    # ── Pricing reference ──────────────────────────────────────────────────────
    print(f"\n{'═' * W}")
    print("  Pricing reference (harga pasar 2025):")
    for model, p in PRICING.items():
        print(
            f"  {model:<15}: ${p['input'] * 1_000_000:.2f}/1M input  "
            f"${p['output'] * 1_000_000:.2f}/1M output"
        )
    print(f"  Heuristic path (≤{_SHORT_TEXT_WORDS} kata): 0 token, 0 ms, $0.0000000")
    print("═" * W)


def save_csv(results: list[RunResult], path: str) -> None:
    fields = [
        "scenario", "model", "run_index", "cls_path",
        "speaker", "confidence",
        "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
        "cost_usd", "error", "text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scenario":          r.scenario,
                "model":             r.model,
                "run_index":         r.run_index,
                "cls_path":          r.cls_path,
                "speaker":           r.speaker,
                "confidence":        f"{r.confidence:.3f}",
                "latency_ms":        f"{r.latency_ms:.3f}",
                "prompt_tokens":     r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens":      r.total_tokens,
                "cost_usd":          f"{r.cost_usd:.9f}",
                "error":             r.error,
                "text":              r.text,
            })


# ── Main ───────────────────────────────────────────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark classify_speaker: gpt-4o-mini vs gpt-4.1-mini"
    )
    parser.add_argument("--runs",        type=int, default=3,
                        help="Jumlah run per (skenario × model) (default: 3)")
    parser.add_argument("--last-speaker", default="sales",
                        choices=["sales", "customer", "unknown"],
                        help="Context last_speaker untuk prompt (default: sales)")
    parser.add_argument("--csv",         default="classifier_model_compare.csv",
                        help="Path file CSV output (default: classifier_model_compare.csv)")
    args = parser.parse_args()

    from backend.config import settings
    api_key = settings.openai_api_key

    models = ["gpt-4.1-mini", "gpt-4.1"]

    print("=" * W)
    print("  Benchmark: classify_speaker — gpt-4.1-mini vs gpt-4.1")
    print(f"  Last speaker context : {args.last_speaker}")
    print(f"  Runs per cell        : {args.runs} × {len(SCENARIOS)} skenario × {len(models)} model"
          f" = {args.runs * len(SCENARIOS) * len(models)} total panggilan API")
    print(f"  Heuristic threshold  : ≤{_SHORT_TEXT_WORDS} kata → skip LLM (0 token)")
    print("=" * W)

    all_results: list[RunResult] = []

    for sc in SCENARIOS:
        word_count = len(sc["text"].strip().split())
        is_heuristic = word_count <= _SHORT_TEXT_WORDS
        heur_note = " [HEURISTIC — skip LLM]" if is_heuristic else f" [{word_count} kata]"

        print(f"\n{'─' * W}")
        print(f"  SKENARIO: {sc['label']}{heur_note}")
        print(f"  Deskripsi: {sc['description']}")
        print(f"  Teks: \"{sc['text'][:90]}{'…' if len(sc['text']) > 90 else ''}\"")
        print("─" * W)

        for run_idx in range(1, args.runs + 1):
            print(f"\n  ── Run {run_idx}/{args.runs} ──")

            # Jalankan kedua model secara concurrent dalam satu run
            tasks = [
                run_single(sc, model, run_idx, args.last_speaker, api_key)
                for model in models
            ]
            run_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in run_results:
                if isinstance(res, Exception):
                    print(f"    ✗ Unexpected: {res}")
                    continue
                all_results.append(res)
                print_run(res)

            # Jeda kecil untuk menghindari rate limit (skip di run terakhir)
            if run_idx < args.runs:
                await asyncio.sleep(0.3)

    # ── Summary table ──────────────────────────────────────────────────────────
    print_summary(all_results, models)

    # ── CSV export ─────────────────────────────────────────────────────────────
    name_file = "classifier_model_compare2.csv"
    save_csv(all_results, name_file)
    print(f"\n  Hasil disimpan: {name_file}  ({len(all_results)} baris)")


if __name__ == "__main__":
    asyncio.run(main())
