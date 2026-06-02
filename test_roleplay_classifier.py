"""
Role-play Test: Speaker Classifier + AI Recommendation
========================================================
Simulasi percakapan sales-customer untuk menguji:
  1. Akurasi speaker classifier (predicted vs ground truth)
  2. Kualitas AI hint + rekomendasi mobil di setiap tahap percakapan

Scenarios:
  1. Keluarga besar + medan berat           → ekspektasi: Xpander Cross / Pajero
  2. Customer-led: sebut Pajero duluan      → ekspektasi: langsung ke detail produk
  3. Budget ketat + pemakaian kota          → ekspektasi: Xpander entry/mid variant
  4. Overlap/reaksi pendek                  → ekspektasi: heuristic short-text

Usage:
    uv run python test_roleplay_classifier.py              # semua scenario, tanpa AI
    uv run python test_roleplay_classifier.py --ai         # + AI analysis di setiap turn
    uv run python test_roleplay_classifier.py --scenario 1 --ai
    uv run python test_roleplay_classifier.py --demo       # pakai demo mode (tanpa API)
"""

import asyncio
import argparse
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field

from backend.models.schemas import Utterance, ConversationContext
from backend.services.speaker_classifier import classify_speaker
from backend.services.ai import analyze_conversation


# ── Tipe data ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    speaker: str   # ground truth: "sales" | "customer"
    text:    str
    note:    str = ""   # keterangan tambahan untuk output


@dataclass
class TurnResult:
    turn:            Turn
    turn_no:         int
    pred_speaker:    str
    confidence:      float
    correct:         bool
    short_heuristic: bool   # True jika diproses heuristic (≤4 kata)


# ── Scenarios ──────────────────────────────────────────────────────────────────

SCENARIOS: dict[int, tuple[str, list[Turn]]] = {

    1: ("Keluarga besar + luar kota (basa-basi dulu)", [
        Turn("sales",    "Selamat pagi, selamat datang di showroom Mitsubishi!"),
        Turn("customer", "Selamat pagi."),
        Turn("sales",    "Ada yang bisa saya bantu Pak?"),
        Turn("customer", "Iya, saya lagi cari-cari mobil untuk keluarga."),
        Turn("sales",    "Wah, boleh saya tahu biasanya pergi berapa orang Pak?"),
        Turn("customer", "Biasanya 5 sampai 6 orang, ada anak kecil juga dua."),
        Turn("sales",    "Sering juga ke luar kota atau lebih banyak dalam kota?"),
        Turn("customer", "Lumayan sering, ke Bandung, ke kampung. Jalannya kadang tidak bagus."),
        Turn("sales",    "Baik, untuk budget yang disiapkan kira-kira berapa Pak?"),
        Turn("customer", "Sekitar 250 sampai 300 juta kalau bisa."),
        Turn("sales",    "Oke, boleh saya tunjukkan dua pilihan yang paling cocok?"),
        Turn("customer", "Boleh, silakan."),
    ]),

    2: ("Customer-led: sebut Pajero duluan", [
        Turn("sales",    "Selamat siang, ada yang bisa saya bantu?"),
        Turn("customer", "Siang, saya mau lihat Pajero Sport dong."),
        Turn("sales",    "Tentu Pak, Pajero Sport kita ada beberapa varian. Bapak sendiri atau untuk keluarga?"),
        Turn("customer", "Untuk keluarga, tapi saya juga sering offroad."),
        Turn("sales",    "Wah menarik, biasanya ke mana Pak kalau offroad?"),
        Turn("customer", "Ke Bromo, Semeru, sering camping. Jalannya ekstrem."),
        Turn("sales",    "Nah kalau begitu 4WD penting ya Pak. Budget sekitar berapa Pak?"),
        Turn("customer", "Sekitar 600 sampai 700 juta."),
        Turn("sales",    "Baik, ada pilihan yang sangat cocok untuk Bapak."),
        Turn("customer", "Oh ya? Apa bedanya sama yang biasa?"),
    ]),

    3: ("Budget ketat + pemakaian dalam kota", [
        Turn("sales",    "Selamat sore, ada yang bisa saya bantu Bu?"),
        Turn("customer", "Sore, saya mau cari mobil buat harian."),
        Turn("sales",    "Pemakaiannya lebih banyak ke mana Bu, kantor atau antar jemput?"),
        Turn("customer", "Antar jemput anak, ke pasar, dalam kota aja."),
        Turn("sales",    "Anaknya berapa Bu?"),
        Turn("customer", "Dua orang, masih SD."),
        Turn("sales",    "Kalau begitu penumpang maksimal berapa orang biasanya?"),
        Turn("customer", "Paling 4 sampai 5 orang."),
        Turn("sales",    "Budget Bu kalau boleh tahu?"),
        Turn("customer", "Paling 200 juta, mau yang irit bensin juga."),
        Turn("sales",    "Oke, ada pilihan yang pas untuk kebutuhan Ibu."),
        Turn("customer", "Yang tipe apa ya kira-kira?"),
    ]),

    4: ("Overlap dan reaksi pendek (heuristic test)", [
        Turn("sales",    "Selamat datang di Mitsubishi!"),
        Turn("customer", "Iya."),          # ≤4 kata
        Turn("sales",    "Ada yang bisa saya bantu?"),
        Turn("customer", "Mau lihat-lihat dulu."),   # ≤4 kata
        Turn("sales",    "Silakan Pak, kalau ada yang ingin ditanyakan saya siap membantu ya."),
        Turn("customer", "Oh oke oke."),   # ≤4 kata
        Turn("sales",    "Ini ada Xpander terbaru Pak, banyak fitur baru."),
        Turn("customer", "Wah bagus juga ya.", note="ambiguous — short but clear customer"),
        Turn("sales",    "Iya Pak, interior-nya baru dirombak. Bapak biasanya pergi sama siapa?"),
        Turn("customer", "Sama istri dan anak, tiga orang."),
    ]),
}


# ── Print helpers ──────────────────────────────────────────────────────────────

W = 80

def hr(char="─"): print(char * W)
def hr2(char="═"): print(char * W)

SPEAKER_COLOR = {"sales": "\033[36m", "customer": "\033[33m", "unknown": "\033[90m"}
RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
GREY  = "\033[90m"
BOLD  = "\033[1m"

def _label(speaker: str) -> str:
    color = SPEAKER_COLOR.get(speaker, "")
    return f"{color}{speaker.upper():<8}{RESET}"


def print_turn_result(r: TurnResult):
    heuristic_tag = f"{GREY}[heuristic]{RESET}" if r.short_heuristic else ""
    correct_tag   = f"{GREEN}✓{RESET}" if r.correct else f"{RED}✗{RESET}"
    conf_bar      = "█" * int(r.confidence * 10) + "░" * (10 - int(r.confidence * 10))

    note = f"  {GREY}↳ {r.turn.note}{RESET}" if r.turn.note else ""
    print(f"  Turn {r.turn_no:>2}  {_label(r.turn.speaker)}  \"{r.turn.text[:60]}\"")
    print(f"          Pred: {_label(r.pred_speaker)} {conf_bar} {r.confidence:.2f}  {correct_tag} {heuristic_tag}{note}")


def print_ai_result(hint, cars, turn_no: int):
    print()
    hr("·")
    print(f"  {BOLD}AI Analysis setelah Turn {turn_no}{RESET}")
    hr("·")
    print(f"  Hint     : {hint.hint_text}")
    print(f"  Question : {BOLD}{hint.suggested_question}{RESET}")
    if hint.probe_topics:
        print(f"  Probe    : {' | '.join(hint.probe_topics[:3])}")
    if hint.detected_needs:
        for need in hint.detected_needs[:3]:
            print(f"  Need     : {need}")
    if cars.cars:
        for car in cars.cars:
            print(f"  Car      : {car.brand} {car.model} {car.variant}  →  Rp {car.price_otr_jakarta:,}")
        print(f"  Reason   : {cars.reason[:80]}")
    else:
        print(f"  Cars     : (belum ada rekomendasi)")
    print(f"  Reason   : {cars.reason[:80]}")


# ── Core runner ────────────────────────────────────────────────────────────────

async def run_scenario(
    scenario_id: int,
    run_ai: bool,
    ai_every: int,
    demo_mode: bool,
) -> dict:
    title, turns = SCENARIOS[scenario_id]

    hr2()
    print(f"  {BOLD}SCENARIO {scenario_id}: {title}{RESET}")
    hr2()
    print()

    # Override app_mode if needed
    from backend.config import settings
    original_mode = settings.app_mode
    if demo_mode:
        settings.__dict__["app_mode"] = "demo"

    context = ConversationContext(session_id=f"test-{scenario_id}")
    results: list[TurnResult] = []
    last_speaker = "unknown"

    for i, turn in enumerate(turns, start=1):
        # Classifier: pakai history SEBELUM utterance ini
        history = context.utterances[-5:]
        pred_speaker, confidence = await classify_speaker(turn.text, history, last_speaker)

        word_count = len(turn.text.strip().split())
        is_heuristic = word_count <= 4

        correct = pred_speaker == turn.speaker

        result = TurnResult(
            turn=turn, turn_no=i,
            pred_speaker=pred_speaker, confidence=confidence,
            correct=correct, short_heuristic=is_heuristic,
        )
        results.append(result)
        print_turn_result(result)

        # Masukkan ke context dengan ground truth speaker
        context.utterances.append(Utterance(
            id=str(uuid.uuid4()),
            speaker=turn.speaker,
            text=turn.text,
            timestamp=datetime.now(timezone.utc),
            confidence=confidence,
        ))

        # Update last_speaker (ground truth untuk test yang bersih)
        if turn.speaker in ("sales", "customer"):
            last_speaker = turn.speaker
        context.last_speaker = last_speaker

        # AI analysis setiap N turn (dan di turn terakhir)
        if run_ai and (i % ai_every == 0 or i == len(turns)):
            try:
                hint, cars = await analyze_conversation(context)
                print_ai_result(hint, cars, i)
                context.asked_questions.append(hint.suggested_question)
            except Exception as e:
                print(f"\n  {RED}[AI ERROR] {e}{RESET}\n")

        print()

    # Restore mode
    if demo_mode:
        settings.__dict__["app_mode"] = original_mode

    # Summary
    total = len(results)
    correct_count   = sum(1 for r in results if r.correct)
    heuristic_count = sum(1 for r in results if r.short_heuristic)
    wrong = [r for r in results if not r.correct]

    hr()
    print(f"  {BOLD}HASIL SCENARIO {scenario_id}{RESET}")
    hr()
    print(f"  Classifier  : {correct_count}/{total} benar ({correct_count/total*100:.0f}%)")
    print(f"  Heuristic   : {heuristic_count} utterance pendek (≤4 kata, skip LLM)")
    if wrong:
        print(f"  Salah       : {len(wrong)} utterance")
        for r in wrong:
            print(f"    Turn {r.turn_no}: actual={r.turn.speaker}  pred={r.pred_speaker}  conf={r.confidence:.2f}")
            print(f"      \"{r.turn.text[:70]}\"")
    print()

    return {
        "scenario_id":    scenario_id,
        "total":          total,
        "correct":        correct_count,
        "heuristic":      heuristic_count,
        "wrong_turns":    [r.turn_no for r in wrong],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=None,
                        help="Jalankan satu scenario (1-4). Default: semua.")
    parser.add_argument("--ai",       action="store_true",
                        help="Jalankan AI analyze di setiap tahap percakapan.")
    parser.add_argument("--ai-every", type=int, default=2,
                        help="Frekuensi AI analysis (setiap N turn). Default: 2.")
    parser.add_argument("--demo",     action="store_true",
                        help="Pakai app_mode=demo (tanpa Featherless API).")
    args = parser.parse_args()

    scenario_ids = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    all_results = []

    for sid in scenario_ids:
        result = await run_scenario(
            scenario_id=sid,
            run_ai=args.ai,
            ai_every=args.ai_every,
            demo_mode=args.demo,
        )
        all_results.append(result)
        print()

    # Global summary
    if len(all_results) > 1:
        hr2()
        print(f"  {BOLD}RINGKASAN SEMUA SCENARIO{RESET}")
        hr2()
        total_all   = sum(r["total"]   for r in all_results)
        correct_all = sum(r["correct"] for r in all_results)
        heuristic_all = sum(r["heuristic"] for r in all_results)
        print(f"  Total utterance : {total_all}")
        print(f"  Classifier acc  : {correct_all}/{total_all} ({correct_all/total_all*100:.0f}%)")
        print(f"  Heuristic (skip): {heuristic_all}")
        print()
        for r in all_results:
            acc = r["correct"] / r["total"] * 100
            icon = "✓" if acc >= 80 else "~" if acc >= 60 else "✗"
            title = SCENARIOS[r["scenario_id"]][0][:45]
            print(f"  {icon} Scenario {r['scenario_id']}: {acc:3.0f}%  {title}")
        hr2()


if __name__ == "__main__":
    asyncio.run(main())
