"""
conftest.py
===========
Pytest hooks untuk menyimpan hasil test ke file CSV.

Hasil disimpan di: test_results/<nama_file_test>_results.csv
Kolom: test_id, status, durasi_detik, pesan_gagal, timestamp
"""

import csv
import time
import pytest
from datetime import datetime, timezone
from pathlib import Path


_results: list[dict] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Tangkap hasil setiap test (setup / call / teardown)."""
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    # Hanya rekam fase "call" (eksekusi test yang sesungguhnya)
    if report.when != "call":
        return

    _results.append({
        "test_id":       report.nodeid,
        "status":        report.outcome.upper(),   # PASSED / FAILED / ERROR
        "durasi_detik":  round(report.duration, 4),
        "pesan_gagal":   _extract_message(report),
        "timestamp":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


def _extract_message(report: pytest.TestReport) -> str:
    """Ambil pesan singkat dari laporan gagal; kosong kalau PASSED."""
    if report.passed:
        return ""
    if report.failed and report.longrepr:
        # longrepr bisa berupa string atau objek ReprExceptionInfo
        text = str(report.longrepr)
        # Ambil baris terakhir yang bermakna (biasanya AssertionError / Exception)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text[:200]
    return ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    """Tulis semua hasil ke CSV setelah seluruh sesi selesai."""
    if not _results:
        return

    # Tentukan nama file berdasarkan test file pertama yang dijalankan
    first_nodeid = _results[0]["test_id"]
    test_file = first_nodeid.split("::")[0].replace("/", "_").replace(".py", "")
    output_dir = Path(session.config.rootpath) / "test_results"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"{test_file}_results.csv"

    fieldnames = ["test_id", "status", "durasi_detik", "pesan_gagal", "timestamp"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_results)

    # Ringkasan ke terminal
    passed = sum(1 for r in _results if r["status"] == "PASSED")
    failed = len(_results) - passed
    print(f"\n📄 Hasil test disimpan → {csv_path}")
    print(f"   {passed} PASSED  |  {failed} FAILED  |  total {len(_results)}")
