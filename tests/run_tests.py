"""Runs every test file in its own process and prints the time of each.  Pure Python (no Numba) needs about 20 s in total on one core; with a slow or shared
core allow several minutes.  Usage: python tests/run_tests.py [timeout_seconds_per_file]"""
import subprocess, sys, time
from pathlib import Path
T = Path(__file__).resolve().parent; lim = float(sys.argv[1]) if len(sys.argv) > 1 else 1800.0; bad = 0
for f in sorted(T.glob('test_*.py')):
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=lim); ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    print(f'{"PASS" if ok else "FAIL"} {f.name} {time.time() - t0:.1f}s'); bad += (not ok)
sys.exit(1 if bad else 0)
