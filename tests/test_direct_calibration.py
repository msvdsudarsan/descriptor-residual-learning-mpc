"""Stored-output checks of the direct per-line calibration experiment (no simulation): the quoted values match results/direct_cal_summary.json,
the 35 outages are present, and calibration/test seeds are disjoint."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'extensions/contingency/results'
def test_values_and_leakage():
    r = subprocess.run([sys.executable, str(ROOT / 'extensions/contingency/check_direct_values.py')], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:]
def test_all_35_outages_valid_and_disjoint():
    files = sorted((R / 'direct_cal').glob('line_*.json')); assert len(files) == 35
    for f in files:
        d = json.load(open(f)); assert d['valid']
        assert len({d['cal']['seed'], d['test0']['seed'], d['test1']['seed']}) == 3
        assert len(d['cal']['score']) == 40 and len(d['test0']['score']) == 20 and len(d['test1']['score']) == 20
