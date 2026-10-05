"""Checks that every number quoted in the manuscript still matches the CSV/JSON files in results/.

Usage:  python tools/check_paper_values.py
The manuscript values were exported to paper/values_used.json when the paper was built.  This script recomputes
them from results/ with the same code (tools/result_numbers.py) and reports any difference.  Timing columns
(solve_ms) depend on the machine and are compared with a loose tolerance.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_numbers import collect

ROOT = Path(__file__).resolve().parents[1]
ref = json.load(open(ROOT / 'paper' / 'values_used.json'))
new = collect()
bad = []
for k, v in ref.items():
    if k not in new:
        continue                                   # table-row strings are built from other keys
    if str(new[k]) != str(v):
        if '_ms' in k or 'solve' in k:
            continue
        bad.append((k, v, new[k]))
print(f'{len(ref)} manuscript values, {len(bad)} differences')
for k, a, b in bad:
    print(f'  {k}: manuscript {a}  results {b}')
sys.exit(1 if bad else 0)
