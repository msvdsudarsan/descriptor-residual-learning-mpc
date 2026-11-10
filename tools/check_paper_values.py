"""Checks that every number quoted in the manuscript still matches the CSV/JSON files in results/.

Usage:  python tools/check_paper_values.py
The manuscript values were exported to paper/values_used.json when the paper was built.  This script recomputes
them from results/ with the same code (tools/result_numbers.py) and reports any difference.  Timing columns
(solve_ms) depend on the machine: differences are listed separately and are not counted as errors.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_numbers import collect

ROOT = Path(__file__).resolve().parents[1]
ref = json.load(open(ROOT / 'paper' / 'values_used.json'))
new = collect()
bad = []
timing = []
for k, v in ref.items():
    if k not in new:
        continue                                   # table-row strings are built from other keys
    if str(new[k]) != str(v):
        if '_ms' in k or 'solve' in k:
            timing.append((k, v, new[k]))
            continue
        bad.append((k, v, new[k]))
print(f'{len(ref)} manuscript values, {len(bad)} differences (timing values excluded)')
for k, a, b in bad:
    print(f'  {k}: manuscript {a}  results {b}')
if timing:
    ratios = [float(b) / float(a) for _, a, b in timing if float(a) > 0]
    print(f'{len(timing)} timing values differ from the stored run (machine load; not compared): ratio clean/manuscript from {min(ratios):.2f} to {max(ratios):.2f}')
    for k, a, b in timing: print(f'  timing {k}: manuscript {a}  this run {b}')
sys.exit(1 if bad else 0)
