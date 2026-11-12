# Reproducing the regime-map outputs (Protocol)
Run from the package root (python3, numpy, scipy). OMP_NUM_THREADS=1 recommended.
1. Validation: `python3 code/network_regime_map.py case9 V` (and case39) -> results/regime_map/<case>_validation.json
2. Prediction: `python3 code/network_regime_map.py <case> PRED`
3. Closed loop: `python3 code/network_regime_map.py <case> CL`
4. Statistics: `python3 tools/analyze_regime_pred.py <case>`; `python3 tools/analyze_regime_cl.py <case>`
5. V-stage disclosures: `python3 tools/v_stage_disclosures.py`
6. Report: `python3 tools/make_regime_report.py` (reads only stored CSV/JSON)
7. Independent verdict recheck (unrounded): `python3 tools/recheck_regime_verdicts.py` -> results/regime_map/verdict_recheck.csv
Steps 4, 6, 7 need only the stored CSV/JSON files. Steps 1-3 need the full package code (code/network_*.py) and the stored coefficient files. Exit status of case39 CL: results/regime_map/case39_CL_exit_status.txt.
