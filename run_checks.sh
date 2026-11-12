#!/usr/bin/env bash
# One entry point for the  contingency experiments.  Exit status 0 only if every step passes.
#   ./run_checks.sh            default: unit tests + regenerate tables/CSV from the stored raw JSON + all value checks   (~1-2 min)
#   (call as `bash run_checks.sh --quick` if the executable bit is lost)
#   ./run_checks.sh --quick    default + fresh rerun of case9 full outages (original seeds) and exact comparison with stored JSON
#   ./run_checks.sh --full     default + complete rerun of ALL contingency sets into a temp dir and comparison (hours on 2 cores)
set -euo pipefail
cd "$(dirname "$0")"; export PYTHONPATH=code OMP_NUM_THREADS=1; MODE="${1:-default}"; T0=$(date +%s)
echo "== 1. unit tests"; python3 -m pytest -q tests 2>&1 | tail -3
echo "== 2. regenerate tables / CSV / values_contingency.json from raw JSON"; python3 extensions/contingency/make_tables.py
echo "== 3. value checks"; python3 extensions/contingency/check_contingency_values.py
echo "== 3b. direct per-line calibration: analysis of stored scores + value checks"; python3 extensions/contingency/analyze_direct.py > /dev/null; python3 extensions/contingency/check_direct_values.py | tail -3
if [[ "$MODE" == "--quick" || "$MODE" == "--full" ]]; then
  TMP=$(mktemp -d); export CONT_OUT="$TMP"
  echo "== 4. rerun case9 full outages (original seeds) -> $TMP"; python3 extensions/contingency/run_contingency.py case9 full > "$TMP/case9_full.log" 2>&1
  python3 extensions/contingency/compare_rerun.py "$TMP"
fi
if [[ "$MODE" == "--full" ]]; then
  for c in case9 case39; do for m in partial full; do SEEDOFF=0 python3 extensions/contingency/run_contingency.py $c $m > "$TMP/$c.$m.log" 2>&1; done; done
  SEEDOFF=10000 python3 extensions/contingency/run_contingency.py case9 full >/dev/null 2>&1; SEEDOFF=10000 python3 extensions/contingency/run_contingency.py case39 full >/dev/null 2>&1
  SEEDOFF=20000 MISMATCH=0.05 python3 extensions/contingency/run_contingency.py case9 full >/dev/null 2>&1; SEEDOFF=20000 MISMATCH=0.05 python3 extensions/contingency/run_contingency.py case39 full >/dev/null 2>&1
  python3 extensions/contingency/compare_rerun.py "$TMP"
  mkdir -p "$TMP/direct"; DIRECT_OUT="$TMP/direct" python3 extensions/contingency/direct_percontingency.py 0 1 > "$TMP/direct.log" 2>&1
  python3 extensions/contingency/check_direct_values.py "$TMP/direct" | tail -3
fi
echo "RUN_CHECKS COMPLETED OK (mode=$MODE) in $(( $(date +%s)-T0 )) s"
