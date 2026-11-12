#!/usr/bin/env bash
# Experiment A: fresh test seeds x several independent nominal calibration draws, case39 (all non-bridge single-line outages), per-run scores recorded.
W="$(cd "$(dirname "$0")" && pwd)"; REPO="$W/.."; export PYTHONPATH="$REPO/code" OMP_NUM_THREADS=1
cd "$REPO"
i=0
for cs in 5001 5002 5003 5004; do
  so=$((40000+10000*i)); i=$((i+1))
  CONT_OUT=$W/results_setA SEEDOFF=$so CALSEED=$cs python $W/contingency_setA.py case39 full > $W/logs/expA_cal${cs}_so${so}.log 2>&1
done
CONT_OUT=$W/results_setA SEEDOFF=90000 CALSEED=5001 python $W/contingency_setA.py case9 full > $W/logs/expA_case9.log 2>&1
echo DONE > $W/logs/expA_done
