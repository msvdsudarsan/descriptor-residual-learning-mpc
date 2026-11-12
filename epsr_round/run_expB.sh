#!/usr/bin/env bash
W="$(cd "$(dirname "$0")" && pwd)"; REPO="$W/.."; export PYTHONPATH="$REPO/code" OMP_NUM_THREADS=1
cd "$REPO"

CONT_OUT=$W/results_setB SEEDOFF=100000 python $W/contingency_setB.py case39 full > $W/logs/expB_case39.log 2>&1
CONT_OUT=$W/results_setB SEEDOFF=110000 CALSEED=5002 python $W/contingency_setB.py case39 full > $W/logs/expB_case39_cal5002.log 2>&1
CONT_OUT=$W/results_setB SEEDOFF=120000 python $W/contingency_setB.py case9 full > $W/logs/expB_case9.log 2>&1
CONT_OUT=$W/results_setB SEEDOFF=130000 MISMATCH=0.05 python $W/contingency_setB.py case39 full > $W/logs/expB_case39_mis.log 2>&1
echo DONE > $W/logs/expB_done
