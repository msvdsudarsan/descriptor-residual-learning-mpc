#!/bin/bash
# Runs the 24 operating points of PROTOCOL.md (two processes in parallel); logs in extensions/ac_recal/logs.  From the repository root.
export OMP_NUM_THREADS=1 PYTHONPATH=code:extensions/ac_baselines
mkdir -p extensions/ac_recal/logs extensions/ac_recal/results
run() { for lam in 0.7 0.8 0.9 1.0 1.1 1.2; do timeout 3000 python3 extensions/ac_recal/ac_recal_comparison.py $1 $2 $lam 12 > extensions/ac_recal/logs/$1_$2_lam$lam.log 2>&1; done; }
(run case9 pq; run case9 z) &
(run case39 pq; run case39 z) &
wait
echo done > extensions/ac_recal/logs/ALL_DONE
