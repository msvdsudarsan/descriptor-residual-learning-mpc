#!/bin/bash
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for a in "case39 pq" "case9 z" "case39 z"; do set -- $a; timeout 1150 python3 ac_pipeline.py $1 $2 > ../results/$1_ac_$2_pipeline.log 2>&1; done
