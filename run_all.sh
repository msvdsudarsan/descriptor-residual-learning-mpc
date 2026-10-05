#!/usr/bin/env bash
# Reproduces every table and figure of the manuscript (run time depends on the hardware).
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/test_consistency.py
python3 code/scalar_experiments.py     > results/scalar_log.txt
python3 code/network_experiments.py    > results/network_log.txt
python3 code/network_tube_test.py      > results/tube_log.txt
python3 code/network_extra_tests.py    > results/extra_log.txt
python3 code/network_transfer_test.py  > results/transfer_log.txt
python3 code/network_analytic_gradient.py > results/analytic_log.txt
python3 code/network_validity_gate.py case9  > results/validity_gate_case9_log.txt
python3 code/network_validity_gate.py case39 > results/validity_gate_case39_log.txt
python3 code/network_reconcile.py case9  > results/reconcile_case9_log.txt
python3 code/network_reconcile.py case39 > results/reconcile_case39_log.txt
python3 code/network_recovery_closed_loop.py > results/recovery_cl_log.txt
python3 code/network_nuisance.py > results/nuisance_log.txt
python3 code/network_heldout.py > results/heldout_log.txt
python3 code/make_reconcile_figures.py
python3 tools/regularity_chain_check.py > results/chain_log.txt
python3 code/make_pipeline_figure.py
python3 tools/check_paper_values.py
echo "Done. Tables: results/*.csv, figures: figures/*.pdf"
