#!/usr/bin/env bash
# Reproduces every table and figure of the manuscript (run time depends on the hardware).
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/test_consistency.py
python3 tests/test_standard_facts.py
python3 code/scalar_experiments.py     > results/scalar_log.txt
python3 code/network_experiments.py    > results/network_log.txt
python3 code/network_tube_test.py      > results/tube_log.txt
python3 code/network_extra_tests.py    > results/extra_log.txt
python3 code/network_transfer_test.py  > results/transfer_log.txt
python3 code/network_analytic_gradient.py > results/analytic_log.txt
python3 code/network_baselines_loaddyn.py        > results/baselines_log.txt        # about 3 minutes
python3 code/network_baselines_loaddyn.py noise  > results/baselines_noise_log.txt  # about 3 minutes
python3 code/make_baseline_figure.py
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
python3 code/worked_example_case9.py > results/worked_example_log.txt
python3 tests/test_worked_example.py
python3 code/application_example_case39.py > results/application_log.txt
python3 code/worked_example_case39.py > results/worked_example_case39_log.txt
python3 tools/detectability_check.py > results/detectability_log.txt
python3 tools/gradient_clip_check.py
python3 code/application_case39.py scan > results/case39_application_scan_log.txt        # about 30 minutes
for sc in 1.0 3.0 6.0; do python3 code/application_case39.py authority $sc > results/case39_application_authority_x$sc.log; done   # about 1 hour
python3 code/merge_authority.py
python3 code/make_application_figure.py
python3 tests/test_application.py
python3 -m pytest -q tests/test_audit_fixes.py
python3 tools/closed_loop_intervals.py
python3 code/network_conformal_independent.py case9 > results/conformal_independent_case9_log.txt     # about 4 minutes
python3 code/network_conformal_independent.py case39 > results/conformal_independent_case39_log.txt   # about 8 minutes
python3 code/network_cl_recalibration.py case9 20 200 > results/cl_recal_case9_log.txt               # about 5 minutes
python3 code/network_cl_recalibration.py case39 16 100 > results/cl_recal_case39_log.txt              # about 10 minutes
python3 code/network_param_baseline.py case9 > results/param_baseline_case9_log.txt                  # about 2 minutes
python3 code/network_param_baseline.py case39 > results/param_baseline_case39_log.txt
python3 code/make_calibration_tables.py > /dev/null
python3 tools/check_calibration_values.py | tee results/run_all_check_log.txt
python3 -m pytest -q tests/test_calibration.py
python3 -m pytest -q tests/test_props_independent.py
python3 tools/check_paper_values.py | tee -a results/run_all_check_log.txt
echo "Done. Tables: results/*.csv, figures: figures/*.pdf"
