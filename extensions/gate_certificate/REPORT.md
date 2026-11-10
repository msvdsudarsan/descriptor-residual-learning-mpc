# Gate theory (D2) and closed-loop finite-sample certificate (D3) - REPORT

All numbers below come from scripts in `. with logs in `.../logs/` and raw arrays in `.../out/`. Repo code was copied to `w/code`, `w/tools` (never edited; `final/` untouched). Python 3.13, numpy 2.5.3, scipy 1.18.1, OMP_NUM_THREADS=1, one process at a time, every run < 20 min.
Reproduction of the paper's own thresholds: `calibrate_q0` run in the copy returns q0 = 0.023839465 (case9) and 0.037060696 (case39), identical to the stored values (log `logs/D2a_analysis.log`).

Scripts: `null_pool.py`, `null_pool_T16.py`, `analyze_null.py` (D2a); `step_delay.py`, `analyze_steps.py` (D2b); `d3_run.py`, `analyze_d3.py` (D3); helpers `code/gatelib.py`.
Logs: `D2a_null_pool.log`, `D2a_analysis.log`, `D2b_step_case9.log`, `D2b_step_case39.log`, `D2b_analysis_case9.log`, `D2b_analysis_case39.log`, `D3_run_lim23.log`, `D3_run_lim30b.log` (the first half of the limit-30 job was killed by a background time limit after lambda = 0, 0.5, 1, 1.5 had been saved; lambda = 2, 3 were rerun with the same seeds; the .npy files in `out/` are the data), `D3_analysis.log`.

---------------------------------------------------------------------------------------------------

## D2a. Per-run false-alarm guarantee of the gate threshold

### Statement (what exchangeability gives, and nothing more)
Let Z_1..Z_n be the maxima over a null run of the smoothed one-step innovation score (smoothed = mean of the last M_SMOOTH=3 scores, run length 8 s = 80 samples), computed on n independent runs drawn from one null generator (initial condition, load law, measurement noise), and Z_{n+1} the same statistic for a new run from the same generator. They are i.i.d., hence exchangeable. Let q0 be the k-th smallest of Z_1..Z_n, k = ceil((n+1)(1-a)), if k <= n. Then
  P(Z_{n+1} <= q0) >= k/(n+1) = ceil((n+1)(1-a))/(n+1),   with equality if there are no ties.
The probability is joint over the calibration set and the new run (marginal), not conditional on the calibration set. "Alarm" means smoothed score > q0 (q > 1) somewhere in the run. The event "tau < 0.5" is a subset of the alarm event, so false fallback <= false alarm.

### Two points the code/paper must respect
1. **With n = 10 and a = 0.05 the 95% guarantee does not exist.** k = ceil(11 x 0.95) = 11 > n, `conformal_quantile` silently returns the sample maximum (`min(k,n)-1`), and the guarantee is n/(n+1) = 10/11 = 0.909 (false alarm <= 0.0909), not 0.95. The smallest n that gives a = 0.05 is n = 19.
2. The guarantee is marginal. For the sample maximum of n runs the conditional false-alarm rate F is Beta(1,n)-distributed, so P(F > a) = (1-a)^n; a training-conditional (PAC) statement "false alarm <= a with prob >= 1-d" needs n >= ln d / ln(1-a), i.e. n >= 45 for a = 0.05, d = 0.10 (exact formula, no simulation).

### Monte Carlo (fresh null runs, seeds 100000+, noise seeds 5100000+; disjoint from the repo calibration seeds 9001/9002)
Pools: case9 N = 5000, case39 N = 2500 fresh null runs, generator = repo `make_shift_runs` (lamp_hi = 0.08, T = 8 s, SIG_MEAS = 1e-3), per-run seed.

**Repo threshold (n_cal = 10), fresh runs:**
| case | q0 | false alarms / fresh runs | freq | Clopper-Pearson 95% | marginal guarantee |
|---|---|---|---|---|---|
| case9 | 0.02384 | 100/5000 | 0.0200 | [0.0163, 0.0243] | <= 0.0909 |
| case39 | 0.03706 | 303/2500 | 0.1212 | [0.1087, 0.1346] | <= 0.0909 |
case39's q0 sits above 1/11. That is not a violation: it is the conditional rate of one particular calibration draw (the repo's 10 runs); under the theory a draw with rate at least that high has probability (1-0.1212)^10 = 0.275. The large-sample 95% quantile of the pool is 0.0216 (case9) and 0.0446 (case39); repo q0 / that = 1.10 and 0.83.

**Marginal check over calibration draws** (two designs). (i) Permutation: 20000 random n-subsets of the pool as calibration set, the rest as test; (ii) disjoint blocks (n calibration runs + 1 test run, independent blocks, so the Clopper-Pearson interval is exact for the marginal probability).
| n_cal | guarantee (false alarm <=) | case9 permutation mean | case9 blocks (CP95) | case39 permutation mean | case39 blocks (CP95) |
|---|---|---|---|---|---|
| 10 | 0.0909 | 0.0909 | 41/454 = 0.090 [0.066, 0.121] | 0.0911 | 26/227 = 0.114 [0.076, 0.163] |
| 19 | 0.0500 | 0.0496 | 13/250 = 0.052 [0.028, 0.087] | 0.0500 | 3/125 = 0.024 [0.005, 0.069] |
| 39 | 0.0500 | 0.0503 | 8/125 = 0.064 [0.028, 0.122] | 0.0503 | 2/62 = 0.032 [0.004, 0.112] |
| 99 | 0.0500 | 0.0500 | 3/50 = 0.060 [0.013, 0.166] | 0.0499 | 0/25 = 0.000 [0.000, 0.137] |
All intervals contain the guaranteed value; the permutation means reproduce it to three decimals (this is the rank argument, a sanity check of the code and of exchangeability of the pool). Spread of the conditional rate across calibration draws (permutation): n = 10: sd 0.083, 5%-95% range 0.005 to 0.258, P(rate > 0.05) = 0.60; n = 99: range 0.020 to 0.091, P(rate > 0.05) = 0.44. So "marginal 5%" does not mean "this q0 has 5%".

**False fallback (tau < 0.5, needs q > 1 + ln2/a)** with the repo q0: a = 0.25 (chosen constant): 0/5000 [0, 0.0007] case9, 0/2500 [0, 0.0015] case39; a = 2.0: 0/5000 case9, 55/2500 = 0.022 [0.017, 0.029] case39.

**Where the guarantee fails (null not from the calibration generator)**, 500 (case9) / 400 (case39) runs each, false-alarm frequency with CP95:
| null | case9, repo q0 | case9, pool-95% q | case39, repo q0 | case39, pool-95% q |
|---|---|---|---|---|
| measurement noise x3 | 0.090 [0.066, 0.119] | 0.166 [0.134, 0.202] | 0.140 [0.108, 0.178] | 0.062 [0.041, 0.091] |
| plant inertia +5% (predictor nominal) | 0.066 [0.046, 0.091] | 0.120 [0.093, 0.152] | 0.223 [0.183, 0.266] | 0.098 [0.070, 0.131] |
| plant inertia +10% | 0.258 [0.220, 0.299] | 0.386 [0.343, 0.430] | 0.532 [0.482, 0.582] | 0.302 [0.258, 0.350] |
| load amplitude up to 0.16 (calib. 0.08) | 0.278 [0.239, 0.320] | 0.332 [0.291, 0.375] | 0.405 [0.357, 0.455] | 0.325 [0.279, 0.373] |
Plant-model error that is not in the calibration set breaks the guarantee badly (up to 5x to 8x the nominal rate). The gate cannot tell "model wrong" from "dispatch wrong"; it is a model-adequacy alarm.

**Run length.** Threshold calibrated on 8 s runs, tested on 16 s null runs: pool-95% q gives 0.048 [0.036, 0.063] (case9, 1000 runs), 0.047 [0.026, 0.077] (case39, 300 runs); repo q0 gives 0.020 and 0.097. No growth with horizon: the maxima come from the initial transient (random de0, w0). So the guarantee is for "one 8 s run from this start-up distribution"; it says nothing about long-run false alarms per hour (no ARL/MTFA statement is made or supported).

Other properties of the null score (pool): per-sample score excess kurtosis 1.9 (case9) and 2.0 (case39); run-maximum CV 0.48 and 0.61, skew about 0.9. Not Gaussian-noise-like.

---------------------------------------------------------------------------------------------------

## D2b. Detection delay after an unannounced dispatch step

### Model and what is predicted
Step: load and dispatch scale together by mu at t = 3 s (sample 30), predictor knows the load but keeps the nominal dispatch (repo's "unannounced mu" scenario). Relative dispatch error D = mu - 1. The first affected innovation is sample 31 (delay 1 sample = 0.1 s).
First-order relation (paper Prop. 5, `tools/detectability_check.py`): the one-step innovation is affine in the dispatch error, e = e_null + D g, with rms norm kappa = ||g||_rms. Two values: kappa0 (lowest-order closed form) = 1.0782 (case9), 2.4989 (case39); kappa_fd (finite-difference slope, my run of the repo's `one_step_state`, identical to the repo's log) = 1.0281 (case9), 2.4067 (case39), nearly independent of mu in 0.75..1.5.
Rough estimate (RSS): E[s_post]^2 ~ s0^2 + kappa^2 D^2 with s0 estimated from the pre-step scores of the same runs (case9: mean 0.00608, rms 0.00719; case39: mean 0.01163, rms 0.01427). This assumes the null innovation is orthogonal to g in the mean (no cross term). The smoothed score at post-step sample j is ((3-j) s0_mean + j s_post)/3 for j < 3, and s_post for j >= 3, giving a plug-in crossing index j*.
Bound (deterministic given assumptions): if (A1) the innovation is exactly e_null + D g, with ||g||_rms >= kappa_lb, and (A2) ||e_null||_rms <= B at every sample, then s_post >= kappa_lb |D| - B at every post-step sample, hence the smoothed score at post-step sample j is >= (j/3) (kappa_lb|D| - B)_+ for j < 3 and >= (kappa_lb|D| - B)_+ for j >= 3 (pre-step scores bounded below only by 0). The alarm (smoothed > q0) is therefore guaranteed by the first j for which that lower bound exceeds q0; in particular for all j if kappa_lb|D| > q0 + B, and by j = 1 if kappa_lb|D| > 3 q0 + B. B = 95% conformal quantile of the per-run maximum of the unsmoothed null score (n = 5000 / 2500): 0.02430 (case9), 0.04919 (case39), so (A2) holds for a whole run with probability >= 0.95 (marginal, same exchangeability argument as D2a), and the bound inherits that 0.95.

### Measured (case9 N = 300 runs/step, case39 N = 200; q0 = 0.02384 / 0.03706; scores include measurement noise)
Mean post-step score, samples j = 1..3 (CI half-width about 0.0006 case9, 0.0017 to 0.003 case39), versus RSS prediction with kappa_fd and kappa0:
| case | D | measured | RSS kappa_fd | RSS kappa0 |
|---|---|---|---|---|
| case9 | +1% | 0.0123 | 0.0125 | 0.0130 |
| case9 | +2% | 0.0220 | 0.0218 | 0.0227 |
| case9 | +3% | 0.0322 | 0.0317 | 0.0331 |
| case9 | +5% | 0.0527 | 0.0519 | 0.0544 |
| case9 | +10% | 0.1044 | 0.1031 | 0.1081 |
| case9 | +25% | 0.2600 | 0.2571 | 0.2696 |
| case9 | +50% | 0.5215 | 0.5144 | 0.5391 |
| case9 | -5% / -10% / -25% | 0.0509 / 0.1025 / 0.2573 | 0.0519 / 0.1031 / 0.2573 | 0.0544 / 0.1081 / 0.2696 |
| case39 | +1% | 0.0264 | 0.0280 | 0.0288 |
| case39 | +2% | 0.0499 | 0.0502 | 0.0520 |
| case39 | +3% | 0.0740 | 0.0736 | 0.0763 |
| case39 | +5% | 0.1226 | 0.1212 | 0.1258 |
| case39 | +10% | 0.2441 | 0.2411 | 0.2503 |
| case39 | +25% | 0.6100 | 0.6019 | 0.6249 |
| case39 | +50% | 1.2255 | 1.2038 | 1.2495 |
The score is flat in j (the first-order term does not decay), as the model says. kappa_fd-based RSS is within about 1% to 2.5% of the measurement (6% at +1% on case39); kappa0 over-predicts by 3% to 5%. Noise-free innovation / (kappa0|D|) = 0.94 to 0.98 for |D| >= 3%.

Alarm delay (first sample with smoothed score > q0; samples of 0.1 s; CI = bootstrap 95%):
| case | D | measured mean (CI) | plug-in j* | bound j | P(delay <= j*) | P(delay <= j*+1) |
|---|---|---|---|---|---|---|
| case9 | +1% | 7.6 (6.3, 9.0), only 44/300 alarm within 20 samples | none | none | - | - |
| case9 | +2% | 6.55 (6.0, 7.1), 229/300 alarm | none | none | - | - |
| case9 | +3% | 2.77 (2.66, 2.89) | 2 | none | 0.387 | 0.923 |
| case9 | +5% | 1.81 (1.77, 1.85) | 2 | 3 | 1.000 | 1.000 |
| case9 | +10%, +25%, +50% | 1.00 (1.00, 1.00) | 1 | 1 | 1.000 | 1.000 |
| case9 | -5% / -10% / -25% | 1.83 / 1.00 / 1.00 | 2 / 1 / 1 | 3 / 1 / 1 | 1.000 | 1.000 |
| case39 | +1% | 7.6 (6.8, 8.4), 106/200 alarm within 20 samples | none | none | - | - |
| case39 | +2% | 2.99 (2.79, 3.21) | 2 | none | 0.375 | 0.855 |
| case39 | +3% | 1.80 (1.75, 1.86) | 2 | none | 0.985 | 1.000 |
| case39 | +5% | 1.00 | 1 | 2 | 1.000 | 1.000 |
| case39 | +10%, +25%, +50%, -5%, -10% | 1.00 | 1 | 1 or 2 | 1.000 | 1.000 |
Bound check: in every cell where the bound gives a guarantee, the empirical P(delay <= guaranteed j) = 1.000 (CP95 lower limit 0.988 case9, 0.982 case39), and 0 of the post-step scores (j = 3..10) fell below kappa_fd|D| - B. Pre-step alarm active at sample 30: 0/300 (case9), 1/200 (case39).
Fallback delay (tau < 0.5, a = 0.25 chosen in the repo, needs q > 3.77): case9 +-10%: 3.0 samples (0.30 s; plug-in 3); +-25%: 1.5 to 1.6 (plug-in 1); +50%: 1.0; +-5%: 0/300 ever (plug-in: never). case39 +-10%: 2.0 (plug-in 2); +25%, +50%: 1.0; +5%: 72/200 fell back, mean 8.3 samples (plug-in said never); -5%: 49/200. Consistent with the paper's measured 0.1 to 0.2 s delays at 20% to 50% steps. Note the paper's "detection" is tau < 0.5, i.e. 3.77 q0, not the q0 crossing.

### Plain verdict on (b)
- The first-order score prediction is accurate (about 1% to 2.5% with kappa_fd) for 1% to 50% steps in both cases, with the model stated above. That is a validated estimate of the mean score, not a bound.
- The RSS plug-in delay is a **rough estimate**: exact for large steps, but within +-1 sample only; for steps within about a factor 1.5 of the minimal detectable step (case9 +3%, case39 +2%) it is wrong by one sample with probability 0.6 (the expected smoothed score sits at q0, so it is a coin flip). Below the threshold it predicts "never" while 15% (case9 +1%) and 53% (case39 +1%) of runs alarm within 20 samples, and for the 5% step on case39 it predicts "never fallback" while 36% fall back.
- The only **bound** is the triangle-inequality one (conditional on A1, A2; 0.95 marginal through B). It held in every cell where it gives a guarantee, but it is conservative: it guarantees detection (for all j) only when kappa_fd|D| > q0 + B, i.e. D > 4.7% on case9 ((0.0238 + 0.0243)/1.0281) and D > 3.6% on case39 ((0.0371 + 0.0492)/2.4067), whereas alarms occur already at 2% to 3%.
- Where it fails: (1) non-Gaussian null (excess kurtosis about 2, run-max CV 0.5 to 0.6): near-threshold probabilities are not predicted; (2) any model error in the null (inertia +10%: 26% to 53% false alarms) makes e_null non-orthogonal to g and inflates B; (3) the sensitivity g is the dispatch direction; steps in other parameters (line, inertia, damping) would be flagged by the same score but the mapping from score to step size is not valid; I did not test them; (4) scale: only a common load+dispatch scale step with known load was tested, not a pure dispatch step or a load-measurement error; (5) CUSUM-style delay-versus-false-alarm trade-off is not derived: the gate is a fixed-threshold smoothed rule, and no average-run-length statement is made.

---------------------------------------------------------------------------------------------------

## D3. Finite-sample certificate for the deployed closed loop (case9)

### Setup
Case9 at the study's stress loading (load scale 2.75), horizon 6 s (60 samples), controller A3(lambda): the repo's A2 MPC (physics + residual + Newton projection, analytic gradient) whose line-angle limit at prediction step k is gam - lambda * eps_k, with eps = the repo's production policy-matched envelope `results/case9_tube_envelope_stress_recal.csv` (0.82, 1.54, 2.13, 2.68, 3.26 degrees at steps 1..5). lambda = 0 is exactly A2. Grid lambda in {0, 0.5, 1, 1.5, 2, 3} (fixed before looking at data), limits 23 and 30 degrees (the limit enters the controller, so each limit is a different policy), n = 400 i.i.d. closed-loop trajectories per (limit, lambda) with the disturbance and initial-state generator of `network_cl_recalibration.cl_runs` (seeds 90000+, disjoint from the repo's 80000 to 84000 and 6001; the same 400 disturbance draws are used for all lambda: valid marginally for each p-value, and Bonferroni needs no independence). Loss = indicator that the time-fraction of samples with max line angle above the limit exceeds r (r = 0: any violation; r = 0.05, 0.10, 0.20). p-value for H_lambda: risk > alpha is the exact binomial P(Bin(n,alpha) <= k) (and Hoeffding-Bentkus min(exp(-n h1), e P(Bin <= ceil(n k/n))); both gave the same certified sets). delta = 0.10. Certification: Bonferroni (p <= delta/6) and fixed-sequence (start at lambda = 3, go down while p <= delta). Cost = mean tracking error rms(delta_rel - ref).
Note: violations are counted at controller sample times (every 0.1 s), not between samples.

### Result: the tightening does nothing useful on case9
Per-lambda summary (n = 400):
| limit | lambda | #any violation | #frac>0.05 | #frac>0.10 | #frac>0.20 | mean peak (deg) | tracking rms (rad) | paired tracking change vs A2 |
|---|---|---|---|---|---|---|---|---|
| 23 | 0 (A2) | 210 | 154 | 105 | 0 | 23.020 | 0.02680 | - |
| 23 | 0.5 | 210 | 154 | 105 | 0 | 23.020 | 0.02680 | 0.00% (400/400 trajectories identical) |
| 23 | 1 | 211 | 158 | 107 | 0 | 23.019 | 0.02706 | +0.94% |
| 23 | 1.5 | 205 | 148 | 110 | 1 | 23.014 | 0.02988 | +11.5% |
| 23 | 2 | 209 | 146 | 111 | 3 | 23.017 | 0.03241 | +20.9% |
| 23 | 3 | 212 | 154 | 120 | 4 | 23.033 | 0.03556 | +32.7% |
| 30 | 0 to 2 | 0 | 0 | 0 | 0 | 23.020 | 0.02680 | exactly 0 (all trajectories identical) |
| 30 | 3 | 0 | 0 | 0 | 0 | 23.020 | 0.02681 | +0.02% |
- At 23 degrees the A2 risk of any violation is 0.525 (CP95 [0.475, 0.575]); tightening up to lambda = 3 does not reduce it (205 to 212 of 400, paired flips 10 lost/12 gained at lambda = 3, mean peak change +0.013 +- 0.009 deg) and increases the frac>0.10 and frac>0.20 counts, while tracking error rises up to +33%. The peak line angle of 23.0 degrees is essentially the same under open-loop damping (C0 in the repo table) and A2: the control has no authority over it.
- Mechanism: A2 trajectories at limit 23 and at limit 30 are bit-identical (max |d peak| = 0 over 400 runs), so the soft angle penalty (pen = 3e3 x (excess in rad)^2, with excess about 0.004 rad) is far too small to change the control; tightening matters only once lambda eps_k is large enough to create excesses of order 0.02 to 0.1 rad (lambda >= 1.5 at 23 degrees), and then it costs tracking but does not move the peak. At 30 degrees the tightened limits (down to 20.2 degrees at step 5 for lambda = 3) leave the trajectory unchanged to 0.02%. This explains the paper's finding that A3 and A2 coincide.

### Certificates (delta = 0.10)
- **Limit 30 degrees: every lambda including A2 (lambda = 0) is certified.** 0/400 losses for all four loss definitions; CP upper bound on the risk 0.0102 at confidence 1-delta/6 (0.0057 at 1-delta for a single lambda). Bonferroni certifies alpha >= 0.02 (p = 3.1e-4 <= 0.0167); alpha = 0.01 fails Bonferroni (p = 0.018) but passes fixed-sequence (p = 0.018 <= 0.10). The smallest-cost certified lambda is lambda = 0, i.e. A2 itself, at exactly the same cost as every lambda <= 2. So the certificate is simply the one-sided binomial (rule-of-three-type) bound on the deployed controller; the margin multiplier plays no role.
- **Limit 23 degrees: nothing is certifiable for "any violation" or "time fraction > 0.05 / 0.10" at alpha <= 0.3** (no lambda; risks 0.51 to 0.53, 0.37 to 0.40, 0.26 to 0.30); even at alpha = 0.5 the any-violation loss is not certifiable (p = 0.71 to 0.89), while the looser time-fraction losses (> 0.05, > 0.10) are certified only at alpha = 0.5. For the loose loss "time fraction above 23 degrees > 0.20" the certified set is lambda <= 1.5 for alpha = 0.02 (Bonferroni) and all six lambda for alpha >= 0.05; the smallest-cost choice is lambda = 0 (A2), again not a tightened controller.
- **Is A2 certified?** At 30 degrees yes (above). At 23 degrees no, for the stated losses; A2 alone (a single hypothesis, no Bonferroni) has peak line angle quantiles over the 400 runs: median 23.02, 90% 23.33, 99% 23.82, max 23.94 degrees, so its limit would be certifiable at P(any violation) <= 0.05 / 0.10 / 0.20 (delta = 0.10) only for limits L >= 23.51 / 23.37 / 23.27 degrees. (This reading of A2 at other limits uses that A2 trajectories are limit-independent, checked bit-wise at 23 and 30 only.)
- LTT mechanics (grid, p-values, Bonferroni or fixed sequence) are as in Learn-then-Test (Angelopoulos, Bates, Candes, Jordan, Lei; cited from the task brief; I did not re-open it). The loss is a [0,1] indicator on whole closed-loop trajectories of the same deployed policy, so the guarantee is the policy-matched one: it holds for new trajectories from the same disturbance generator and the same simulator, not for another plant, other disturbances, or the true grid.

---------------------------------------------------------------------------------------------------

## Contribution to novelty (author self-rating 4/10) and what is NOT claimed

- **D2a (false-alarm guarantee): about +0.1 at most.** It is split-conformal / order-statistic exchangeability applied to a run-level max statistic. Its real value is a correctness fix: with n = 10 and a = 0.05 the repo's threshold is the sample maximum and the guarantee is 10/11 (marginal), not 95%; case39's actual q0 has fresh-run false-alarm rate 0.121. Needs n >= 19 for 95% marginal and n >= 45 (max) for a training-conditional statement. The mismatched-null experiments are a diagnostic of limits, not a result.
- **D2b (delay prediction): about +0.2.** The first-order score prediction and the triangle-inequality guaranteed-delay bound are elementary consequences of the paper's own Prop. 5 plus the D2a quantile; classical quickest-detection theory (CUSUM, Lorden; delay ~ log(MTFA)/KL) is stronger and already applied to power-system events (LIT_REPORT section D). The empirical agreement (1% to 2.5% in mean score; bound valid in all guaranteed cells) is a useful validation of Prop. 5, not new detection theory.
- **D3 (closed-loop certificate): about +0.2 to +0.3, mostly as a transparent negative result.** Same-closed-loop calibration is in Jahanshahi and Chen (2026) and Binny and Dixit (2025), LTT in Angelopoulos et al.; I only instantiate LTT on the deployed MPC with a margin grid on a DAE power case. The finding that on case9 the margin does not change the outcome (limit 23: violations 205 to 212 of 400 for every lambda; limit 30: 0 of 400) is a limitation of the tightening, and the certificate reduces to a binomial bound on A2.
- **Total: perhaps 4.0 to 4.5 out of 10; none of these moves the paper to a new tier.** They improve rigour and honesty (replace the "95%" claim, explain A2 = A3, bound the delay) rather than novelty.

### NOT claimed
- No average-run-length / mean-time-to-false-alarm guarantee; no optimality of the gate; no delay bound better than the triangle-inequality one; no claim that the RSS delay is a bound.
- No training-conditional (PAC) false-alarm guarantee for the repo's n = 10 threshold.
- No guarantee when the plant differs from the calibration generator (inertia, noise, load amplitude: measured breakdown above); the gate cannot separate dispatch error from other model error.
- No closed-loop safety or stability certificate for any physical grid; the D3 certificate is statistical, for one simulator and one disturbance law, 400 trajectories, delta = 0.10, samples at controller instants.
- No claim that the tightening eps improves violation or certifiability on case9 (it does not); no claim for case39 (D3 was run on case9 only); the pure-dispatch (load unchanged) step, and steps of other parameters, were not tested.
- LTT and the other cited works were taken from the task brief/LIT_REPORT; I did not re-verify their statements here.
