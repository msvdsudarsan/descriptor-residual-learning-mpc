
## case9 prediction (selected P1* by validation = P1c)
- in-family: pooled ratio to A0: P2 library residual 0.1926, P1a 0.2010, P1b 0.1977, P1c 0.2053, P3 0.1927
    P2 library residual - P1c: mean d -0.0117, CI [-0.0160, -0.0075] (n=8, half-width 0.0043, margin 0.02) -> EQUIVALENT (CI inside margin)
    P2 library residual - P1c: mean d -0.0117, CI [-0.0160, -0.0075] (n=8, half-width 0.0043, margin 0.02) -> EQUIVALENT (CI inside margin)
    P2 library residual - P1a: mean d -0.0080, CI [-0.0101, -0.0059] (n=8, half-width 0.0021, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P1c: mean d -0.0117, CI [-0.0159, -0.0074] (n=8, half-width 0.0042, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P2 library residual: mean d +0.0001, CI [+0.0000, +0.0001] (n=8, half-width 0.0000, margin 0.02) -> EQUIVALENT (CI inside margin)
- saliency x1.5: pooled ratio to A0: P2 library residual 0.3388, P1a 0.3594, P1b 0.3564, P1c 0.3666, P3 0.3392
    P2 library residual - P1c: mean d -0.0278, CI [-0.0333, -0.0224] (n=8, half-width 0.0055, margin 0.02) -> P2 better (CI below -margin)
    P2 library residual - P1c: mean d -0.0278, CI [-0.0333, -0.0224] (n=8, half-width 0.0055, margin 0.02) -> P2 better (CI below -margin)
    P2 library residual - P1a: mean d -0.0206, CI [-0.0212, -0.0200] (n=8, half-width 0.0006, margin 0.02) -> P2 better (CI below -margin)
    P3 - P1c: mean d -0.0275, CI [-0.0330, -0.0220] (n=8, half-width 0.0055, margin 0.02) -> P2 better (CI below -margin)
    P3 - P2 library residual: mean d +0.0003, CI [+0.0003, +0.0003] (n=8, half-width 0.0000, margin 0.02) -> EQUIVALENT (CI inside margin)
- load lag 1.0 s: pooled ratio to A0: P2 library residual 0.1096, P1a 0.1136, P1b 0.1105, P1c 0.0554, P3 0.1097
    P2 library residual - P1c: mean d +0.0410, CI [+0.0025, +0.0795] (n=8, half-width 0.0385, margin 0.02) -> INCONCLUSIVE
    P2 library residual - P1c: mean d +0.0410, CI [+0.0025, +0.0795] (n=8, half-width 0.0385, margin 0.02) -> INCONCLUSIVE
    P2 library residual - P1a: mean d -0.0045, CI [-0.0066, -0.0023] (n=8, half-width 0.0021, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P1c: mean d +0.0410, CI [+0.0025, +0.0796] (n=8, half-width 0.0385, margin 0.02) -> INCONCLUSIVE
    P3 - P2 library residual: mean d +0.0000, CI [-0.0000, +0.0001] (n=8, half-width 0.0000, margin 0.02) -> EQUIVALENT (CI inside margin)

## case9 closed loop (selected P1* by validation = P1c; n=6 runs)
- mean w_rms: C0 static law 1.0263, A0 physics only 1.0270, library residual (paper) 0.8758, damping + constant 0.8752, damping + cubic speed + constant 0.8744, P1c physical recalibration 0.8769, P3 residual on P1c 0.8755, C0 energy-matched 0.8911
- mean input energy: C0 static law 0.756, A0 physics only 0.893, library residual (paper) 1.179, damping + constant 1.186, damping + cubic speed + constant 1.193, P1c physical recalibration 1.172, P3 residual on P1c 1.181, C0 energy-matched 1.177
    library residual (paper) - P1c physical recalibration: per-seed d [-0.0009, -0.001, -0.0012, -0.0013, -0.0011, -0.0013], mean -0.0011, CI [-0.0013, -0.0009] (half-width 0.0002, margin 0.0175) -> EQUIVALENT (CI inside margin); energy ratio 1.007
    library residual (paper) - P1c physical recalibration: per-seed d [-0.0009, -0.001, -0.0012, -0.0013, -0.0011, -0.0013], mean -0.0011, CI [-0.0013, -0.0009] (half-width 0.0002, margin 0.0175) -> EQUIVALENT (CI inside margin); energy ratio 1.007
    P3 residual on P1c - P1c physical recalibration: per-seed d [-0.0011, -0.0012, -0.0016, -0.0016, -0.0013, -0.0016], mean -0.0014, CI [-0.0016, -0.0012] (half-width 0.0002, margin 0.0175) -> EQUIVALENT (CI inside margin); energy ratio 1.008
    P3 residual on P1c - library residual (paper): per-seed d [-0.0002, -0.0002, -0.0003, -0.0003, -0.0002, -0.0003], mean -0.0003, CI [-0.0003, -0.0002] (half-width 0.0001, margin 0.0175) -> EQUIVALENT (CI inside margin); energy ratio 1.002
    library residual (paper) - C0 energy-matched: per-seed d [-0.0133, -0.0133, -0.0163, -0.0146, -0.0132, -0.0213], mean -0.0153, CI [-0.0186, -0.0120] (half-width 0.0033, margin 0.0178) -> INCONCLUSIVE; energy ratio 1.002
    library residual (paper) - C0 static law: per-seed d [-0.1309, -0.1271, -0.1565, -0.1559, -0.1631, -0.1698], mean -0.1505, CI [-0.1689, -0.1322] (half-width 0.0184, margin 0.0205) -> P2 better (CI below -margin); energy ratio 1.561

## case39 prediction (selected P1* by validation = P1c)
- in-family: pooled ratio to A0: P2 library residual 0.1477, P1a 0.1502, P1b 0.1494, P1c 0.1538, P3 0.1480
    P2 library residual - P1c: mean d -0.0042, CI [-0.0121, +0.0037] (n=5, half-width 0.0079, margin 0.02) -> EQUIVALENT (CI inside margin)
    P2 library residual - P1c: mean d -0.0042, CI [-0.0121, +0.0037] (n=5, half-width 0.0079, margin 0.02) -> EQUIVALENT (CI inside margin)
    P2 library residual - P1a: mean d -0.0025, CI [-0.0039, -0.0011] (n=5, half-width 0.0014, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P1c: mean d -0.0040, CI [-0.0117, +0.0038] (n=5, half-width 0.0077, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P2 library residual: mean d +0.0002, CI [-0.0000, +0.0005] (n=5, half-width 0.0002, margin 0.02) -> EQUIVALENT (CI inside margin)
- saliency x1.5: pooled ratio to A0: P2 library residual 0.3585, P1a 0.3681, P1b 0.3675, P1c 0.3832, P3 0.3596
    P2 library residual - P1c: mean d -0.0248, CI [-0.0306, -0.0190] (n=5, half-width 0.0058, margin 0.02) -> INCONCLUSIVE
    P2 library residual - P1c: mean d -0.0248, CI [-0.0306, -0.0190] (n=5, half-width 0.0058, margin 0.02) -> INCONCLUSIVE
    P2 library residual - P1a: mean d -0.0097, CI [-0.0115, -0.0079] (n=5, half-width 0.0018, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P1c: mean d -0.0237, CI [-0.0295, -0.0179] (n=5, half-width 0.0058, margin 0.02) -> INCONCLUSIVE
    P3 - P2 library residual: mean d +0.0011, CI [+0.0010, +0.0011] (n=5, half-width 0.0001, margin 0.02) -> EQUIVALENT (CI inside margin)
- load lag 1.0 s: pooled ratio to A0: P2 library residual 0.1970, P1a 0.1948, P1b 0.1933, P1c 0.0778, P3 0.1973
    P2 library residual - P1c: mean d +0.1192, CI [+0.0873, +0.1511] (n=5, half-width 0.0319, margin 0.02) -> P1 better (CI above +margin)
    P2 library residual - P1c: mean d +0.1192, CI [+0.0873, +0.1511] (n=5, half-width 0.0319, margin 0.02) -> P1 better (CI above +margin)
    P2 library residual - P1a: mean d +0.0022, CI [+0.0007, +0.0037] (n=5, half-width 0.0015, margin 0.02) -> EQUIVALENT (CI inside margin)
    P3 - P1c: mean d +0.1196, CI [+0.0876, +0.1515] (n=5, half-width 0.0319, margin 0.02) -> P1 better (CI above +margin)
    P3 - P2 library residual: mean d +0.0004, CI [+0.0003, +0.0004] (n=5, half-width 0.0001, margin 0.02) -> EQUIVALENT (CI inside margin)

## case39 closed loop (selected P1* by validation = P1c; n=4 runs)
- mean w_rms: C0 static law 1.9601, A0 physics only 2.0669, library residual (paper) 1.8784, damping + constant 1.8779, damping + cubic speed + constant 1.8775, P1c physical recalibration 1.8786, P3 residual on P1c 1.8784, C0 energy-matched 1.8870
- mean input energy: C0 static law 19.591, A0 physics only 24.732, library residual (paper) 26.001, damping + constant 26.081, damping + cubic speed + constant 26.138, P1c physical recalibration 26.046, P3 residual on P1c 25.992, C0 energy-matched 27.648
    library residual (paper) - P1c physical recalibration: per-seed d [-0.0003, 0.0, -0.0005, -0.0001], mean -0.0002, CI [-0.0006, +0.0001] (half-width 0.0003, margin 0.0376) -> EQUIVALENT (CI inside margin); energy ratio 0.998
    library residual (paper) - P1c physical recalibration: per-seed d [-0.0003, 0.0, -0.0005, -0.0001], mean -0.0002, CI [-0.0006, +0.0001] (half-width 0.0003, margin 0.0376) -> EQUIVALENT (CI inside margin); energy ratio 0.998
    P3 residual on P1c - P1c physical recalibration: per-seed d [-0.0003, 0.0, -0.0004, -0.0001], mean -0.0002, CI [-0.0005, +0.0001] (half-width 0.0003, margin 0.0376) -> EQUIVALENT (CI inside margin); energy ratio 0.998
    P3 residual on P1c - library residual (paper): per-seed d [0.0, 0.0, 0.0001, 0.0], mean +0.0000, CI [-0.0000, +0.0001] (half-width 0.0000, margin 0.0376) -> EQUIVALENT (CI inside margin); energy ratio 1.000
    library residual (paper) - C0 energy-matched: per-seed d [-0.0141, -0.0002, -0.0099, -0.0104], mean -0.0086, CI [-0.0181, +0.0008] (half-width 0.0094, margin 0.0377) -> EQUIVALENT (CI inside margin); energy ratio 0.940
    library residual (paper) - C0 static law: per-seed d [-0.0797, -0.0925, -0.078, -0.0765], mean -0.0817, CI [-0.0934, -0.0700] (half-width 0.0117, margin 0.0392) -> P2 better (CI below -margin); energy ratio 1.327