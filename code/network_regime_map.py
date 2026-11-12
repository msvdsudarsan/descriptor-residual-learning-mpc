"""Regime map (PROTOCOL_REGIME_MAP). NEW script; no existing code edited.
Usage: python3 network_regime_map.py CASE V      validation-only checks V1-V3 (seeds 1001/1002 and pilot 80000 only; no test seed is drawn)
       python3 network_regime_map.py CASE PRED   prediction cells (needs V PASS)
       python3 network_regime_map.py CASE CL     closed-loop cells (needs V PASS)
Regimes: M0 baseline plant; M1 saliency kappa0 x2.5; M2 plant load lag tau=3 s (prediction only). Authority: x1, x3 (controller bound only; the plant has no clipping of its own)."""
import sys, json, csv as _csv, time
import numpy as np
from network_recal_baseline import recal_physics, val_rmse_linear, pred_run, csvw, energy
from network_param_baseline import VARIANTS, restricted
from network_cl_recalibration import metrics
from network_baselines_loaddyn import stack
from network_validity_gate import *

OUT = RES / 'regime_map'; OUT.mkdir(exist_ok=True)
KAPPA_M1 = 2.5; TAU_M2 = 3.0; AUTH = (1, 3); SEED_TEST = 9400; SEED_CL = 6001; SEED_PILOT = 80000
NRUN_PRED = {'case9': 8, 'case39': 5}; NRUN_CL = {'case9': 6, 'case39': 4}


def regime_plant(c0, kind):
    c = dict(c0)
    if kind == 'M1': c['kappa'] = c0['kappa'].copy(); c['kappa'][0] *= KAPPA_M1
    return c


def lag_load(load, tau):
    a = load[1] * tau; return (load[0] / np.sqrt(1 + a * a), load[1], load[2] - np.arctan(a))


def make_training_reg(cp, c0, th_eq, de_eq, n_traj, seed, noise, kind, h=0.025, T=3.0, ic=0.20, wic=0.8, lamp=0.04):
    """Same as make_training (code/network_experiments.py); the plant is cp, the recorded (told) load is the nominal law; for M2 the plant load is lagged."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_traj):
        de0 = de_eq + rng.normal(0, ic, c0['ng']); w0 = rng.normal(0, wic, c0['ng'])
        th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
        load = (rng.uniform(0, lamp), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28))
        def ctrl(t, de, w, th, up, Pd): return np.clip(-cp['D'] * w + rng.normal(0, 0.35, cp['ng']) * cp['umax'], -cp['umax'], cp['umax'])
        lp = lag_load(load, TAU_M2) if kind == 'M2' else load
        lg = simulate_plant(cp, th0, de0, w0, ctrl, T, lp, hp=h / 2, Ts=h)
        de_m = lg['de'] + rng.normal(0, noise, lg['de'].shape); w_m = lg['w'] + rng.normal(0, noise, lg['w'].shape)
        Pd_series = np.array([c0['Pd'] * (1 + load[0] * np.sin(load[1] * tt + load[2])) for tt in lg['t']])
        out.append(dict(de=de_m, w=w_m, U=lg['u'][:-1], Pd=Pd_series, h=h, de_true=lg['de'], w_true=lg['w'], th_true=lg['th']))
    return out


def test_runs(cp, c0, th_eq, de_eq, n, seed, kind, T=8.0):
    """Same draws and call order as scenario_runs / make_test_runs; plant cp; lag for M2; the predictor is told the nominal load."""
    rng = np.random.default_rng(seed); runs = []
    for _ in range(n):
        de0 = de_eq + rng.normal(0, 0.10, c0['ng']); w0 = rng.normal(0, 0.3, c0['ng'])
        th0, _ = solve_theta(th_eq, de0, c0['Pd'], *args_of(c0), 1e-12, 30)
        load = (rng.uniform(0.0, 0.06), rng.uniform(1.0, 6.0), rng.uniform(0, 6.28)); lp = lag_load(load, TAU_M2) if kind == 'M2' else load
        runs.append((load, simulate_plant(cp, th0, de0, w0, static_ctrl(c0), T, lp)))
    return runs


def identify(c0, th_eq, train, val):
    TH, Y = stack([weak_form_net(c0, t, th_eq) for t in train]); THv, Yv = stack([weak_form_net(c0, t, th_eq) for t in val])
    coef2 = fit_net(c0, train, val, th_eq)[0]
    cf = {k: restricted(TH, Y, THv, Yv, c0['ng'], v) for k, v in VARIANTS.items() if v is not None}
    c1c, info, v1c = recal_physics(c0, TH, Y, THv, Yv)
    coef3 = fit_net(c1c, train, val, th_eq)[0]
    vs = {'P1a': val_rmse_linear(THv, Yv, cf['damping + constant'], c0['ng']), 'P1b': val_rmse_linear(THv, Yv, cf['damping + cubic speed + constant'], c0['ng']), 'P1c': v1c}
    pick = min(vs, key=vs.get)
    return dict(coef2=coef2, cf=cf, c1c=c1c, info=info, coef3=coef3, val=vs, pick=pick, TH=TH, Y=Y, THv=THv, Yv=Yv)


def windows_check(c0, tr_list, th_eq):
    """V1: Newton projection (tol 1e-11, 30 it) in every window of the weak-form construction; returns (n windows, n failing, n non-finite)."""
    nw = nf = nn = 0
    for t in tr_list:
        guess = th_eq
        for k in range(t['U'].shape[0] + 1):
            guess, nit = solve_theta(guess, t['de'][k], t['Pd'][k], *args_of(c0), 1e-11, 30)
            g, _ = alg_res_jac(guess, t['de'][k], t['Pd'][k], *args_of(c0)[:3], c0['gbus'], c0['bg'], False); nw += 1
            if not np.all(np.isfinite(guess)): nn += 1
            elif np.max(np.abs(g)) > 1e-9: nf += 1
    return nw, nf, nn


def finite_traj(tr_list): return all(np.all(np.isfinite(t[k])) for t in tr_list for k in ('de', 'w', 'U', 'Pd', 'de_true', 'w_true', 'th_true'))


def gen_pilot(cs, th_s, de_s):
    rng = np.random.default_rng(SEED_PILOT); pil = []
    for _ in range(10):
        load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_s + rng.normal(0, 0.03, cs['ng']); w0 = rng.normal(0, 0.1, cs['ng'])
        th0, _ = solve_theta(th_s, de0, cs['Pd'], *args_of(cs), 1e-12, 30); pil.append((load, de0, w0, th0))
    return pil


def sat_frac(lg, umax_cell): u = lg['u'][:-1]; return float(np.mean(np.abs(u) >= 0.999 * umax_cell[None, :]))


def setup_cl(name, kind, auth):
    """Closed-loop cell: nominal stress case cs0 (model), regime plant cs, authority-scaled copies for controllers; identification at x1 data (authority is a controller property)."""
    cs0 = build_case(name, load_scale=STRESS[name]); th_s, de_s, _ = equilibrium(cs0); cs = regime_plant(cs0, kind)
    trs = make_training_reg(cs, cs0, th_s, de_s, 30, 1001, 1e-3, kind); vls = make_training_reg(cs, cs0, th_s, de_s, 8, 1002, 1e-3, kind)
    return cs0, cs, th_s, de_s, trs, vls


def with_auth(c, auth): d = dict(c); d['umax'] = c['umax'] * auth; return d


def validation(name):
    """Validation-only checks. Returns dict cell -> result; writes results/regime_map/{name}_validation.json."""
    res = {}; c0 = build_case(name); th_eq, de_eq, _ = equilibrium(c0); base_tr = make_training(c0, th_eq, de_eq, 30, 1001, 1e-3); base_vl = make_training(c0, th_eq, de_eq, 8, 1002, 1e-3)
    ib = identify(c0, th_eq, base_tr, base_vl)
    for kind in ('M0', 'M1', 'M2'):
        cp = regime_plant(c0, kind); tr = make_training_reg(cp, c0, th_eq, de_eq, 30, 1001, 1e-3, kind); vl = make_training_reg(cp, c0, th_eq, de_eq, 8, 1002, 1e-3, kind)
        fin = finite_traj(tr + vl); nw, nf, nn = windows_check(c0, tr + vl, th_eq); idn = identify(c0, th_eq, tr, vl)
        fits = all(np.all(np.isfinite(x)) for x in (idn['coef2'], idn['coef3'], idn['cf']['damping + constant'], idn['cf']['damping + cubic speed + constant'], idn['c1c']['M'], idn['c1c']['Dphys'], idn['c1c']['Pm'])) and all(np.isfinite(v) for v in idn['val'].values())
        # V3: frozen baseline coefficients on the regime validation trajectories
        THv, Yv = stack([weak_form_net(c0, t, th_eq) for t in vl]); v3 = bool(np.all(np.isfinite(Yv - np.einsum('kgf,gf->kg', THv, ib['coef2'][:, :NF])))) and bool(np.all(np.isfinite(val_rmse_linear(THv, Yv, ib['cf']['damping + constant'], c0['ng']))))
        res[f'V1/V3 {kind}'] = dict(trajectories=len(tr) + len(vl), finite=fin, windows=nw, newton_fail=nf, nonfinite=nn, fits_finite=fits, deploy_shift_finite=v3, PASS=bool(fin and nf == 0 and nn == 0 and fits and v3), P1_pick=idn['pick'], val=idn['val'])
        if kind == 'M0':   # consistency: re-identified P2 equals the stored baseline coefficients
            res['V1/V3 M0']['P2_equals_stored_coef_nominal'] = bool(np.allclose(idn['coef2'], np.load(RES / f'coef_nominal_{name}.npy'), rtol=0, atol=1e-12))
    # H1: initial states and loads of the M0/M1/M2 test realizations are identical (test seed generator invoked WITHOUT running a test: only the random draws are compared)
    rngs = [np.random.default_rng(SEED_TEST) for _ in range(3)]; d = [[(r.normal(0, 0.10, c0['ng']), r.normal(0, 0.3, c0['ng']), r.uniform(0, 0.06), r.uniform(1, 6), r.uniform(0, 6.28)) for _ in range(NRUN_PRED[name])] for r in rngs]
    res['H1 matched draws M0=M1=M2 (seed %d)' % SEED_TEST] = dict(PASS=bool(all(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(x, y)) for x, y in zip(d[0], d[1])) and all(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(x, y)) for x, y in zip(d[0], d[2]))))
    res['H1 equilibrium independent of kappa'] = dict(PASS=bool(np.array_equal(equilibrium(regime_plant(c0, 'M1'))[1], de_eq)))
    # V2: authority x3 (controller bound), plant passes controller output unchanged; PILOT seed 80000 only
    for kind in ('M0', 'M1'):
        for auth in AUTH:
            cs0, cs, th_s, de_s, trs, vls = setup_cl(name, kind, auth); pil = gen_pilot(cs0, th_s, de_s); cauth = with_auth(cs0, auth)
            assert np.array_equal(cauth['umax'], auth * cs0['umax'])
            coef = fit_net(cs0, trs, vls, th_s)[0]; rel_ref = de_s - np.sum(cs0['M'] * de_s) / np.sum(cs0['M']); mr = {}; fin = True; eq_ok = True
            for cn, mk in (('P2', lambda: make_net_mpc(cauth, 'A2', coef, rel_ref, np.zeros(6), analytic=True)), ('C0', lambda: static_ctrl(cauth)), ('A0', lambda: make_net_mpc(cauth, 'A0', coef, rel_ref, np.zeros(6), analytic=False))):
                mx = 0.0
                for load, de0, w0, th0 in pil[:3]:
                    outs = []; f = mk(); ctrl = (lambda f: (lambda t, de, w, th, up, Pd: (outs.append(f(t, de, w, th, up, Pd).copy()), outs[-1])[1]))(f)
                    lg = simulate_plant(cs, th0, de0, w0, ctrl, 6.0 if name == 'case9' else 5.0, load); fin &= bool(np.all(np.isfinite(lg['w']))); u = lg['u'][:-1]
                    eq_ok &= bool(np.array_equal(np.array(outs), u)); mx = max(mx, float(np.max(np.abs(u) / cs0['umax'][None, :])))
                mr[cn] = mx
            # unit test: input of 2.5 x nominal bound reaches the dynamics unchanged
            u25 = 2.5 * cs0['umax']; lg = simulate_plant(cs, pil[0][3], pil[0][1], pil[0][2], lambda t, de, w, th, up, Pd: u25, 0.1, pil[0][0]); ut = bool(np.array_equal(lg['u'][0], u25))
            de1, w1, th1 = pil[0][1].copy(), pil[0][2].copy(), pil[0][3].copy(); t_ = 0.0
            for _ in range(8): de1, w1, th1 = plant_step(de1, w1, th1, u25, 0.0125, t_, cs['Pd'], *pil[0][0], cs['lf'], cs['lt'], cs['lb'], cs['gbus'], cs['bg'], cs['M'], cs['D'], cs['Pm'], cs['kappa'], cs['fric']); t_ += 0.0125
            ut &= bool(np.allclose(lg['w'][1], w1, rtol=0, atol=1e-12))
            lim = 1 + 1e-9 if auth == 1 else 3 + 1e-9
            res[f'V2 {kind} authority x{auth}'] = dict(PASS=bool(fin and eq_ok and ut and max(mr.values()) <= lim), finite=fin, applied_equals_controller_output=eq_ok, unit_test_2p5x_passes_unchanged=ut, max_ratio_to_x1_bound=mr, umax_is_auth_times_nominal=True,
                                                          nominal_bound_exceeded_on_pilot=bool(max(mr.values()) > 1 + 1e-9))
    json.dump(res, open(OUT / f'{name}_validation.json', 'w'), indent=1, default=str)
    return res


def prediction(name):
    val = json.load(open(OUT / f'{name}_validation.json')); c0 = build_case(name); th_eq, de_eq, _ = equilibrium(c0)
    base_tr = make_training(c0, th_eq, de_eq, 30, 1001, 1e-3); base_vl = make_training(c0, th_eq, de_eq, 8, 1002, 1e-3); ib = identify(c0, th_eq, base_tr, base_vl)
    rows, sel, fails, ne = [], {}, [], {}
    for kind in ('M0', 'M1', 'M2'):
        if not val[f'V1/V3 {kind}']['PASS']: ne[kind] = 'NOT EVALUABLE (validation)'; continue
        cp = regime_plant(c0, kind); runs = test_runs(cp, c0, th_eq, de_eq, NRUN_PRED[name], SEED_TEST, kind)
        for setting in ('a', 'b'):
            if kind == 'M0' and setting == 'b': continue            # identical to (a) for the baseline plant
            if setting == 'a':
                tr = make_training_reg(cp, c0, th_eq, de_eq, 30, 1001, 1e-3, kind); vl = make_training_reg(cp, c0, th_eq, de_eq, 8, 1002, 1e-3, kind); idn = identify(c0, th_eq, tr, vl)
            else: idn = ib
            c1c = idn['c1c']; z = np.zeros((c0['ng'], NF + 1))
            models = [('A0', c0, 'A0', idn['coef2']), ('P2', c0, 'A2', idn['coef2']), ('P1a', c0, 'A2', idn['cf']['damping + constant']), ('P1b', c0, 'A2', idn['cf']['damping + cubic speed + constant']), ('P1c', c1c, 'A2', z), ('P3', c1c, 'A2', idn['coef3'])]
            sel[f'{kind}{setting}'] = dict(pick=idn['pick'], val=idn['val'])
            for ri, run in enumerate(runs):
                for mn, cp_, mode, co in models:
                    try: ss, n = pred_run(cp_, c0['M'], run, mode, co)
                    except Exception as ex: ss, n = float('nan'), 0; fails.append((kind, setting, ri, mn, repr(ex)))
                    rows.append((kind, setting, ri, mn, ss, n))
            print(name, kind, setting, 'pick', idn['pick'], flush=True)
    csvw(OUT / f'{name}_prediction_runs.csv', ['regime', 'setting', 'run', 'model', 'sumsq', 'n'], rows)
    json.dump(dict(selection=sel, not_evaluable=ne, failures=fails), open(OUT / f'{name}_prediction_meta.json', 'w'), indent=1)


def closed_loop(name):
    val = json.load(open(OUT / f'{name}_validation.json')); rows, meta, fails = [], {}, []
    T = 6.0 if name == 'case9' else 5.0
    for kind in ('M0', 'M1'):
        for auth in AUTH:
            cell = f'{kind}_x{auth}'; v = val[f'V2 {kind} authority x{auth}']
            if not v['PASS']: meta[cell] = 'NOT EVALUABLE (V2)'; continue
            cs0, cs, th_s, de_s, trs, vls = setup_cl(name, kind, auth); idn = identify(cs0, th_s, trs, vls); cauth = with_auth(cs0, auth); c1c_a = with_auth(idn['c1c'], auth)
            rel_ref = de_s - np.sum(cs0['M'] * de_s) / np.sum(cs0['M']); z = np.zeros((cs0['ng'], NF + 1)); umax_cell = cauth['umax']
            P1c = idn['pick'] == 'P1c'
            def ctrl_of(k, mult=None):
                if k == 'C0': return static_ctrl(cauth)
                if k == 'C0eq': kg = mult * cs0['D']; return lambda t, de, w, th, up, Pd: np.clip(-kg * w, -umax_cell, umax_cell)
                if k == 'A0': return make_net_mpc(cauth, 'A0', idn['coef2'], rel_ref, np.zeros(6), analytic=False)
                if k == 'P2': return make_net_mpc(cauth, 'A2', idn['coef2'], rel_ref, np.zeros(6), analytic=True)
                if k == 'P1*': return make_net_mpc(c1c_a, 'A2', z, rel_ref, np.zeros(6), analytic=True) if P1c else make_net_mpc(cauth, 'A2', idn['cf']['damping + constant' if idn['pick'] == 'P1a' else 'damping + cubic speed + constant'], rel_ref, np.zeros(6), analytic=True)
                if k == 'P3': return make_net_mpc(c1c_a, 'A2', idn['coef3'], rel_ref, np.zeros(6), analytic=True)
            pil = gen_pilot(cs0, th_s, de_s)
            run_p = lambda fn: [energy(simulate_plant(cs, th0, de0, w0, fn(), T, load)) for load, de0, w0, th0 in pil]
            e_t = float(np.mean(run_p(lambda: ctrl_of('P2')))); lo_, hi_ = np.log(0.05), np.log(60.0); hist = []
            for _ in range(16):
                mid = 0.5 * (lo_ + hi_); e = float(np.mean(run_p(lambda: ctrl_of('C0eq', np.exp(mid))))); hist.append((float(np.exp(mid)), e))
                if e < e_t: lo_ = mid
                else: hi_ = mid
            g = float(np.exp(0.5 * (lo_ + hi_))); e_ach = float(np.mean(run_p(lambda: ctrl_of('C0eq', g))))
            meta[cell] = dict(pick=idn['pick'], val=idn['val'], pilot_energy_P2=e_t, pilot_energy_C0eq=e_ach, gain_multiple_of_D=g, bracket_reached=bool(0.05 * 1.01 < g < 60 / 1.01), hist=hist)
            rng = np.random.default_rng(SEED_CL)
            for i in range(NRUN_CL[name]):
                load = (rng.uniform(0.07, 0.11), rng.uniform(2.5, 4.0), rng.uniform(0, 6.28)); de0 = de_s + rng.normal(0, 0.03, cs0['ng']); w0 = rng.normal(0, 0.1, cs0['ng'])
                th0, _ = solve_theta(th_s, de0, cs['Pd'], *args_of(cs), 1e-12, 30)
                for k in ('C0', 'A0', 'P2', 'P1*', 'P3', 'C0eq'):
                    try:
                        outs = []; f = ctrl_of(k, g); ctrl = (lambda f: (lambda t, de, w, th, up, Pd: (outs.append(f(t, de, w, th, up, Pd).copy()), outs[-1])[1]))(f)
                        lg = simulate_plant(cs, th0, de0, w0, ctrl, T, load); m = metrics(cs0, lg, GAMMA)
                        if not np.all(np.isfinite(lg['w'])): raise FloatingPointError('non-finite speed')
                        assert np.array_equal(np.array(outs), lg['u'][:-1]), 'applied input differs from controller output'
                        rows.append((kind, auth, i, k, m['w_rms'], m['peak'], m['viol'], m['frac'], energy(lg), sat_frac(lg, umax_cell), m['solve_ms'], float(np.max(np.abs(lg['u'][:-1]) / cs0['umax'][None, :]))))
                    except Exception as ex:
                        rows.append((kind, auth, i, k) + ('nan',) * 8); fails.append((kind, auth, i, k, repr(ex)))
                print(name, cell, 'run', i, flush=True)
    csvw(OUT / f'{name}_closed_loop_runs.csv', ['regime', 'authority', 'run', 'controller', 'w_rms', 'peak_deg', 'viol_integral', 'frac_over_gamma', 'input_energy', 'saturation_fraction', 'solve_ms', 'max_u_over_x1_bound'], rows)
    json.dump(dict(cells=meta, failures=fails), open(OUT / f'{name}_closed_loop_meta.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    nm, stage = sys.argv[1], sys.argv[2]
    if stage == 'V':
        r = validation(nm)
        for k, v in r.items(): print(k, 'PASS' if v['PASS'] else 'FAIL', {a: b for a, b in v.items() if a != 'PASS'}, flush=True)
    elif stage == 'PRED': prediction(nm)
    elif stage == 'CL': closed_loop(nm)
