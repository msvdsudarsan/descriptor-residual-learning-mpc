"""Cell-by-cell check of Supplement S1 tables that had no generating table script, against stored result files.
Each cell is compared with the stored value with tolerance of half a unit of the last displayed digit (scientific notation: of the mantissa). Table text is read from the supplement source; no table value is stored in this script.
Usage: python tools/verify_remaining_tables.py SUPPLEMENT.tex   Exit status 1 if any compared cell differs."""
import csv, json, re, sys, numpy as np
from pathlib import Path
from scipy.stats import t as tdist
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / 'results'; EXT = ROOT / 'extensions'


def load_tables(tex):
    m = Path(tex).read_text(); base = Path(tex).parent
    for inc in re.findall(r'\\input\{([^}]+)\}', m):
        f = base / (inc + '.tex')
        if f.exists(): m = m.replace('\\input{' + inc + '}', f.read_text())
    return [t.group(0) for t in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', m, re.S)]


def rows_of(tab):
    body = re.sub(r'\\caption\{.*?\}\s*(\\label\{[^}]*\})?', '', tab, flags=re.S)
    body = body[body.find('\\midrule') + 8 if '\\midrule' in body else 0:body.find('\\bottomrule')]
    out = []
    for r in re.split(r'\\\\', body):
        r = re.sub(r'\\(midrule|cmidrule\([^)]*\)\{[^}]*\})', '', r).strip()
        if r: out.append([c.strip() for c in r.split('&')])
    return out


def nums(cell):
    c = cell.replace('$', '').replace('\\,', '').replace('{,}', '').replace('\\pm', ' ± ').replace('--', ' ')
    out = []
    for m in re.finditer(r'([+-]?\d+\.?\d*)\s*\\times\s*10\^\{?(-?\d+)\}?|(?<![\d.])10\^\{?(-?\d+)\}?|([+-]?\d+\.?\d*)', c):
        if m.group(1) is not None: dec = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0; e = int(m.group(2)); out.append((float(m.group(1)) * 10 ** e, 0.5 * 10 ** (-dec) * 10 ** e))
        elif m.group(3) is not None: e = int(m.group(3)); out.append((10.0 ** e, 0.5 * 10.0 ** e))
        else: dec = len(m.group(4).split('.')[1]) if '.' in m.group(4) else 0; out.append((float(m.group(4)), 0.5 * 10 ** (-dec)))
    return out


LOG = []


def cmp(tab, where, cell, exp):
    got = nums(cell); ok = len(got) == len(exp) and all(abs(g[0] - e) <= g[1] * 1.0000001 + 1e-12 for g, e in zip(got, exp))
    LOG.append((tab, where, ok, cell, [float(f'{e:.6g}') for e in exp])); return ok


def find(tabs, key): return next(t for t in tabs if key in t)


def main(tex):
    T = load_tables(tex)
    # --- Table 43: worked example case9 prediction (results/worked_example_case9_prediction.csv)
    t = find(T, 'tab:worked_s'); R = rows_of(t); src = list(csv.DictReader(open(RES / 'worked_example_case9_prediction.csv')))
    for r in R:
        col = {'A0': 'rmse_A0', 'A1': 'rmse_A1', 'A2': 'rmse_A2'}.get(r[0])
        if col:
            for j, tt in enumerate((0.2, 0.5, 0.8, 1.0)): cmp('43', f'{r[0]} t={tt}', r[1 + j], [float(next(x for x in src if abs(float(x['t_s']) - tt) < 1e-9)[col])])
    # --- Table 44: worked example case9 values by step (results/worked_example_case9.json)
    d = json.load(open(RES / 'worked_example_case9.json')); t = find(T, 'values by step'); R = rows_of(t)
    spec = [('dispatch', ['Pm_pu', 'Pd_total_pu']), ('rotor-to-bus', ['rotor_angle_deg']), ('lambda_{\\min}(B)', ['lam_min_B', 'lam_bound', 'lam_min_J']), ('J^{-1}', ['Jinv_norm', 'Jinv_norm_bound', 'cond_J']),
            ('weak-form target', ['wf_target', 'wf_model']), ('RMS of target', ['wf_rms_target', 'wf_rms_resid']), ('Newton', ['newton_trace']), ('state RMSE at 1', ['pred_A0_1s', 'pred_A1_1s', 'pred_A2_1s']),
            ('max|g|$ at 1', ['g_A0', 'g_A1', 'g_A2']), ('$q_0$', ['q0', 'a_gate']), ('first flag', ['gate_delay_s', 'q_first', 'tau_first', 'tau_min_after']), ('reconciled', ['rho_final', 'q_final_reconciled', 'tau_final_reconciled'])]
    for r in R:
        for key, fields in spec:
            if key in r[1] or key.replace('$', '') in r[1].replace('$', ''):
                ex = []
                for f in fields: v = d[f]; ex += list(v) if isinstance(v, list) else [v]
                if key == 'first flag': ex = [d['gate_delay_s'], d['q_first'], d['tau_first'], d['tau_min_after']]
                cmp('44', r[1][:40], r[2], ex); break
    # --- Table 51: sensitivity ratios
    t = find(T, 'tab:sens_s'); R = rows_of(t)
    for r in R:
        m = re.match(r'(case\d+), (load|line reactance|inertia|damping)', r[0])
        if m:
            p = {'load': 'load_scale', 'line reactance': 'line_scale', 'inertia': 'inertia_scale', 'damping': 'damping_scale'}[m[2]]
            rows = [x for x in csv.DictReader(open(RES / f'{m[1]}_sensitivity.csv')) if x['parameter'] == p]
            for j, sc in enumerate((0.6, 0.8, 1.0, 1.25, 1.5)): cmp('51', f'{r[0]} x{sc}', r[1 + j], [float(next(x for x in rows if abs(float(x['scale']) - sc) < 1e-9)['ratio'])])
    # --- Table 52: topology
    t = find(T, 'tab:topo_s'); R = rows_of(t)
    for r in R:
        m = re.match(r'(case\d+)$', r[0])
        if m and len(r) >= 3:
            row = next(x for x in csv.DictReader(open(RES / f'{m[1]}_topology.csv')) if int(x['line_index']) == int(r[1]))
            if 'no equilibrium' in ' '.join(r): LOG.append(('52', f'{r[0]} line {r[1]}', row['status'] == 'no_equilibrium', 'no equilibrium', [row['status']]))
            else: cmp('52', f'{r[0]} line {r[1]}', ' '.join(r[3:6]), [float(row['rmse_A0']), float(row['rmse_A2']), float(row['ratio_A2_over_A0'])])
    # --- Table 54: penalty tightening (extensions/tube_theory/results/closed_loop_tb_x{1,2,4}.csv); +- is the 95% t half-width of the paired difference in frac_over
    t = find(T, 'actuator limit'); R = rows_of(t)
    for r, x in zip(R, (1, 2, 4)):
        rows = list(csv.DictReader(open(EXT / 'tube_theory/results' / f'closed_loop_tb_x{x}.csv'))); g = lambda c: np.array([float(q['frac_over']) for q in rows if q['ctrl'] == c])
        h = lambda dd: (dd.mean(), tdist.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd)))
        cmp('54', f'x{x} A2', r[1], [g('A2').mean()]); cmp('54', f'x{x} paper', r[2], list(h(g('A3p') - g('A2')))); cmp('54', f'x{x} hybrid', r[3], list(h(g('A3h') - g('A2'))))
    # --- Table 55: learning curve (extensions/hybrid/hybrid_curve_case9.csv)
    t = find(T, 'training runs'); R = rows_of(t); src = list(csv.DictReader(open(EXT / 'hybrid/hybrid_curve_case9.csv')))
    for r in R:
        if re.match(r'\$?\d+\$?$', r[0].replace('$', '')):
            n = r[0].replace('$', '')
            ex = [float(next(x for x in src if x['ntrain'] == n and x['scenario'] == sc and x['method'] == me)['mean_ratio_to_A0']) for sc in ('in-family', 'saliency x1.5') for me in ('library', 'hybrid', 'parametric')]
            for j in range(6): cmp('55', f'n={n} col{j + 1}', r[1 + j], [ex[j]])
    # --- Table 56: learned map vs A2 (extensions/ac_baselines/results/case{9,39}_B1.json)  cells: ratio to A0 / angle rms 1s / max |g|
    t = find(T, 'learned map, linear'); R = rows_of(t); sc_of = {'in family': 'in-family (ic 0.10)', 'loading $\\times2$': None, 'line outage': None}
    B = {n: json.load(open(EXT / f'ac_baselines/results/{n}_B1.json')) for n in ('case9', 'case39')}
    keys = {n: list(B[n]) for n in B}; cur = None
    pick = lambda n, pre: next(k for k in keys[n] if k.startswith(pre))
    for r in R:
        lab = r[1] if len(r) == 4 and r[0] == '' else r[0]
        if len(r) == 4:
            if r[0]: cur = r[0].replace('$', '')
            model = r[1]
            pre = {'in family': 'in-family', 'loading \\times2': 'loading x2', 'line outage': 'line outage'}[cur if cur in ('in family', 'line outage') else 'loading \\times2']
            for j, n in enumerate(('case9', 'case39')):
                blk = B[n][pick(n, pre)]; mm = blk['A2'] if model.startswith('A2') else blk['B1 linear map']
                a0 = blk['A0']['rmse_1s']; cmp('56', f'{n} {pre} {model}', r[2 + j], [mm['rmse_1s'] / a0, mm['ang_rms_1s'], mm['g_max']])
    # --- Table 58: voltage-magnitude variant (extensions/ac_baselines/results/*_ac_{pq,z}_prediction.json, *_ac_props.log)
    t = find(T, 'constant impedance'); R = rows_of(t)
    for r in R:
        n = r[0] if r[0].startswith('case') else None
        if r[0].startswith('case'): last = r[0]
        if 'constant' in r[1]:
            tag = 'pq' if 'power' in r[1] else 'z'; j = json.load(open(EXT / f'ac_baselines/results/{last}_ac_{tag}_prediction.json'))
            ci = j['paired_ratio_ci']['A2']; cmp('58', f'{last} {tag} ratio', r[2], [j['ratio_1s_A2_over_A0']]); cmp('58', f'{last} {tag} paired', r[3], [ci[0], ci[1], ci[2]])
            cmp('58', f'{last} {tag} g', r[4], [j['rows']['A0']['gmax_max'], j['rows']['A2']['gmax_max']])
    # --- Table 71: reconciliation prediction (results/{case}_reconcile_prediction.csv)
    t = find(T, 'tab:A:rec}'); R = rows_of(t); last = None
    for r in R:
        if r[0].startswith('case'): last = r[0]
        mu = r[1].strip()
        if last and re.match(r'[\d.]+$', mu):
            row = next(x for x in csv.DictReader(open(RES / f'{last}_reconcile_prediction.csv')) if x['scenario'] == f'persistent mu {float(mu):g}')
            cmp('71', f'{last} mu {mu}', ' '.join(r[2:8]), [float(row[k]) for k in ('rmse_A0', 'rmse_A2', 'rmse_oracle', 'rmse_gate', 'rmse_reconcile', 'mean_rho_estimate')])
            cmp('71', f'{last} mu {mu} delay/tau', r[8], [float(row['detection_delay_s']), float(row['min_trust'])])
    # --- Table 1: reconciliation closed loop (results/{case}_reconcile_closed_loop.csv); 'physics + proj.' rows have no stored source
    t = find(T, 'tab:M_gate_cl'); R = rows_of(t); mu = None
    mapc = {'C0': 'C0', 'A0': 'A0', 'A2': 'A2', 'gate': 'gate', 'gate + reconc.': 'rec', 'oracle': 'oracle'}
    for r in R:
        if len(r) == 6:
            if r[0]: mu = float(r[0].replace('$', ''))
            c = r[1].strip()
            if c in mapc:
                for k, n in enumerate(('case9', 'case39')):
                    row = next(x for x in csv.DictReader(open(RES / f'{n}_reconcile_closed_loop.csv')) if abs(float(x['mu_after']) - mu) < 1e-9 and x['controller'] == mapc[c])
                    cmp('1', f'{n} mu {mu} {c}', r[2 + 2 * k], [float(row['w_rms'])]); cmp('1', f'{n} mu {mu} {c} bound', r[3 + 2 * k], [float(row['input_at_bound_share'])])
            else: LOG.append(('1', f'mu {mu} {c}', None, ' '.join(r[2:]), ['no stored source (UNVERIFIED rows)']))
    # --- Table 53: transfer matrix (extensions/transfer/check_stochastic.log); rotor-to-bus column has no stored source
    t = find(T, 'transfer matrix in the lossless'); R = rows_of(t); log = (EXT / 'transfer/check_stochastic.log').read_text().splitlines()
    P = {n: dict(zip(('rs', 'me', 'inf', 'gain'), map(float, re.findall(r'(?:rowsum-1\||min entry|inf-norm|row diff\)) ([0-9.e+-]+)', l)))) for n, l in zip(('case9', 'case39'), log[:2])}
    cmp('53', 'rowsum', R[0][1] + ' ' + R[0][2], [P['case9']['rs'], P['case39']['rs']]); cmp('53', 'min entry', R[1][1] + ' ' + R[1][2], [P['case9']['me'], P['case39']['me']]); cmp('53', 'inf norm', R[2][1] + ' ' + R[2][2], [P['case9']['inf'], P['case39']['inf']])
    LOG.append(('53', 'half l1 line angles', all(abs(a - b / 2) <= 0.005 for a, b in zip((float(x) for x in re.findall(r'0\.\d+', R[3][1])[:1] + re.findall(r'0\.\d+', R[3][2])[:1]), (P['case9']['gain'], P['case39']['gain']))), R[3][1] + ' | ' + R[3][2], [P['case9']['gain'] / 2, P['case39']['gain'] / 2]))
    LOG.append(('53', 'half l1 rotor-to-bus angles', None, R[3][1] + ' | ' + R[3][2], ['no stored source for the rotor-to-bus values 0.32 / 0.58']))
    return LOG


if __name__ == '__main__':
    L = main(sys.argv[1]); bad = 0; import collections; agg = collections.defaultdict(lambda: [0, 0, 0])
    for tab, where, ok, cell, ex in L:
        agg[tab][0 if ok else (1 if ok is False else 2)] += 1
        if ok is not True: print(f'Table {tab} [{where}] {"MISMATCH" if ok is False else "NO SOURCE"}: cell {cell!r} expected {ex}'); bad += ok is False
    print('\ntable: pass / mismatch / no-source cells'); [print(f'  {k}: {v[0]} / {v[1]} / {v[2]}') for k, v in sorted(agg.items(), key=lambda kv: int(kv[0]))]
    sys.exit(1 if bad else 0)
