"""Regenerate the numeric columns of selected Supplement S1 tables from the stored result files and, optionally, compare them cell by cell with the supplement.

Usage (from the repository root):
  python tools/trace_supplement_tables.py --out DIR                       # write the regenerated tables (csv) only
  python tools/trace_supplement_tables.py --out DIR --supplement S1.tex   # also compare with the supplement source (kept outside this repository)

The mapping (stored file, column, scale, number of decimals) is read from tools/supplement_table_specs.json and contains no table values.
Only numeric cells are compared; row labels are not. Exit status 1 if any compared cell differs.
"""
import argparse, csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_source(path):
    p = ROOT / path
    if path.endswith('.csv'):
        return list(csv.DictReader(open(p)))
    raise SystemExit('unsupported source ' + path)

def fmt(x, d):
    if isinstance(x, tuple):
        return '[' + ', '.join(f'{v:.{d}f}' for v in x) + ']'
    return f'{x:.{d}f}'

def parse_supplement(tex_path):
    m = Path(tex_path).read_text()
    base = Path(tex_path).parent
    for inc in re.findall(r'\\input\{([^}]+)\}', m):
        f = base / (inc + '.tex')
        if f.exists():
            m = m.replace('\\input{' + inc + '}', f.read_text())
    out = []
    for mt in re.finditer(r'\\begin\{(table|longtable)\}(.*?)\\end\{\1\}', m, re.S):
        b = mt.group(2)
        lab = re.search(r'\\label\{([^}]+)\}', b); cap = re.search(r'\\caption\{(.*?)\}', b, re.S)
        body = re.sub(r'\\caption\{.*?\}(?=\s*(\\label|\\\\|\n))', '', b, flags=re.S)
        body = re.sub(r'\\label\{[^}]*\}', '', body); body = re.sub(r'\\begin\{(tabular|longtable)\}\{[^}]*\}', '', body)
        rows = []
        for r in body.replace('\\\\', '\n').split('\n'):
            if '&' not in r:
                continue
            rows.append([re.sub(r'\\[a-zA-Z]+\*?|[${}]', '', c).replace('%', '').strip() for c in r.split('&')])
        out.append(dict(label=lab.group(1) if lab else None, caption=re.sub(r'\s+', ' ', cap.group(1)) if cap else '', rows=rows))
    return out

def num(c):
    return float(c) if re.fullmatch(r'-?\d+(?:\.\d+)?(?:e-?\d+)?', c) else None

def tol(c):
    """Half a unit of the last displayed digit of the cell text c (mantissa digits and exponent respected)."""
    m = re.fullmatch(r'-?\d+(?:\.(\d+))?(?:e(-?\d+))?', c)
    d = len(m.group(1)) if m and m.group(1) else 0
    e = int(m.group(2)) if m and m.group(2) else 0
    return 0.5 * 10 ** (-d) * 10 ** e

def ci_parse(c):
    m = re.fullmatch(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', c)
    return (m.group(1), m.group(2)) if m else None

def cell_ok(cell, x):
    if isinstance(x, tuple):
        p = ci_parse(cell)
        return bool(p) and all(abs(float(a) - b) <= tol(a) * (1 + 1e-9) for a, b in zip(p, x))
    return num(cell) is None or abs(num(cell) - x) <= tol(cell) * (1 + 1e-9)

def compare_pivot(s, sup, out):
    """Wide table (rows: values of one stored key, columns: values of another) from a long stored file."""
    pv = s['pivot']; rows = read_source(s['source'])
    lookup = {(float(r[pv['row_key']]), float(r[pv['col_key']])): float(r[pv['value']]) for r in rows}
    with open(out / f"table{s['table_index']:02d}_{(s['label'] or 'unlabelled').replace(':', '_')}.csv", 'w', newline='') as f:
        w = csv.writer(f); w.writerow([pv['row_key']] + [pv['col_key'] + '=' + str(k) for k in sorted({k[1] for k in lookup})])
        for a in sorted({k[0] for k in lookup}):
            w.writerow([a] + [f'{lookup[(a, b)]:.{pv["decimals"]}f}' for b in sorted({k[1] for k in lookup})])
    if sup is None:
        return (s['table_index'], s['label'], 'regenerated')
    t = sup[s['table_index'] - 1]
    hdr = [float(re.sub(r'[^0-9.\-]', '', h.split('=')[1])) for h in t['rows'][0][1:]]
    miss = 0; cells = 0
    for tr in t['rows'][1:]:
        a = float(tr[0])
        for b, cell in zip(hdr, tr[1:]):
            cells += 1
            if (a, b) not in lookup or abs(float(cell) - lookup[(a, b)]) > tol(cell) * (1 + 1e-9):
                miss += 1
    return (s['table_index'], s['label'], f"{'PASS' if miss == 0 else 'FAIL'}: {len(t['rows']) - 1} rows, {cells} numeric cells compared, {miss} cells differ, source {s['source']} (pivot {pv['row_key']} x {pv['col_key']})")

def compare_aggregate(s, sup, out):
    """Table of summary statistics (count, mean, min, max) of stored columns; rows are looked up by their statistic name."""
    import statistics
    ag = s['aggregate']; rows = read_source(s['source'])
    fn = {'count': len, 'mean': statistics.fmean, 'min': min, 'max': max}
    data = {j: [float(r[c]) for r in rows] for j, c in ag['columns'].items()}
    with open(out / f"table{s['table_index']:02d}_{(s['label'] or 'unlabelled').replace(':', '_')}.csv", 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['statistic'] + list(ag['columns'].values()))
        for k in ag['statistics']:
            w.writerow([k] + [f'{fn[k](data[j]):.{ag["decimals"]}f}' for j in ag['columns']])
    if sup is None:
        return (s['table_index'], s['label'], 'regenerated')
    t = sup[s['table_index'] - 1]; miss = 0; cells = 0
    for tr in t['rows'][1:]:
        k = tr[0]
        for j in ag['columns']:
            cells += 1
            if k not in fn or abs(float(tr[int(j)]) - fn[k](data[j])) > tol(tr[int(j)]) * (1 + 1e-9):
                miss += 1
    return (s['table_index'], s['label'], f"{'PASS' if miss == 0 else 'FAIL'}: {len(t['rows']) - 1} rows, {cells} numeric cells compared, {miss} cells differ, source {s['source']} (statistics of stored column values)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--supplement')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = json.load(open(ROOT / 'tools' / 'supplement_table_specs.json'))['tables']
    sup = parse_supplement(a.supplement) if a.supplement else None
    summary = []; bad = 0
    for s in specs:
        if 'aggregate' in s:
            summary.append(compare_aggregate(s, sup, out)); bad += summary[-1][2].startswith('FAIL') if sup else 0
            continue
        if 'pivot' in s:
            summary.append(compare_pivot(s, sup, out)); bad += summary[-1][2].startswith('FAIL') if sup else 0
            continue
        cols = s['columns']; filt = s.get('filter', {})
        main_srcs = s['source'] if isinstance(s['source'], list) else [s['source']]
        srcs = {}
        for src in {c.get('source', ms) for c in cols for ms in main_srcs} | set(main_srcs):
            srcs[src] = [r for r in read_source(src) if all(r.get(k) == v for k, v in filt.items())]
        rows = [r for ms in main_srcs for r in srcs[ms]]
        def val(c, i):
            if 'source' in c:
                return float(srcs[c['source']][i][c['source_column']])
            return float(rows[i][c['source_column']])
        regen = [[None if 'ci' in c else (val(c, i) + c.get('offset', 0)) * c['scale'] for c in cols] for i in range(len(rows))]
        for i in range(len(rows)):
            for k, c in enumerate(cols):
                if 'ci' in c:
                    regen[i][k] = (float(rows[i][c['ci'][0]]), float(rows[i][c['ci'][1]]))
        with open(out / (f"table{s['table_index']:02d}_{(s['label'] or 'unlabelled').replace(':', '_')}.csv"), 'w', newline='') as f:
            w = csv.writer(f); w.writerow([c['header'] or c['source_column'] for c in cols])
            for r in regen:
                w.writerow([fmt(x, c['decimals']) for x, c in zip(r, cols)])
        if sup is None:
            continue
        t = sup[s['table_index'] - 1]
        if t['label'] != s['label'] or not t['caption'].startswith(s['caption_start'][:40]):
            summary.append((s['table_index'], s['label'], 'TABLE NOT FOUND AT INDEX')); bad += 1; continue
        trows = t['rows'][1:]; used = set(); cells = 0; miss = 0
        for tr in trows:
            hit = None
            for k, r in enumerate(regen):
                if k in used:
                    continue
                if all(cell_ok(tr[c['table_column']], x) for x, c in zip(r, cols) if c['table_column'] < len(tr)):
                    hit = k; break
            cells += sum(1 for c in cols if c['table_column'] < len(tr) and (num(tr[c['table_column']]) is not None or ci_parse(tr[c['table_column']])))
            if hit is None:
                miss += 1
            else:
                used.add(hit)
        status = 'PASS' if miss == 0 else 'FAIL'
        bad += miss
        summary.append((s['table_index'], s['label'], f"{status}: {len(trows)} rows, {cells} numeric cells compared, {miss} unmatched rows, source {s['source']} ({len(rows)} stored rows)"))
    if sup is not None:
        (out / 'comparison_summary.txt').write_text('\n'.join(f'{i}\t{l}\t{r}' for i, l, r in summary) + '\n')
        print('\n'.join(f'{i}\t{l}\t{r}' for i, l, r in summary))
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
