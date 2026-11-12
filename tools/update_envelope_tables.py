"""Rewrite the four envelope tables of the supplement from results/*_tube_envelope*.csv (componentwise margin)."""
import re, sys, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'
tex_path = Path(sys.argv[1]); s = tex_path.read_text()
for label, csv in (('tab:S_tes_case9', 'case9_tube_envelope_stress.csv'), ('tab:S_te_case9', 'case9_tube_envelope.csv'),
                   ('tab:S_tes_case39', 'case39_tube_envelope_stress.csv'), ('tab:S_te_case39', 'case39_tube_envelope.csv')):
    d = pd.read_csv(R / csv)
    body = '\n'.join(f'{int(k)} & {e:.5f}\\\\' for k, e in zip(d.k, d.eps_line_angle_rad))
    pat = re.compile(r'(\\label\{' + re.escape(label) + r'\}\s*\\begin\{tabular\}\{lc\}\s*\\toprule\s*Step \$k\$ & \$\\varepsilon_k\$ \[rad\]\\\\\s*\\midrule\s*).*?(\s*\\bottomrule)', re.S)
    s, n = pat.subn(lambda m: m.group(1) + body + m.group(2), s); assert n == 1, label
s = s.replace('envelope of the largest line-angle difference at nominal loading', 'envelope of the largest componentwise error (all line angles and rotor-to-bus angles) at nominal loading')
tex_path.write_text(s)
