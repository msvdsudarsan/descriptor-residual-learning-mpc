"""Joins results/case39_application_authority_x*.csv into results/case39_application_authority.csv."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'
d = pd.concat([pd.read_csv(f) for f in sorted(R.glob('case39_application_authority_x*.csv'))]).sort_values(['authority_scale'], kind='stable')
d.to_csv(R / 'case39_application_authority.csv', index=False); print(d)
