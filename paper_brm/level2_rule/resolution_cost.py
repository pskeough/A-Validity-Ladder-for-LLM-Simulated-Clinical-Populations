"""For each of the paper's pooled and per-model level-2 rows: how many times the current number
of matched pairs the R3 interval rule would need to give a single-name verdict, holding the
point estimates and the per-pair spread fixed. "anchor-limited" means no amount of simulation
is enough, because the population gap's own SE (NHANES design SE) keeps the interval across a
boundary. A planning figure, not a power analysis: it assumes the point estimates stay put."""
from pathlib import Path

import numpy as np
import pandas as pd

from level2_rule_eval import LABELS, REPO, interval_rule, fieller

OUT = Path(__file__).resolve().parent / "results"
pap = pd.read_csv(REPO / "analysis" / "level2_permodel_equivalence.csv")
pap = pap[pap.se.notna() & pap.scope_type.isin(["pooled", "model"])]
K = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e6])
rows = []
for r in pap.itertuples():
    args = lambda k: ([r.simulated], [r.se / np.sqrt(k)], [r.population], [r.pop_se], [0.0], [r.df])
    now = interval_rule(*args(1))[0][0]
    need, lab = None, None
    for k in K:
        v = interval_rule(*args(k))[0][0]
        if v in LABELS:
            need, lab = k, v
            break
    lo, hi = fieller(*[np.array(a, float) for a in args(1)])
    lo_inf, hi_inf = fieller(*[np.array(a, float) for a in args(1e6)])
    rows.append(dict(contrast=r.contrast, scope=r.scope.split("/")[-1], n_pairs=r.n_pairs,
                     ratio=round(r.simulated / r.population, 3),
                     ratio_ci90=f"{lo[0]:.2f} to {hi[0]:.2f}",
                     ratio_ci90_if_simulation_exact=f"{lo_inf[0]:.2f} to {hi_inf[0]:.2f}",
                     published=r.verdict, R3_now=now,
                     pairs_multiplier_for_single_name=("already" if now in LABELS else
                                                       ("anchor-limited" if need is None or need >= 1e6 else need)),
                     single_name_reached=lab))
pd.DataFrame(rows).to_csv(OUT / "paper_resolution_cost.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
