"""R2 size condition on the corpus: root mean square difference between each population's loadings
and the NHANES loadings, with its upper 90% limit from the same paired bootstrap replicates that
81b uses for the congruence interval (l4_boot_loadings{_dec}.npz, replicate b of the population
against replicate b of NHANES). R2 passes when the congruence lower 90% limit is at least .95 and
the RMSD upper 90% limit is at most .10.
Output: analysis/brm/l4_r2_size.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "analysis", "brm")
R2_PHI, R2_RMSD = 0.95, 0.10
NH = "NHANES_2005-2018_(weighted)"


def main():
    rows = []
    for suf, subset in (("", "all"), ("_dec", "december")):
        z = np.load(os.path.join(OUT, f"l4_boot_loadings{suf}.npz"))
        st = pd.read_csv(os.path.join(OUT, f"l4_structure{suf}.csv"))
        bootN = z[NH]
        lamN = st[st.population.str.replace(" ", "_") == NH][[f"load_{i}" for i in range(1, 9)]].to_numpy()[0]
        for key in z.files:
            if key == NH:
                continue
            r = st[st.population.str.replace(" ", "_") == key].iloc[0]
            lam = r[[f"load_{i}" for i in range(1, 9)]].to_numpy(float)
            bl = z[key]
            nb = min(len(bl), len(bootN))
            rm = np.sqrt(np.mean((bl[:nb] - bootN[:nb]) ** 2, axis=1))
            hi = float(np.quantile(rm, 0.95))
            rows.append(dict(subset=subset, population=r.population, phi_nhanes=r.phi_nhanes,
                             phi_lo90=r.phi_nhanes_lo90, loading_rmsd=float(np.sqrt(np.mean((lam - lamN) ** 2))),
                             loading_rmsd_hi90=hi, R2_shape=bool(r.phi_nhanes_lo90 >= R2_PHI),
                             R2_size=bool(hi <= R2_RMSD), R2=bool(r.phi_nhanes_lo90 >= R2_PHI and hi <= R2_RMSD),
                             load_mean=lam.mean(), nhanes_load_mean=lamN.mean()))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "l4_r2_size.csv"), index=False)
    pd.set_option("display.width", 220)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
