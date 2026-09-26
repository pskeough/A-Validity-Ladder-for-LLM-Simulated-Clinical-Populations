"""Level 4, part 1: item structure of each model's simulated population against NHANES.

Unit. A simulated population is one model in one framing: every valid draw, one draw one respondent
(81_l4_lib docstring). NHANES is 2005-2018 adults, MEC-weighted. Pooled rows (over models, or over a
model's two framings) are emitted as description and carry no verdict.

For every population:
  polychoric matrix (two-step ML, weighted tables for NHANES) and Pearson matrix
  eigenvalues, lambda1/lambda2, parallel analysis (100 column permutations, polychoric, 95th pct)
  one-factor ULS (minres) loadings on the polychoric matrix, SRMR, omega total
  general factor: every loading >= .30 and lambda1/lambda2 >= 3 (stated before any model was fitted;
    NHANES clears it by a wide margin, see output)
  Tucker's phi of the loadings with NHANES, with a percentile interval from B cluster-bootstrap
    replicates on each side (personas for the corpus, Rao-Wu PSU-within-stratum for NHANES), and the
    share of replicates in which the general-factor rule holds
  loading RMSD against NHANES, phi against a flat loading vector (a calibration of how much of phi
    any all-positive vector gets for free)

Also:
  between/within split: share of each item's variance between cells (persona; NHANES demographic
    cell), and the one-factor structure of the within-cell (cell-centred) correlation matrix, on
    the full frame and on the 48 anchored cells (sex x 4 races x 3 incomes x 2 relationship), with
    B_WB cluster-bootstrap intervals; NHANES also on finer cells (adding age band and cycle) to
    check that its within-cell factor does not depend on how coarse the cells are
  composition check: the 48 anchored personas reweighted to the NHANES weighted share of their cell
  group-vs-group phi within each source for the seven anchored contrasts (the published level-4
    statistic), per model and framing, with B_GROUP persona bootstraps within group

usage: python 81b_l4_structure.py [all|dec] [B] [B_GROUP] [B_WB]
Emits analysis/brm/l4_structure{_dec}.csv, l4_polychoric{_dec}.csv, l4_between_within{_dec}.csv,
l4_composition{_dec}.csv, l4_group_congruence{_dec}.csv, l4_boot_loadings{_dec}.npz.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l4lib", os.path.join(HERE, "81_l4_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "all"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
B_GROUP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
B_WB = int(sys.argv[4]) if len(sys.argv) > 4 else 500
N_PA = 100
SEED = 8102
SUF = "" if SUBSET == "all" else "_" + SUBSET
IT = L.ITEMS

CONTRASTS = [("Women vs Men", "sex", "F", "M"), ("Black vs White", "race", "Black", "White"),
             ("Asian vs White", "race", "Asian", "White"),
             ("Hispanic vs White", "race", "Hispanic", "White"),
             ("Low vs High income", "ses", "Low", "High"),
             ("Middle vs High income", "ses", "Middle", "High"),
             ("Low vs Middle income", "ses", "Low", "Middle")]


def fit_R(R):
    lam, srmr, omega = L.one_factor_uls(R)
    ev = L.eig_desc(R)
    return lam, srmr, omega, ev


def parallel_analysis(X, rng):
    """Polychoric parallel analysis: permute each column independently (keeps every marginal)."""
    ev_obs = L.eig_desc(L.poly_from(L.Tables(X, np.ones(len(X)), np.zeros(len(X))).agg()))
    null = np.empty((N_PA, L.P))
    for t in range(N_PA):
        Xp = np.column_stack([rng.permutation(X[:, i]) for i in range(L.P)])
        null[t] = L.eig_desc(L.poly_from(L.Tables(Xp, np.ones(len(X)), np.zeros(len(X))).agg()))
    thr = np.quantile(null, 0.95, axis=0)
    k = 0
    while k < L.P and ev_obs[k] > thr[k]:
        k += 1
    return k, thr


def boot_loadings(T, rng, nboot, strata=None):
    lams, gfs, ratios = [], [], []
    for _ in range(nboot):
        mult = L.cluster_boot_mult(T.C, rng, strata)
        R = L.poly_from(T.agg(mult))
        lam, _, _ = L.one_factor_uls(R)
        ev = L.eig_desc(R)
        lams.append(lam); gfs.append(L.general_factor(lam, ev)); ratios.append(ev[0] / ev[1])
    return np.array(lams), np.array(gfs), np.array(ratios)


class CellUnits:
    """Weighted sums per (cluster, cell) unit, so the within/between split can be bootstrapped by
    resampling clusters. For the corpus the cell is the persona and the cluster is the persona; for
    NHANES the cell is the demographic cell and the cluster is the PSU."""

    def __init__(self, df, cellcol):
        X = df[IT].to_numpy(float); w = df.w.to_numpy(float)
        key = df.cluster.astype(str) + "#" + df[cellcol].astype(str)
        _, ui = np.unique(key.to_numpy(), return_inverse=True)
        U = ui.max() + 1
        self.cells, cell_of_row = np.unique(df[cellcol].astype(str).to_numpy(), return_inverse=True)
        self.clusters, cl_of_row = np.unique(df.cluster.to_numpy(), return_inverse=True)
        self.u_cell = np.zeros(U, int); self.u_cell[ui] = cell_of_row
        self.u_cl = np.zeros(U, int); self.u_cl[ui] = cl_of_row
        self.s0 = np.bincount(ui, weights=w, minlength=U)
        self.s1 = np.stack([np.bincount(ui, weights=w * X[:, i], minlength=U) for i in range(L.P)], 1)
        self.s2 = np.zeros((U, L.P, L.P))
        for i in range(L.P):
            for j in range(i, L.P):
                v = np.bincount(ui, weights=w * X[:, i] * X[:, j], minlength=U)
                self.s2[:, i, j] = v; self.s2[:, j, i] = v
        self.C = len(self.clusters)

    def split(self, mult=None):
        um = np.ones(len(self.s0)) if mult is None else mult[self.u_cl]
        nc = len(self.cells)
        S0 = np.bincount(self.u_cell, weights=um * self.s0, minlength=nc)
        S1 = np.stack([np.bincount(self.u_cell, weights=um * self.s1[:, i], minlength=nc)
                       for i in range(L.P)], 1)
        S2 = np.tensordot(um, self.s2, 1)
        ok = S0 > 0
        N = S0.sum()
        mu = S1.sum(0) / N
        M = S1[ok] / S0[ok, None]
        Sw = (S2 - np.einsum("c,ci,cj->ij", S0[ok], M, M)) / N
        Bm = M - mu
        Sb = (Bm * S0[ok, None]).T @ Bm / N
        share = np.diag(Sb) / np.diag(Sw + Sb)
        d = np.sqrt(np.diag(Sw)); db = np.sqrt(np.diag(Sb))
        return share, Sw / np.outer(d, d), Sb / np.outer(db, db), int(ok.sum())


def within_between(df, cellcol, ref=None, nboot=0, rng=None, strata_of=None):
    """Between-cell share, within-cell one-factor summary and bootstrap intervals."""
    cu = CellUnits(df, cellcol)
    share, Rw, Rb, nc = cu.split()
    lw, _, _, evw = fit_R(Rw)
    lb, _, _, evb = fit_R(Rb)
    out = dict(n_cells=nc, between_share_mean=share.mean(), between_share_min=share.min(),
               between_share_max=share.max(), within_ev_ratio=evw[0] / evw[1],
               within_load_min=lw.min(), within_general_factor=L.general_factor(lw, evw),
               within_phi_nhanes_within=np.nan if ref is None else L.congruence(lw, ref),
               between_ev1_share=evb[0] / L.P, between_load_min=lb.min())
    if nboot:
        strata = None if strata_of is None else np.array([strata_of[c] for c in cu.clusters])
        bs, br, bp, bg = [], [], [], []
        for _ in range(nboot):
            mult = L.cluster_boot_mult(cu.C, rng, strata)
            s, Rw_b, _, _ = cu.split(mult)
            l_b, _, _, ev_b = fit_R(Rw_b)
            bs.append(s.mean()); br.append(ev_b[0] / ev_b[1]); bg.append(L.general_factor(l_b, ev_b))
            if ref is not None:
                bp.append(L.congruence(l_b, ref))
        out.update(between_share_lo95=np.quantile(bs, 0.025), between_share_hi95=np.quantile(bs, 0.975),
                   within_ev_ratio_lo95=np.quantile(br, 0.025), within_ev_ratio_hi95=np.quantile(br, 0.975),
                   within_gf_boot_share=float(np.mean(bg)))
        if ref is not None:
            out.update(within_phi_lo95=np.quantile(bp, 0.025), within_phi_hi95=np.quantile(bp, 0.975))
    return out, lw


def main():
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    sim = L.load_sim(SUBSET)
    nh = L.load_nhanes()
    print(f"subset={SUBSET}  corpus rows {len(sim)}  NHANES respondents {len(nh)}  B={B}  "
          f"B_GROUP={B_GROUP}")

    # ---------------------------------------------------------------- populations
    pops = [("NHANES 2005-2018 (weighted)", "NHANES", "population", "all", nh, True)]
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            pops.append((f"{mdl} {fr}", "corpus", mdl, fr,
                         sim[(sim.model_s == mdl) & (sim.prompt_condition == fr)], False))
    for mdl in L.MODELS:
        pops.append((f"{mdl} both framings (descriptive)", "corpus pooled", mdl, "both",
                     sim[sim.model_s == mdl], False))
    for fr in ("clinical", "narrative"):
        pops.append((f"four models pooled, {fr} (descriptive)", "corpus pooled", "pooled", fr,
                     sim[sim.prompt_condition == fr].assign(cluster=lambda d: d.model_s + "|" + d.cluster),
                     False))

    rows, polyrows, boots = [], [], {}
    lamN = None
    bootN = None
    for idx, (label, source, mdl, fr, df, is_nh) in enumerate(pops):
        rng = np.random.default_rng([SEED, idx])
        T = L.Tables(df[IT].values, df.w.values, df.cluster.values)
        strata = None
        if is_nh:
            s_of = df.groupby("cluster").stratum.first().to_dict()
            strata = L.cluster_strata(T, s_of)
        a = T.agg()
        Rp = L.poly_from(a)
        Rr = L.pearson_from(a)
        lam, srmr, omega, ev = fit_R(Rp)
        lamr, srmr_r, _, evr = fit_R(Rr)
        k_pa, thr = parallel_analysis(df[IT].to_numpy(int), rng)
        bl, bgf, bratio = boot_loadings(T, rng, B, strata)
        boots[label] = bl
        if is_nh:
            lamN, bootN = lam, bl
            lamN_r = lamr
        phi = L.congruence(lam, lamN)
        nb = min(len(bl), len(bootN))
        phib = np.array([L.congruence(bl[b], bootN[b]) for b in range(nb)])
        row = dict(population=label, source=source, model=mdl, framing=fr, subset=SUBSET,
                   n_rows=len(df), n_clusters=T.C,
                   ev1=ev[0], ev2=ev[1], ev3=ev[2], ev_ratio_12=ev[0] / ev[1],
                   ev_ratio_12_boot_lo=np.quantile(bratio, 0.025),
                   ev_ratio_12_boot_hi=np.quantile(bratio, 0.975),
                   pa_factors_retained=k_pa, pa_thr1=thr[0], pa_thr2=thr[1],
                   **{f"load_{i + 1}": lam[i] for i in range(L.P)},
                   load_min=lam.min(), n_load_below_30=int((lam < 0.30).sum()),
                   n_load_negative=int((lam < 0).sum()),
                   srmr_1f=srmr, omega_total=omega,
                   general_factor=L.general_factor(lam, ev), general_factor_boot_share=bgf.mean(),
                   phi_nhanes=phi, phi_nhanes_lo95=np.quantile(phib, 0.025),
                   phi_nhanes_lo90=np.quantile(phib, 0.05), phi_nhanes_hi95=np.quantile(phib, 0.975),
                   loading_rmsd_nhanes=float(np.sqrt(np.mean((lam - lamN) ** 2))),
                   phi_flat=L.congruence(lam, np.ones(L.P)),
                   pearson_ev_ratio_12=evr[0] / evr[1], pearson_load_min=lamr.min(),
                   pearson_general_factor=L.general_factor(lamr, evr),
                   pearson_phi_nhanes=L.congruence(lamr, lamN_r), n_boot=B)
        rows.append(row)
        for i in range(L.P):
            for j in range(L.P):
                if i < j:
                    polyrows.append(dict(population=label, item_i=IT[i], item_j=IT[j],
                                         polychoric=Rp[i, j], pearson=Rr[i, j]))
        print(f"[{time.time() - t0:6.0f}s] {label:44s} ev1/ev2 {ev[0] / ev[1]:5.2f}  min load "
              f"{lam.min():+.2f}  GF {row['general_factor']!s:5s} ({bgf.mean():.2f})  PA {k_pa}  "
              f"phi {phi:.3f} [{row['phi_nhanes_lo95']:.3f}, {row['phi_nhanes_hi95']:.3f}]  "
              f"SRMR {srmr:.3f}", flush=True)

    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(L.OUT, f"l4_structure{SUF}.csv"), index=False)
    pd.DataFrame(polyrows).to_csv(os.path.join(L.OUT, f"l4_polychoric{SUF}.csv"), index=False)
    np.savez_compressed(os.path.join(L.OUT, f"l4_boot_loadings{SUF}.npz"),
                        **{k.replace(" ", "_"): v for k, v in boots.items()})

    # NHANES self-congruence: how far a bootstrap replicate of the population sits from itself
    selfphi = np.array([L.congruence(b, lamN) for b in bootN])
    print(f"NHANES bootstrap self-congruence: median {np.median(selfphi):.4f}, "
          f"2.5% {np.quantile(selfphi, 0.025):.4f}; phi with a flat vector {L.congruence(lamN, np.ones(8)):.4f}")
    # NHANES rows do not depend on the corpus subset (same data, same seed), so no suffix
    pd.DataFrame([dict(nhanes_self_phi_median=np.median(selfphi),
                       nhanes_self_phi_p025=np.quantile(selfphi, 0.025),
                       nhanes_self_phi_min=selfphi.min(), n_boot=len(selfphi),
                       phi_flat_vs_nhanes=L.congruence(lamN, np.ones(L.P)))]).to_csv(
        os.path.join(L.OUT, f"l4_nhanes_selfphi{SUF}.csv"), index=False)

    # ---------------------------------------------------------------- between / within
    nh = nh.copy()
    nh["cell_full"] = nh.sex + "|" + nh.race + "|" + nh.ses + "|" + nh.rel
    anch = (nh.race != "Other") & (nh.ses != "") & (nh.rel != "")
    nh48 = nh[anch].copy()
    wb_rows = []
    s_of_nh = nh.groupby("cluster").stratum.first().to_dict()
    ab = pd.cut(nh.RIDAGEYR, [17, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"]).astype(str)
    nh["cell_age"] = nh.cell_full + "|" + ab + "|" + nh.cycle
    r, ref_full = within_between(nh, "cell_full", None, B_WB, np.random.default_rng([SEED, 900]), s_of_nh)
    r.update(population="NHANES", frame="full", within_phi_nhanes_within=1.0)
    wb_rows.append(r)
    r, lw48 = within_between(nh48, "cell_full", None, B_WB, np.random.default_rng([SEED, 901]), s_of_nh)
    r.update(population="NHANES", frame="anchored48", within_phi_nhanes_within=1.0)
    wb_rows.append(r)
    r, _ = within_between(nh, "cell_age", ref_full, B_WB, np.random.default_rng([SEED, 902]), s_of_nh)
    r.update(population="NHANES", frame="full x age band x cycle (finer cells)")
    wb_rows.append(r)
    k = 0
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            g = sim[(sim.model_s == mdl) & (sim.prompt_condition == fr)]
            for frame, gg, ref in (("full", g, ref_full),
                                   ("anchored48", g[g.sex.notna() & (g.race != "Multiracial")], lw48)):
                k += 1
                r, _ = within_between(gg, "profile_id", ref, B_WB, np.random.default_rng([SEED, 910 + k]))
                r.update(population=f"{mdl} {fr}", frame=frame)
                wb_rows.append(r)
    wb = pd.DataFrame(wb_rows)
    first = ["population", "frame"]
    wb = wb[first + [c for c in wb.columns if c not in first]]
    wb.to_csv(os.path.join(L.OUT, f"l4_between_within{SUF}.csv"), index=False)
    print("\nBETWEEN / WITHIN")
    print(wb.round(3).to_string(index=False))

    # ---------------------------------------------------------------- composition
    cellw = nh48.groupby("cell_full").w.sum() / nh48.w.sum()
    T48 = L.Tables(nh48[IT].values, nh48.w.values, nh48.cluster.values)
    l48, _, _, _ = fit_R(L.poly_from(T48.agg()))
    comp = []
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            g = sim[(sim.model_s == mdl) & (sim.prompt_condition == fr) & sim.sex.notna()
                    & (sim.race != "Multiracial")].copy()
            g["cell_full"] = g.sex + "|" + g.race + "|" + g.ses + "|" + g.rel
            npers = g.groupby("cell_full").profile_id.nunique()
            assert (npers == 1).all()
            ndraw = g.groupby("cell_full").size()
            # December-only runs can lack a persona's clinical draws; its NHANES share is dropped
            # and the rest renormalised
            g["w_std"] = g.cell_full.map(cellw / ndraw)
            out = dict(population=f"{mdl} {fr}", n_cells_present=len(npers))
            for tag, wcol in (("design", None), ("nhanes_composition", "w_std")):
                ww = np.ones(len(g)) if wcol is None else g[wcol].to_numpy()
                Tg = L.Tables(g[IT].values, ww, g.cluster.values)
                lam, srmr, _, ev = fit_R(L.poly_from(Tg.agg()))
                out[f"phi_{tag}"] = L.congruence(lam, l48)
                out[f"ev_ratio_{tag}"] = ev[0] / ev[1]
                out[f"load_min_{tag}"] = lam.min()
                out[f"gf_{tag}"] = L.general_factor(lam, ev)
            comp.append(out)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(L.OUT, f"l4_composition{SUF}.csv"), index=False)
    print("\nCOMPOSITION (48 anchored cells; phi against NHANES restricted to the same 48 cells)")
    print(comp.round(3).to_string(index=False))

    # ---------------------------------------------------------------- group-vs-group congruence
    gpops = [("NHANES", nh, True)]
    for mdl in L.MODELS:
        for fr in ("clinical", "narrative"):
            gpops.append((f"{mdl} {fr}", sim[(sim.model_s == mdl) & (sim.prompt_condition == fr)], False))
    for fr in ("clinical", "narrative"):
        gpops.append((f"four models pooled, {fr} (descriptive)",
                      sim[sim.prompt_condition == fr].assign(cluster=lambda d: d.model_s + "|" + d.cluster),
                      False))
    grows = []
    for gi, (label, df, is_nh) in enumerate(gpops):
        for ci, (cname, col, ga, gb) in enumerate(CONTRASTS):
            rng = np.random.default_rng([SEED, 500 + gi, ci])
            res = {}
            for tag, grp in (("a", ga), ("b", gb)):
                sub = df[df[col] == grp]
                T = L.Tables(sub[IT].values, sub.w.values, sub.cluster.values)
                strata = None
                if is_nh:
                    strata = L.cluster_strata(T, sub.groupby("cluster").stratum.first().to_dict())
                R = L.poly_from(T.agg())
                lam, _, _, ev = fit_R(R)
                lamr, _, _, _ = fit_R(L.pearson_from(T.agg()))
                bl, bgf, _ = boot_loadings(T, rng, B_GROUP, strata)
                res[tag] = (lam, ev, bl, bgf, len(sub), T.C, lamr)
            la, eva, bla, gfa, na, ca, lra = res["a"]
            lb, evb, blb, gfb, nb_, cb, lrb = res["b"]
            phib = np.array([L.congruence(bla[b], blb[b]) for b in range(B_GROUP)])
            grows.append(dict(population=label, contrast=cname, n_a=na, n_b=nb_, clusters_a=ca,
                              clusters_b=cb, phi=L.congruence(la, lb),
                              phi_lo95=np.quantile(phib, 0.025), phi_hi95=np.quantile(phib, 0.975),
                              phi_pearson=L.congruence(lra, lrb),
                              gf_a=L.general_factor(la, eva), gf_b=L.general_factor(lb, evb),
                              load_min_a=la.min(), load_min_b=lb.min(), n_boot=B_GROUP))
            print(f"[{time.time() - t0:6.0f}s] {label:40s} {cname:22s} phi {grows[-1]['phi']:.3f} "
                  f"[{grows[-1]['phi_lo95']:.3f}, {grows[-1]['phi_hi95']:.3f}]", flush=True)
    pd.DataFrame(grows).to_csv(os.path.join(L.OUT, f"l4_group_congruence{SUF}.csv"), index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
