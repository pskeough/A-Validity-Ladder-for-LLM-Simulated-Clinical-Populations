"""Level 1 rebuild, step e: unit of analysis, pattern reuse, and planted controls.

1. Unit. Level 1 is scored on single draws. For contrast, the gateway rule is also applied to cell
   means (the mean item vector of the 30 draws in a model x persona x framing cell). Averaging pulls
   every item toward its cell mean, so a cell mean is not a response vector a respondent could give;
   the contrast shows how much the unit changes the rate.

2. Pattern reuse. Within each total score, the probability that two draws from different personas
   of the same model and framing give the identical eight-item vector (cross-persona collision), set
   against the same probability for two different NHANES respondents at that total (unweighted).
   Pairs within a persona are excluded because repeated draws of one persona are not independent
   people. Both are averaged over the model's own total distribution.

3. Planted controls. The corpus design (rows, personas, totals) is kept and the item vectors are
   replaced, so each control has the models' total mix:
     nhanes_same_total   a weighted random NHANES adult with the same total (should pass)
     random_routing      a uniform random composition of the total over eight 0-3 items
                         (should read less coherent)
     most_regular        the vector with the highest lz* among all vectors with that total
                         (should read excess overfit)
     cardinal_routing    the NHANES adult's own eight values, with the two largest moved to items 1
                         and 2 (the textbook routing a stereotype would produce)
   Each is scored with the same GRM, the same total-matched tails and the same gateway O/E as the
   corpus, with persona-clustered 90% intervals (NHANES reference held fixed, B = 1000).

Emits (analysis/brm/): l1_unit_cellmeans.csv, l1_pattern_reuse.csv, l1_planted_controls.csv,
l1_most_regular_vectors.csv.
Seeded (20260929).
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("l1", os.path.join(HERE, "79_l1_lib.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
spec_c = importlib.util.spec_from_file_location("pf", os.path.join(HERE, "79c_l1_personfit.py"))
PF = importlib.util.module_from_spec(spec_c)
spec_c.loader.exec_module(PF)
spec_b = importlib.util.spec_from_file_location("gw", os.path.join(HERE, "79b_l1_gateway.py"))
GW = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(GW)

rng = np.random.default_rng(20260929)
BB = 1000


def collision(patterns_key, persona, totals):
    """Cross-persona collision probability per total. Returns dict total -> (prob, n)."""
    out = {}
    df = pd.DataFrame(dict(k=patterns_key, p=persona, t=totals))
    for t, g in df.groupby("t"):
        n = len(g)
        if n < 2:
            continue
        nk = g.k.value_counts().to_numpy(float)
        npk = g.groupby(["p", "k"]).size().to_numpy(float)
        np_ = g.p.value_counts().to_numpy(float)
        denom = n ** 2 - (np_ ** 2).sum()
        if denom <= 0:
            continue
        out[t] = ((nk ** 2).sum() - (npk ** 2).sum()) / denom, n
    return out


def main():
    corpus = L.load_corpus()
    nall = pd.read_csv(os.path.join(L.OUT, "l1_scores_nhanes.csv"))
    n0 = nall[nall.reference == "2005_2018"].reset_index(drop=True)
    par = pd.read_csv(os.path.join(L.OUT, "l1_grm_params.csv"))
    pw = par[par.fit == "weighted"]
    a, b = pw.a.to_numpy(), pw[["b1", "b2", "b3"]].to_numpy()

    # ---------------- 1. unit ----------------
    rows = []
    cm = corpus.groupby(["model", "profile_id", "framing"])[L.ITEMS].mean().reset_index()
    cm["total"] = cm[L.ITEMS].sum(axis=1)
    for mdl in L.MODELS + ["pooled"]:
        c = cm if mdl == "pooled" else cm[cm.model == mdl]
        d = corpus if mdl == "pooled" else corpus[corpus.model == mdl]
        ec = c[c.total >= 10]
        ed = d[d.total >= 10]
        rows.append(dict(model=L.SHORT.get(mdl, mdl), n_cells=len(c), elevated_cells=len(ec),
                         cellmean_violation_rate=((ec.phq8_1 < 2) & (ec.phq8_2 < 2)).mean() if len(ec) else np.nan,
                         n_draws=len(d), elevated_draws=len(ed),
                         draw_violation_rate=((ed.phq8_1 < 2) & (ed.phq8_2 < 2)).mean(),
                         share_cellmean_items_integer=float(np.isclose(c[L.ITEMS] % 1, 0).all(axis=1).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "l1_unit_cellmeans.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---------------- 2. pattern reuse ----------------
    key_c = (corpus[L.ITEMS].to_numpy() * 4 ** np.arange(8)).sum(axis=1)
    key_n = (n0[L.DPQ].to_numpy() * 4 ** np.arange(8)).sum(axis=1)
    nh_col = collision(key_n, np.arange(len(n0)), n0.total.to_numpy())
    rr = []
    for mdl in L.MODELS:
        for fr in ["clinical", "narrative"]:
            m = ((corpus.model == mdl) & (corpus.framing == fr)).to_numpy()
            cc = collision(key_c[m], corpus.profile_id.to_numpy()[m], corpus.total.to_numpy()[m])
            ts = [t for t in cc if t in nh_col]
            w = np.array([cc[t][1] for t in ts], float)
            pm = np.average([cc[t][0] for t in ts], weights=w)
            pn = np.average([nh_col[t][0] for t in ts], weights=w)
            rr.append(dict(model=L.SHORT[mdl], framing=fr, n_draws=int(m.sum()),
                           unique_vectors=len(np.unique(key_c[m])),
                           cross_persona_collision=pm, nhanes_collision_at_model_totals=pn, ratio=pm / pn,
                           share_draws_on_top10_vectors=pd.Series(key_c[m]).value_counts().iloc[:10].sum() / m.sum()))
    rr.append(dict(model="NHANES 2005-2018", framing="", n_draws=len(n0), unique_vectors=len(np.unique(key_n)),
                   share_draws_on_top10_vectors=pd.Series(key_n).value_counts().iloc[:10].sum() / len(n0)))
    pd.DataFrame(rr).to_csv(os.path.join(L.OUT, "l1_pattern_reuse.csv"), index=False)
    print(pd.DataFrame(rr).round(4).to_string(index=False))

    # ---------------- 3. planted controls ----------------
    tot = corpus.total.to_numpy()
    # NHANES adult with the same total, weighted
    by_t = {t: g.index.to_numpy() for t, g in n0.groupby("total")}
    Xn = n0[L.DPQ].to_numpy()
    wn = n0.w.to_numpy()
    pick = np.empty(len(corpus), int)
    for t in np.unique(tot):
        ix = np.where(tot == t)[0]
        pool = by_t[t]
        pick[ix] = rng.choice(pool, size=len(ix), p=wn[pool] / wn[pool].sum())
    X_nh = Xn[pick]
    # uniform composition
    X_rr = np.empty((len(corpus), 8), int)
    for t in np.unique(tot):
        ix = np.where(tot == t)[0]
        X_rr[ix] = L.compositions_uniform(int(t), len(ix), rng)
    # most regular vector per total
    allpat = np.stack([(np.arange(4 ** 8) // 4 ** j) % 4 for j in range(8)], axis=1)
    _, _, lzs_all = L.score_vectors(a, b, allpat)
    alltot = allpat.sum(axis=1)
    best = {t: allpat[np.where(alltot == t)[0][np.argmax(lzs_all[alltot == t])]] for t in range(25)}
    X_mr = np.stack([best[t] for t in tot])
    # cardinal routing
    X_cr = X_nh.copy()
    for i in range(len(X_cr)):
        v = X_nh[i]
        top2 = [int(p) for p in np.argsort(-v, kind="stable")[:2]]
        # the two largest values go to items 1 and 2; the other six values fill items 3-8 in order
        X_cr[i] = [v[top2[0]], v[top2[1]]] + [v[k] for k in range(8) if k not in top2]
    assert (X_cr.sum(axis=1) == tot).all() and (X_mr.sum(axis=1) == tot).all() and (X_nh.sum(axis=1) == tot).all()
    assert (X_rr.sum(axis=1) == tot).all()

    smap = L.total_strata(n0.total.to_numpy(), 100)
    ref = PF.Ref(n0, "lzstar_weighted", smap)
    e_nh = n0[n0.total >= 10]
    v_pt, _ = GW.nh_band_rates(e_nh, e_nh.w.to_numpy()[None, :])
    v_pt = np.nan_to_num(v_pt[0])
    persona = (corpus.model + "|" + corpus.profile_id).to_numpy()
    pu, pinv = np.unique(persona, return_inverse=True)
    pmodel = np.array([x.split("|")[0] for x in pu])
    idx = np.concatenate([np.where(pmodel == s)[0][rng.integers(0, (pmodel == s).sum(), size=(BB, (pmodel == s).sum()))]
                          for s in np.unique(pmodel)], axis=1)
    out = []
    for lab, X in [("observed_corpus", corpus[L.ITEMS].to_numpy()), ("nhanes_same_total", X_nh),
                   ("random_routing", X_rr), ("most_regular", X_mr), ("cardinal_routing", X_cr)]:
        _, _, lzs = L.score_vectors(a, b, X)
        cl, ch, cmid, _, _ = ref.tails(tot, lzs)
        el = tot >= 10
        viol = ((X[:, 0] < 2) & (X[:, 1] < 2) & el).astype(float)
        expv = np.where(el, v_pt[np.clip(GW.band_of(tot), 0, None)], 0.0)
        for mdl in ["pooled"] + L.MODELS:
            mm = np.ones(len(corpus), bool) if mdl == "pooled" else (corpus.model == mdl).to_numpy()
            ii = idx if mdl == "pooled" else np.where(pmodel == mdl)[0][
                rng.integers(0, (pmodel == mdl).sum(), size=(BB, (pmodel == mdl).sum()))]

            def boot(x):
                s = np.bincount(pinv[mm], weights=x[mm], minlength=len(pu))
                c = np.bincount(pinv[mm], minlength=len(pu)).astype(float)
                return s[ii].sum(axis=1) / c[ii].sum(axis=1)
            rm, ro = boot(cl) / 0.05, boot(ch) / 0.05
            s_v = np.bincount(pinv[mm], weights=viol[mm], minlength=len(pu))
            s_e = np.bincount(pinv[mm], weights=expv[mm], minlength=len(pu))
            oe_b = s_v[ii].sum(axis=1) / s_e[ii].sum(axis=1)
            r = dict(control=lab, model=L.SHORT.get(mdl, mdl), n=int(mm.sum()),
                     misfit_ratio=cl[mm].mean() / 0.05, misfit_ci90_lo=np.quantile(rm, 0.05),
                     misfit_ci90_hi=np.quantile(rm, 0.95), overfit_ratio=ch[mm].mean() / 0.05,
                     overfit_ci90_lo=np.quantile(ro, 0.05), overfit_ci90_hi=np.quantile(ro, 0.95),
                     mean_mid_pit=cmid[mm].mean(), gateway_violation_rate=viol[mm & el].mean(),
                     gateway_O_over_E=viol[mm].sum() / expv[mm].sum(), gateway_OE_ci90_lo=np.quantile(oe_b, 0.05),
                     gateway_OE_ci90_hi=np.quantile(oe_b, 0.95))
            _, _, r["personfit_verdict_tau1.5"] = PF.read_verdict(dict(
                misfit_ratio_ci90_lo=r["misfit_ci90_lo"], misfit_ratio_ci90_hi=r["misfit_ci90_hi"],
                overfit_ratio_ci90_lo=r["overfit_ci90_lo"], overfit_ratio_ci90_hi=r["overfit_ci90_hi"]), 1.5)
            r["gateway_verdict_tau1.5"] = L.verdict(r["gateway_OE_ci90_lo"], r["gateway_OE_ci90_hi"], 1.5) \
                .replace("below", "too prototypical").replace("above", "less coherent")
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(L.OUT, "l1_planted_controls.csv"), index=False)
    print(res[res.model == "pooled"].round(3).to_string(index=False))
    rv = pd.DataFrame([dict(total=t, most_regular_vector="-".join(map(str, best[t])),
                            lzstar=float(lzs_all[(alltot == t)].max())) for t in range(25)])
    rv.to_csv(os.path.join(L.OUT, "l1_most_regular_vectors.csv"), index=False)


if __name__ == "__main__":
    main()
