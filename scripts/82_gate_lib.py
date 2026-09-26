"""Shared estimators for the 82_* gate scripts (generalizability study of the regeneration gate).

Loaded by the 82a-82f scripts through importlib because the file name starts with a digit.

Design. A cell is one persona (p, 120) under one model (m, 4) and one framing (f, 2). Draws (r) are
nested in cells: iterations of one persona were generated back to back (persona-major run order),
so the draw index is a replicate label, not a crossed occasion. Everything here works from cell
summaries (n, mean, within-cell sum of squares), which makes the persona bootstrap cheap.

Estimators.
  one_facet      p with r nested (one model, one framing). Unbalanced one-way ANOVA method of
                 moments: sigma2_p = (MS_p - MS_e) / n0, n0 = (N - sum n_i^2 / N) / (a - 1).
  two_facet      p x f with r nested (one model). Unweighted-means ANOVA on the cell means with the
                 harmonic cell size n_h; exact expected-mean-squares solution when balanced.
  three_facet    p x m x f with r nested (joint). Same approach, all facets random.
Negative variance estimates are reported raw and truncated at zero for coefficients (Cronbach et
al., 1972; Brennan, 2001).
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTD = os.path.join(BASE, "analysis", "brm")
CORPUS = os.path.join(BASE, "data", "model_outputs_v3.csv")

MODELS = ["deepseek/deepseek-chat-v3", "google/gemini-3-flash-preview",
          "openai/gpt-4o-mini", "z-ai/glm-4.7"]
SHORT = {"deepseek/deepseek-chat-v3": "DeepSeek-V3", "google/gemini-3-flash-preview": "Gemini-3-Flash",
         "openai/gpt-4o-mini": "GPT-4o-mini", "z-ai/glm-4.7": "GLM-4.7"}
FRAMES = ["clinical", "narrative"]
CUTS = [5, 10, 15, 20]                 # None 0-4 | Mild 5-9 | Moderate 10-14 | Mod-severe 15-19 | Severe 20-24
CUT_BOUNDARIES = [4.5, 9.5, 14.5, 19.5]
THRESHOLD = 10
DEC_SOURCES = {"dec28_main", "dec28_restored"}


def load_corpus(december_only=False):
    """v3 corpus, invalid PHQ-8 rows dropped. december_only keeps December clinical rows and all
    narrative rows."""
    v = pd.read_csv(CORPUS, low_memory=False)
    n0 = len(v)
    v = v[v.phq8_valid.astype(str) == "True"].copy()
    assert n0 - len(v) == 1, "expected exactly one invalid PHQ-8 row"
    if december_only:
        keep = (v.prompt_condition == "narrative") | v.row_source.isin(DEC_SOURCES)
        v = v[keep].copy()
    v["y"] = v.phq8_total.astype(float)
    assert v.y.between(0, 24).all()
    v["elev"] = (v.y >= THRESHOLD).astype(float)
    v["band"] = np.digitize(v.y, CUTS)
    v["framing"] = v.prompt_condition
    return v


def cell_summaries(v, outcome="y"):
    """One row per (model, persona, framing): n, mean, within-cell SS and variance."""
    g = v.groupby(["model", "profile_id", "framing"])[outcome]
    s = g.agg(n="size", mean="mean", var=lambda x: x.var(ddof=1)).reset_index()
    s["ss"] = s["var"].fillna(0.0) * (s.n - 1)
    return s


def to_arrays(s, personas, models, frames):
    """Cell summaries to arrays of shape (P, M, F); missing cells are NaN / n = 0."""
    idx = pd.MultiIndex.from_product([models, personas, frames], names=["model", "profile_id", "framing"])
    t = s.set_index(["model", "profile_id", "framing"]).reindex(idx)
    shp = (len(models), len(personas), len(frames))
    mean = t["mean"].to_numpy().reshape(shp).transpose(1, 0, 2)
    n = t["n"].fillna(0).to_numpy().reshape(shp).transpose(1, 0, 2)
    ss = t["ss"].fillna(0).to_numpy().reshape(shp).transpose(1, 0, 2)
    return mean, n, ss


# ----------------------------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------------------------

def one_facet(mean, n, ss):
    """p with r nested. Inputs are 1-D over personas (cells with n = 0 are skipped)."""
    ok = n > 0
    mean, n, ss = mean[ok], n[ok], ss[ok]
    a, N = len(n), n.sum()
    grand = (n * mean).sum() / N
    ms_p = (n * (mean - grand) ** 2).sum() / (a - 1)
    df_e = (n - 1).clip(min=0).sum()
    ms_e = ss.sum() / df_e
    n0 = (N - (n ** 2).sum() / N) / (a - 1)
    return dict(p=(ms_p - ms_e) / n0, e=ms_e, n_cells=int(a), n_draws=int(N), n0=float(n0),
                ms_p=ms_p, ms_e=ms_e, grand=grand)


def _harmonic(n):
    n = n[n > 0]
    return len(n) / (1.0 / n).sum()


def two_facet(mean, n, ss):
    """p x f with r nested, for one model. mean/n/ss have shape (P, F); rows with a missing cell are
    dropped. Unweighted means on cell means, expressed on the draw scale with n_h."""
    keep = (n > 0).all(axis=1)
    mean, n, ss = mean[keep], n[keep], ss[keep]
    P, F = mean.shape
    nh = _harmonic(n.ravel())
    ms_e = ss.sum() / (n - 1).clip(min=0).sum()
    g = mean.mean()
    mp, mf = mean.mean(axis=1), mean.mean(axis=0)
    ms_p = nh * F * ((mp - g) ** 2).sum() / (P - 1)
    ms_f = nh * P * ((mf - g) ** 2).sum() / (F - 1)
    res = mean - mp[:, None] - mf[None, :] + g
    ms_pf = nh * (res ** 2).sum() / ((P - 1) * (F - 1))
    s_pf = (ms_pf - ms_e) / nh
    s_p = (ms_p - ms_pf) / (nh * F)
    s_f = (ms_f - ms_pf) / (nh * P)
    return dict(p=s_p, f=s_f, pf=s_pf, e=ms_e, n_persona=int(P), nh=float(nh))


def three_facet(mean, n, ss):
    """p x m x f with r nested, all random. Shape (P, M, F); personas with any missing cell dropped."""
    keep = (n > 0).all(axis=(1, 2))
    mean, n, ss = mean[keep], n[keep], ss[keep]
    P, M, F = mean.shape
    nh = _harmonic(n.ravel())
    ms_e = ss.sum() / (n - 1).clip(min=0).sum()
    g = mean.mean()
    yp, ym, yf = mean.mean(axis=(1, 2)), mean.mean(axis=(0, 2)), mean.mean(axis=(0, 1))
    ypm, ypf, ymf = mean.mean(axis=2), mean.mean(axis=1), mean.mean(axis=0)
    ss_p = M * F * ((yp - g) ** 2).sum()
    ss_m = P * F * ((ym - g) ** 2).sum()
    ss_f = P * M * ((yf - g) ** 2).sum()
    ss_pm = F * ((ypm - yp[:, None] - ym[None, :] + g) ** 2).sum()
    ss_pf = M * ((ypf - yp[:, None] - yf[None, :] + g) ** 2).sum()
    ss_mf = P * ((ymf - ym[:, None] - yf[None, :] + g) ** 2).sum()
    r3 = (mean - ypm[:, :, None] - ypf[:, None, :] - ymf[None, :, :]
          + yp[:, None, None] + ym[None, :, None] + yf[None, None, :] - g)
    ss_pmf = (r3 ** 2).sum()
    ms = dict(p=ss_p / (P - 1), m=ss_m / (M - 1), f=ss_f / (F - 1),
              pm=ss_pm / ((P - 1) * (M - 1)), pf=ss_pf / ((P - 1) * (F - 1)),
              mf=ss_mf / ((M - 1) * (F - 1)), pmf=ss_pmf / ((P - 1) * (M - 1) * (F - 1)))
    ms = {k: nh * x for k, x in ms.items()}
    s = {}
    s["pmf"] = (ms["pmf"] - ms_e) / nh
    s["pm"] = (ms["pm"] - ms["pmf"]) / (nh * F)
    s["pf"] = (ms["pf"] - ms["pmf"]) / (nh * M)
    s["mf"] = (ms["mf"] - ms["pmf"]) / (nh * P)
    s["p"] = (ms["p"] - ms["pm"] - ms["pf"] + ms["pmf"]) / (nh * M * F)
    s["m"] = (ms["m"] - ms["pm"] - ms["mf"] + ms["pmf"]) / (nh * P * F)
    s["f"] = (ms["f"] - ms["pf"] - ms["mf"] + ms["pmf"]) / (nh * P * M)
    s["e"] = ms_e
    s["n_persona"] = int(P)
    s["nh"] = float(nh)
    return s


# ----------------------------------------------------------------------------------------------
# D-study
# ----------------------------------------------------------------------------------------------

def pos(x):
    return max(float(x), 0.0)


def phi_k(s_p, s_err, k):
    """Dependability of a k-draw mean: s_p / (s_p + s_err / k), components truncated at zero."""
    sp, se = pos(s_p), pos(s_err)
    if sp + se == 0:
        return np.nan
    return sp / (sp + se / k)


def k_for_phi(s_p, s_err, target):
    """Smallest integer k with phi(k) >= target; inf when the persona component is zero."""
    sp, se = pos(s_p), pos(s_err)
    if se == 0:
        return 1.0
    if sp == 0:
        return np.inf
    return float(max(1, int(np.ceil(target / (1 - target) * se / sp - 1e-12))))


def k_for_se(s_err, tol):
    """Smallest integer k with sqrt(s_err / k) <= tol."""
    se = pos(s_err)
    return float(max(1, int(np.ceil(se / tol ** 2 - 1e-12))))


def fmt_k(k):
    return "inf" if not np.isfinite(k) else str(int(k))
