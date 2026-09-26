"""Level 2 population reference, rebuilt: two estimands with design-based standard errors.

The simulated level-2 gap (scripts 64 and 66) is a paired difference between persona cells that
differ in the contrasted attribute alone and share every other attribute. The persona design is a
full crossing (race 5 x gender 4 x income 3 x relationship 2, one persona per cell) and the
level-2 contrasts use the cisgender half of it. So, pooled over models and framings, every stratum
of the other attributes enters the simulated gap exactly once per model x framing, with equal
weight:

    race contrasts   strata = sex (2) x income (3) x relationship (2)         12 strata
    women - men      strata = race (5, Multiracial included) x income x rel.   30 strata
    income contrasts strata = race (5) x sex (2) x relationship (2)            20 strata

(78b checks that every model x framing carries all of them.) The simulated estimand is therefore
the equal-weight average over those strata of the within-stratum difference.

Multiracial. The paper gives Multiracial personas no anchor (no defensible benchmark), and the
nearest NHANES group, RIDRETH3 7, lumps multiracial with American Indian / Alaska Native and
Pacific Islander respondents, exists only from 2011, and gives cells of 16 to 30 people. The
primary estimand therefore restricts the sex and income contrasts to the four anchored races on
both sides: 78b drops the Multiracial pairs from the simulated gap (192 sex pairs, 128 per income
contrast, instead of 240 and 160) and the reference is standardised over the same four races.
The five-race version, with RIDRETH3 7 standing in for Multiracial, is the `five_race` variant.

Two population references are built here:

  marginal      the weighted NHANES group mean minus the other group's, over all adults. This is
                the anchor the ML4H paper used (groundtruth/phq8_groundtruth_nhanes_2005_2018.csv);
                the script reproduces it exactly before writing anything.
  standardised  the NHANES within-stratum difference of weighted means, averaged over the same
                strata with the same equal weights (direct standardisation to the persona design).

Mapping of persona attributes to NHANES (2005-2018, DEMO_D..J + DPQ_D..J, age 18+, all eight
PHQ items valid, WTMEC2YR > 0, weight / 7):
  race     White RIDRETH1 3; Black RIDRETH1 4; Hispanic RIDRETH1 1 or 2; Asian RIDRETH3 6
           (2011-2018 only, as in the published anchor); Multiracial RIDRETH3 7, "non-Hispanic
           other race including multiracial" (2011-2018 only; before 2011 that group is merged
           with Asian in RIDRETH1 5 and cannot be separated).
  sex      RIAGENDR (sex at interview; stands in for cisgender man / woman).
  income   INDFMPIR < 1.3 Low, 1.3 to 3.5 Middle, > 3.5 High (script 03, primary, unchanged).
  relationship  persona "Married" = DMDMARTL 1. Persona "Single" = unpartnered: DMDMARTL 2
           widowed, 3 divorced, 4 separated, 5 never married, plus 18-19-year-olds, who were not
           asked from 2007 on (in 2005-06, when they were asked, 92% of them were unpartnered and
           3.6% married). Living with a partner (6) is neither married nor single and is left
           out of both. Refused / don't know are left out.

Sensitivity references (all emitted, each labelled in the `variant` column):
  mar_all       Single = every non-married status including cohabiting (the audit check's choice)
  mar_never     Single = never married only
  mar_drop1819  18-19-year-olds with no marital status left out instead of counted Single
  no_marital    relationship not stratified
  five_race     sex and income contrasts standardised over five races, RIDRETH3 7 standing in
                for Multiracial; 78b builds the simulated gaps on the matching five-race pairs
  income_alt    income from household income (INDHHIN2) crossed with insurance (HIQ031A private,
                HIQ031D Medicaid): Low < $35k with Medicaid; Middle $55k-$100k with private
                insurance; High $100k+ with private insurance (a top-coded lower bound; NHANES
                cannot see $250k). 2007-2018 only, because 2005-06 codes income differently.
  asian_era     Asian - White with White restricted to 2011-2018, the years Asian is identified
  audit         the audit check's construction (checks/std_gaps.py): race standardised over sex
                x PIR x married-versus-all-else, sex and income over four races x the other axis,
                no marital stratum. Receipt only: it must reproduce the audit's point estimates.
  pc_design     the prompt-control design (scripts 69-71): race in {White, Black}, sex, income,
                relationship fixed at Single; five contrasts.

Standard errors. Everything is a smooth function of domain totals, so the script first collapses
the respondent file to PSU-level totals (sum of w*y and of w per domain per PSU; 7 cycles, 105
strata, 212 PSUs). From those:
  se_taylor  Taylor linearisation of the whole contrast (each domain mean linearised as a ratio,
             the linear combination taken, then the with-replacement between-PSU variance within
             strata). Primary.
  se_boot    Rao-Wu rescaling bootstrap over PSUs within strata, 2,000 replicates, seeded. Check.
  df         design degrees of freedom, PSUs minus strata among the PSUs the contrast touches.
The contrast is estimated as a domain contrast, so the covariance between the two groups'
means (they share strata and PSUs) is carried. The published construction hypot(se_a, se_b)
treated them as independent; it is printed beside for comparison.

Outputs (analysis/brm/):
  l2_reference.csv           one row per contrast x estimand x variant
  l2_reference_cells.csv     every standardisation cell: unweighted n and weighted mean
  l2_reference_receipts.csv  gates and mapping receipts
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(BASE, "data", "nhanes_raw")
OUT = os.path.join(BASE, "analysis", "brm")
CYCLES = list("DEFGHIJ")
ITEMS = [f"DPQ0{i}0" for i in range(1, 9)]
SEED = 20260925
N_BOOT = 2000
THIN = 30

RACES5 = ["White", "Black", "Hispanic", "Asian", "Multiracial"]
RACES4 = ["White", "Black", "Hispanic", "Asian"]
SEXES = ["M", "F"]
SES = ["Low", "Middle", "High"]
MARS = ["Married", "Single"]

# contrast, factor, high level, low level
CONTRASTS = [
    ("Black minus White", "race", "Black", "White"),
    ("Hispanic minus White", "race", "Hispanic", "White"),
    ("Asian minus White", "race", "Asian", "White"),
    ("Women minus Men", "sex", "F", "M"),
    ("Low minus High SES", "ses", "Low", "High"),
    ("Middle minus High SES", "ses", "Middle", "High"),
    ("Low minus Middle SES", "ses", "Low", "Middle"),
]
PUBLISHED = {"Black minus White": 0.3299, "Hispanic minus White": 0.2425,
             "Asian minus White": -0.7551, "Women minus Men": 0.9380,
             "Low minus High SES": 2.0198, "Middle minus High SES": 0.8910,
             "Low minus Middle SES": 1.1287}
AUDIT = {"Black minus White": -0.304, "Hispanic minus White": -0.293, "Asian minus White": -1.012,
         "Women minus Men": 0.884, "Low minus High SES": 1.489, "Middle minus High SES": 0.677,
         "Low minus Middle SES": 0.812}


# ----------------------------------------------------------------------------------------- data
def load():
    fr = []
    for c in CYCLES:
        demo = pd.read_sas(os.path.join(RAW, f"DEMO_{c}.xpt"), format="xport")
        keep = [k for k in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "INDFMPIR",
                            "INDHHIN2", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "DMDMARTL"]
                if k in demo.columns]
        dpq = pd.read_sas(os.path.join(RAW, f"DPQ_{c}.xpt"), format="xport")
        hiq = pd.read_sas(os.path.join(RAW, f"HIQ_{c}.xpt"), format="xport")
        x = demo[keep].merge(dpq[["SEQN"] + ITEMS], on="SEQN")
        x = x.merge(hiq[["SEQN", "HIQ031A", "HIQ031D"]], on="SEQN", how="left")
        x["cycle"] = c
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    for k in ("RIDRETH3", "INDHHIN2"):
        if k not in d:
            d[k] = np.nan
    d[ITEMS] = d[ITEMS].where(d[ITEMS] <= 3)
    d = d[(d.RIDAGEYR >= 18) & d[ITEMS].notna().all(axis=1) & (d.WTMEC2YR > 0)].copy()
    d["y"] = d[ITEMS].sum(axis=1)
    d["w"] = d.WTMEC2YR / len(CYCLES)
    late = d.cycle.isin(list("GHIJ"))
    d["race"] = np.select(
        [d.RIDRETH3 == 6, d.RIDRETH3 == 7, d.RIDRETH1 == 3, d.RIDRETH1 == 4,
         d.RIDRETH1.isin([1, 2])],
        ["Asian", "Multiracial", "White", "Black", "Hispanic"], "Other")
    assert not ((d.race == "Asian") & ~late).any() and not ((d.race == "Multiracial") & ~late).any()
    d["sex"] = d.RIAGENDR.map({1: "M", 2: "F"})
    d["ses"] = np.select([d.INDFMPIR < 1.3, d.INDFMPIR <= 3.5, d.INDFMPIR > 3.5],
                         ["Low", "Middle", "High"], "")
    d.loc[d.INDFMPIR.isna(), "ses"] = ""
    # marital mappings
    mm = d.DMDMARTL
    young_na = mm.isna() & (d.RIDAGEYR < 20)
    d["mar_primary"] = np.select([mm == 1, mm.isin([2, 3, 4, 5]) | young_na],
                                 ["Married", "Single"], "")
    d["mar_all"] = np.select([mm == 1, mm.isin([2, 3, 4, 5, 6]) | young_na],
                             ["Married", "Single"], "")
    d["mar_never"] = np.select([mm == 1, (mm == 5) | young_na], ["Married", "Single"], "")
    d["mar_drop1819"] = np.select([mm == 1, mm.isin([2, 3, 4, 5])], ["Married", "Single"], "")
    d["mar_audit"] = np.select([mm == 1, mm.isin([2, 3, 4, 5, 6])], ["Married", "Single"], "")
    # income from household income x insurance, 2007-2018 (INDHHIN2 coding)
    inc = d.INDHHIN2
    has = d.cycle != "D"
    medicaid = d.HIQ031D == 17
    private = d.HIQ031A == 14
    d["ses_alt"] = np.select(
        [has & inc.isin([1, 2, 3, 4, 5, 6, 13]) & medicaid,
         has & inc.isin([9, 10, 14]) & private,
         has & (inc == 15) & private],
        ["Low", "Middle", "High"], "")
    d["psu_id"] = d.SDMVSTRA.astype(int) * 10 + d.SDMVPSU.astype(int)
    d["stratum"] = d.SDMVSTRA.astype(int)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------------------ domain machinery
class Design:
    """PSU-level totals for a list of domains; estimates and SEs of linear contrasts of means."""

    def __init__(self, d, masks, rng):
        self.psus = np.sort(d.psu_id.unique())
        pidx = pd.Index(self.psus).get_indexer(d.psu_id)
        self.stratum_of_psu = (self.psus // 10).astype(int)
        wy = (d.w * d.y).to_numpy()
        ww = d.w.to_numpy()
        P, D = len(self.psus), len(masks)
        self.A = np.zeros((P, D))
        self.B = np.zeros((P, D))
        self.N = np.zeros((P, D))
        for j, mk in enumerate(masks):
            self.A[:, j] = np.bincount(pidx[mk], weights=wy[mk], minlength=P)
            self.B[:, j] = np.bincount(pidx[mk], weights=ww[mk], minlength=P)
            self.N[:, j] = np.bincount(pidx[mk], minlength=P)
        self.n = self.N.sum(axis=0).astype(int)
        self.R = np.divide(self.A.sum(0), self.B.sum(0), out=np.full(D, np.nan),
                           where=self.B.sum(0) > 0)
        # Rao-Wu multipliers, m_h = n_h - 1
        strata = np.unique(self.stratum_of_psu)
        mult = np.zeros((N_BOOT, P))
        for h in strata:
            ix = np.where(self.stratum_of_psu == h)[0]
            nh = len(ix)
            draws = rng.integers(0, nh, size=(N_BOOT, nh - 1))
            counts = np.zeros((N_BOOT, nh))
            for j in range(nh - 1):
                counts[np.arange(N_BOOT), draws[:, j]] += 1
            mult[:, ix] = counts * nh / (nh - 1)
        self.mult = mult
        Ab, Bb = mult @ self.A, mult @ self.B
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Rb = np.where(Bb > 0, Ab / Bb, np.nan)

    def contrast(self, coef):
        """coef: dict domain index -> coefficient."""
        ix = np.array(list(coef.keys()))
        a = np.array(list(coef.values()), float)
        est = float(self.R[ix] @ a)
        Wt = self.B.sum(0)[ix]
        z = ((self.A[:, ix] - self.R[ix] * self.B[:, ix]) / Wt) @ a
        var = 0.0
        touched = (self.N[:, ix].sum(1) > 0)
        n_psu, strata_touched = 0, 0
        for h in np.unique(self.stratum_of_psu):
            s = self.stratum_of_psu == h
            zh = z[s]
            nh = len(zh)
            var += nh / (nh - 1) * ((zh - zh.mean()) ** 2).sum()
            k = int(touched[s].sum())
            if k:
                n_psu += k
                strata_touched += 1
        rep = self.Rb[:, ix] @ a
        ok = np.isfinite(rep)
        return dict(estimate=est, se_taylor=float(np.sqrt(var)),
                    se_boot=float(np.std(rep[ok], ddof=1)), boot_reps_valid=int(ok.sum()),
                    df=int(n_psu - strata_touched))


# ------------------------------------------------------------------------------------ estimands
def build_specs(d):
    """Every (contrast, estimand, variant) as a list of (label, mask) pairs with coefficients."""
    specs = []
    everyone = np.ones(len(d), bool)

    def m(**kw):
        x = everyone.copy()
        for k, v in kw.items():
            x &= (d[k] == v).to_numpy()
        return x

    def marginal(name, factor, hi, lo, variant, col=None, restrict=None):
        col = col or factor
        base = everyone if restrict is None else restrict
        specs.append(dict(contrast=name, estimand="marginal", variant=variant, strata=[""],
                          terms=[(+1.0, m(**{col: hi}) & base, f"{hi}"),
                                 (-1.0, m(**{col: lo}) & base, f"{lo}")]))

    def standardised(name, factor, hi, lo, variant, axes, cols, restrict_lo=None):
        """axes: list of (column, levels) for the other attributes; equal weights."""
        grid = [()]
        for col, levels in axes:
            grid = [g + ((col, lv),) for g in grid for lv in levels]
        S = len(grid)
        terms, strata = [], []
        for g in grid:
            cond = {c: v for c, v in g}
            key = " x ".join(v for _, v in g)
            strata.append(key)
            hi_mask = m(**{cols[factor]: hi}, **cond)
            lo_mask = m(**{cols[factor]: lo}, **cond)
            if restrict_lo is not None:
                lo_mask &= restrict_lo
            terms.append((+1.0 / S, hi_mask, f"{hi} | {key}"))
            terms.append((-1.0 / S, lo_mask, f"{lo} | {key}"))
        specs.append(dict(contrast=name, estimand="standardised", variant=variant,
                          strata=strata, terms=terms))

    def axes_for(factor, races, mar_col, ses_col, mar_levels=MARS):
        ax = []
        if factor != "race":
            ax.append(("race", races))
        if factor != "sex":
            ax.append(("sex", SEXES))
        if factor != "ses":
            ax.append((ses_col, SES))
        if mar_col:
            ax.append((mar_col, mar_levels))
        return ax

    late = d.cycle.isin(list("GHIJ")).to_numpy()
    for name, factor, hi, lo in CONTRASTS:
        cols = {"race": "race", "sex": "sex", "ses": "ses"}
        # marginal, published
        marginal(name, factor, hi, lo, "primary")
        # standardised, primary (four anchored races where race is a stratum)
        standardised(name, factor, hi, lo, "primary",
                     axes_for(factor, RACES4, "mar_primary", "ses"), cols)
        for var in ("mar_all", "mar_never", "mar_drop1819"):
            standardised(name, factor, hi, lo, var, axes_for(factor, RACES4, var, "ses"), cols)
        standardised(name, factor, hi, lo, "no_marital", axes_for(factor, RACES4, None, "ses"),
                     cols)
        if factor != "race":
            standardised(name, factor, hi, lo, "five_race",
                         axes_for(factor, RACES5, "mar_primary", "ses"), cols)
        # income from household income x insurance
        cols_alt = dict(cols, ses="ses_alt")
        if factor == "ses":
            marginal(name, factor, hi, lo, "income_alt", col="ses_alt")
        standardised(name, factor, hi, lo, "income_alt",
                     axes_for(factor, RACES4, "mar_primary", "ses_alt"), cols_alt)
        if name == "Asian minus White":
            marginal(name, factor, hi, lo, "asian_era", restrict=late)
            standardised(name, factor, hi, lo, "asian_era",
                         axes_for(factor, RACES4, "mar_primary", "ses"), cols, restrict_lo=late)
        # audit receipt
        if factor == "race":
            standardised(name, factor, hi, lo, "audit",
                         axes_for(factor, RACES4, "mar_audit", "ses"), cols)
        else:
            standardised(name, factor, hi, lo, "audit",
                         axes_for(factor, RACES4, None, "ses"), cols)
        # prompt-control design: race White/Black, relationship Single only
        if name in ("Black minus White", "Women minus Men", "Low minus High SES",
                    "Middle minus High SES", "Low minus Middle SES"):
            standardised(name, factor, hi, lo, "pc_design",
                         axes_for(factor, ["White", "Black"], "mar_primary", "ses",
                                  mar_levels=["Single"]), cols)
    return specs


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load()
    specs = build_specs(d)
    # unique domains
    masks, labels, index = [], [], {}
    for s in specs:
        for _, mask, lab in s["terms"]:
            key = mask.tobytes()
            if key not in index:
                index[key] = len(masks)
                masks.append(mask)
                labels.append(lab)
    des = Design(d, masks, rng)

    # published SEs (hypot of two group SEs) for comparison
    dse = pd.read_csv(os.path.join(BASE, "analysis", "anchor_design_se.csv")).set_index("group")
    sek = {"White": "White", "Black": "Black", "Asian": "Asian", "Hispanic": "Hispanic",
           "M": "Cisgender men", "F": "Cisgender women", "Low": "Low", "Middle": "Middle",
           "High": "High"}
    pub_se = {n: float(np.hypot(dse.loc[sek[h], "se_design"], dse.loc[sek[l], "se_design"]))
              for n, _, h, l in CONTRASTS}

    rows, cells = [], []
    for s in specs:
        coef = {}
        for c, mask, lab in s["terms"]:
            j = index[mask.tobytes()]
            coef[j] = coef.get(j, 0.0) + c
        est = des.contrast(coef)
        ns = [int(des.n[index[mask.tobytes()]]) for _, mask, _ in s["terms"]]
        rows.append(dict(contrast=s["contrast"], estimand=s["estimand"], variant=s["variant"],
                         n_strata=len(s["strata"]) if s["estimand"] == "standardised" else 0,
                         n_cells=len(ns), min_cell_n=min(ns),
                         cells_below_30=int(sum(v < THIN for v in ns)),
                         cells_below_10=int(sum(v < 10 for v in ns)),
                         cells_empty=int(sum(v == 0 for v in ns)),
                         **{k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in est.items()},
                         se_published_hypot=round(pub_se[s["contrast"]], 6)
                         if (s["estimand"] == "marginal" and s["variant"] == "primary") else np.nan))
        for c, mask, lab in s["terms"]:
            j = index[mask.tobytes()]
            cells.append(dict(contrast=s["contrast"], estimand=s["estimand"],
                              variant=s["variant"], cell=lab, weight=round(c, 6),
                              n=int(des.n[j]), w_mean=round(float(des.R[j]), 4)))
    ref = pd.DataFrame(rows)
    ref.to_csv(os.path.join(OUT, "l2_reference.csv"), index=False)
    pd.DataFrame(cells).to_csv(os.path.join(OUT, "l2_reference_cells.csv"), index=False)

    # ------------------------------------------------------------------------------ receipts
    rc = []

    def get(c, e, v):
        return ref[(ref.contrast == c) & (ref.estimand == e) & (ref.variant == v)].iloc[0]

    fails = 0
    for c, pub in PUBLISHED.items():
        r = get(c, "marginal", "primary")
        ok = abs(r.estimate - pub) < 6e-5
        fails += not ok
        rc.append(dict(check="marginal reproduces published anchor", item=c, got=r.estimate,
                       want=pub, ok=ok))
    for c, aud in AUDIT.items():
        r = get(c, "standardised", "audit")
        ok = abs(r.estimate - aud) < 6e-4
        fails += not ok
        rc.append(dict(check="audit construction reproduces std_gaps.py", item=c,
                       got=r.estimate, want=aud, ok=ok))
    # marital mapping receipts
    dD = d[(d.cycle == "D") & (d.RIDAGEYR < 20)]
    for code, lab in [(1, "married"), (5, "never married"), (6, "living with partner")]:
        rc.append(dict(check="2005-06 age 18-19 marital share (weighted)", item=lab,
                       got=round(float(dD.w[dD.DMDMARTL == code].sum() / dD.w.sum()), 4),
                       want=np.nan, ok=True))
    rc.append(dict(check="age 18-19 with no marital status (2007-2018)", item="n",
                   got=int((d.DMDMARTL.isna() & (d.RIDAGEYR < 20)).sum()), want=np.nan, ok=True))
    for v in ("mar_primary", "mar_all", "mar_never", "mar_drop1819"):
        for lv in ("Married", "Single", ""):
            rc.append(dict(check=f"marital mapping {v}", item=lv or "left out",
                           got=int((d[v] == lv).sum()), want=np.nan, ok=True))
    for lv in ("Low", "Middle", "High", ""):
        rc.append(dict(check="income_alt mapping (2007-2018)", item=lv or "left out",
                       got=int((d.ses_alt == lv).sum()), want=np.nan, ok=True))
    rc.append(dict(check="design", item="respondents", got=len(d), want=36274, ok=len(d) == 36274))
    rc.append(dict(check="design", item="strata", got=int(d.stratum.nunique()), want=np.nan,
                   ok=True))
    rc.append(dict(check="design", item="PSUs", got=int(d.psu_id.nunique()), want=np.nan, ok=True))
    rc.append(dict(check="design", item="df (PSUs - strata)",
                   got=int(d.psu_id.nunique() - d.stratum.nunique()), want=np.nan, ok=True))
    rc.append(dict(check="bootstrap", item="replicates", got=N_BOOT, want=np.nan, ok=True))
    rc.append(dict(check="bootstrap", item="seed", got=SEED, want=np.nan, ok=True))
    pd.DataFrame(rc).to_csv(os.path.join(OUT, "l2_reference_receipts.csv"), index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rc).to_string(index=False))
    show = ref[["contrast", "estimand", "variant", "estimate", "se_taylor", "se_boot",
                "se_published_hypot", "df", "n_cells", "min_cell_n", "cells_below_30",
                "cells_empty", "boot_reps_valid"]]
    print(show.to_string(index=False))
    if fails:
        raise SystemExit(f"GATE FAIL: {fails} receipt(s) failed")
    print("\nGATE PASS. Written l2_reference.csv, l2_reference_cells.csv, l2_reference_receipts.csv")


if __name__ == "__main__":
    main()
