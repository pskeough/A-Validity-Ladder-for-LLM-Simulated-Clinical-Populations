"""Shakedown of the ladder on another group's release: PersonaLLM (Jiang et al., 2024, NAACL Findings),
BFI-44 answers from GPT-3.5-turbo-0613, GPT-4-0613 and Llama-2 under 32 Big Five persona types,
10 draws each at temperature 0.7, against human BFI-44 answers from Twin-2K-500 (Toubia et al., 2025).

Purpose. The ladder's rules were set on the PHQ-8 corpus. Running them unchanged on a different
instrument (five scales, five response options, reverse-keyed items), a different persona design
(trait instructions, no demographics) and a different reference (an online panel, no survey design)
shows which rules carry over and which break. Rules may change on this run only; the frozen ladder
is then applied unchanged to the remaining external releases.

Data (cloned or downloaded 2 Oct 2026, raw files under paper_brm/external/raw):
  personallm/repo   github.com/hjian42/PersonaLLM, commit 286c149 (MIT licence);
                    outputs/<model>/temp0.7/bfi/<persona>_p<draw>.json, "(a) 5\\n(b) 2 ...".
  twin2k            huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500 (CC BY 4.0);
                    question_catalog_and_human_response_csv/wave1_3_response.csv, QID25_1..44
                    (BFI-44, 1-5), QID12 sex, QID13 age, QID15 race.
PersonaLLM words the items in the BFI's simpler form ("Talks a lot" for "Is talkative"); the 44
items are in the same order and keyed the same way.

Rungs, per model x Big Five scale (items reverse-keyed and scored 0-4):
  gate  91_ladder_core.gate: 32 personas, k = 10, tolerances in human SD units.
  L1    GRM fitted to the humans; lz* tail ratios, persona bootstrap x person bootstrap (B = 2,000);
        tau from the human subgroups (sex, age band, race) by the paper's rule.
  L4    R1 and R2 against the human loadings (B = 500), R3 within-persona diagnostic.
        R4, L2 and L3 need persona attributes that a population records; the PersonaLLM personas
        have none, so those rungs are not applicable (recorded as such).
Output: paper_brm/external/personallm/92_*.csv and SHAKEDOWN.md (written by hand from them).
Seed: default_rng(20261092).
"""
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "paper_brm", "external", "raw")
OUT = os.path.join(BASE, "paper_brm", "external", "personallm")
SEED = 20261092
B_L1, B_L4 = 2000, 500
MODELS = {"gpt-3.5-turbo-0613": "GPT-3.5", "gpt-4-0613": "GPT-4", "llama-2": "Llama-2"}
# BFI-44 keying (John & Srivastava, 1999), 1-based item numbers, negative = reverse-keyed
SCALES = {
    "Extraversion": [1, -6, 11, 16, -21, 26, -31, 36],
    "Agreeableness": [-2, 7, -12, 17, 22, -27, 32, -37, 42],
    "Conscientiousness": [3, -8, 13, -18, -23, 28, 33, 38, -43],
    "Neuroticism": [4, -9, 14, 19, -24, 29, -34, 39],
    "Openness": [5, 10, 15, 20, 25, 30, -35, 40, -41, 44],
}
LETTERS = [chr(97 + i) for i in range(26)] + ["a" + chr(97 + i) for i in range(18)]


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = [old[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


V = _load("core91", "91_ladder_core.py")


def parse_answer(txt):
    """44 answers 1-5 from "(a) 5\\n(b) 2 ..."; None when any item is missing or out of range."""
    got = dict(re.findall(r"\(([a-z]{1,2})\)\s*([1-5])\b", txt))
    if not all(k in got for k in LETTERS):
        return None
    return [int(got[k]) for k in LETTERS]


def load_personallm():
    rows, bad = [], []
    for mdir, short in MODELS.items():
        for f in sorted(glob.glob(os.path.join(RAW, "personallm", "repo", "outputs", mdir, "temp0.7", "bfi", "*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            ans = parse_answer(j["annotation"])
            if ans is None:
                bad.append(dict(model=short, file=os.path.basename(f)))
                continue
            rows.append(dict(model=short, persona=j["persona_encoding"], draw=j["iteration"],
                             **{f"i{k + 1}": v for k, v in enumerate(ans)}))
    return pd.DataFrame(rows), pd.DataFrame(bad, columns=["model", "file"])


def load_humans():
    d = pd.read_csv(os.path.join(RAW, "twin2k", "wave1_3_response.csv"), low_memory=False)
    h = d[["QID12", "QID13", "QID15"] + [f"QID25_{i}" for i in range(1, 45)]].dropna()
    h = h.rename(columns={f"QID25_{i}": f"i{i}" for i in range(1, 45)})
    return h.reset_index(drop=True)


def keyed(df, items):
    """Items scored 0-4 with reverse-keyed items flipped."""
    return np.column_stack([(df[f"i{abs(i)}"].to_numpy(int) - 1) if i > 0 else (5 - df[f"i{abs(i)}"].to_numpy(int))
                            for i in items])


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sim, bad = load_personallm()
    hum = load_humans()
    bad.to_csv(os.path.join(OUT, "92_unparsed.csv"), index=False)
    print(f"PersonaLLM draws parsed: {len(sim)}; unparsed {len(bad)}; humans {len(hum)}", flush=True)
    print(sim.groupby("model").agg(n=("draw", "size"), personas=("persona", "nunique")).to_string(), flush=True)
    for q in ("QID12", "QID13", "QID15"):
        print(q, hum[q].value_counts().head(8).to_dict(), flush=True)

    sex = hum.QID12.to_numpy()
    age = hum.QID13.to_numpy()
    race = hum.QID15.to_numpy()
    groups = {f"sex={v}": sex == v for v in np.unique(sex)}
    groups.update({f"age={v}": age == v for v in np.unique(age)})
    groups.update({f"race={v}": race == v for v in np.unique(race) if (race == v).sum() >= 100})

    g_rows, l1_rows, tau_rows, l4_rows = [], [], [], []
    for scale, items in SCALES.items():
        J, M = len(items), 5
        Xh = keyed(hum, items)
        th = Xh.sum(1)
        sd = float(th.std(ddof=1))
        grm = V.GRM(J, M)
        fit = grm.fit(Xh)
        _, _, lz_h = grm.score(fit["a"], fit["b"], Xh)
        ref1 = V.L1Ref(th, lz_h, None, J * (M - 1), B_L1, rng)
        tau, worst, trows = V.tau_from_subgroups(ref1, th, lz_h, {k: m for k, m in groups.items() if m.sum() >= 100})
        tau_rows += [dict(scale=scale, **r) for r in trows]
        ref4 = V.L4Ref(Xh, None, M, B_L4, rng)
        l4_rows.append(dict(scale=scale, model="humans (Twin-2K)", n=len(Xh), ev_ratio=ref4.ev[0] / ref4.ev[1],
                            load_min=ref4.lam.min(), R1_absolute=ref4.gf,
                            lam=" ".join(f"{x:.2f}" for x in ref4.lam)))
        print(f"{scale}: GRM converged {fit['converged']}, human SD {sd:.2f}, tau {tau} (worst resolved subgroup "
              f"{worst:.2f}, worst point {max(r['point_departure'] for r in trows):.2f}) | humans R1 (absolute) {ref4.gf} "
              f"(ev {ref4.ev[0] / ref4.ev[1]:.1f}, loadings {' '.join(f'{x:.2f}' for x in ref4.lam)})", flush=True)
        for model, s in sim.groupby("model"):
            Xs = keyed(s, items)
            ts = Xs.sum(1)
            per = s.persona.to_numpy()
            g = V.gate(ts, per, 10, sd)
            g_rows.append(dict(scale=scale, model=model, **g))
            _, _, lz_s = grm.score(fit["a"], fit["b"], Xs)
            r1 = V.l1_eval(ref1, ts, lz_s, per, rng, tau=tau)
            l1_rows.append(dict(scale=scale, model=model, n=len(Xs), sim_mean_total=ts.mean(),
                                human_mean_total=th.mean(), **r1))
            r4 = V.l4_eval(Xs, per, ref4, rng)
            lam = r4.pop("lam")
            l4_rows.append(dict(scale=scale, model=model, n=len(Xs), **r4,
                                lam=" ".join(f"{x:.2f}" for x in lam)))
            print(f"  {model:8s} gate phi(10) {g['phi_k']:.3f} SE {g['se_k']:.2f}/{g['tol_min']:.2f} "
                  f"pass_min {g['pass_min']} | L1 misfit {r1['misfit_ratio']:.2f} [{r1['misfit_ratio_ci90_lo']:.2f}, "
                  f"{r1['misfit_ratio_ci90_hi']:.2f}] overfit {r1['overfit_ratio']:.2f} [{r1['overfit_ratio_ci90_lo']:.2f}, "
                  f"{r1['overfit_ratio_ci90_hi']:.2f}] {r1['verdict']} | L4 R1 {r4['R1']} (ev {r4['ev_ratio']:.1f}, "
                  f"min {r4['load_min']:.2f}) R2 {r4['R2']} (phi {r4['phi']:.3f}, lo {r4['phi_lo90']:.3f}; equal {r4['phi_equal_loadings']:.3f}; "
                  f"rmsd {r4['loading_rmsd']:.3f}, hi {r4['loading_rmsd_hi90']:.3f}) "
                  f"R3 {r4['R3']}", flush=True)
    pd.DataFrame(g_rows).to_csv(os.path.join(OUT, "92_gate.csv"), index=False)
    pd.DataFrame(l1_rows).to_csv(os.path.join(OUT, "92_l1.csv"), index=False)
    pd.DataFrame(tau_rows).to_csv(os.path.join(OUT, "92_l1_tau.csv"), index=False)
    pd.DataFrame(l4_rows).to_csv(os.path.join(OUT, "92_l4.csv"), index=False)


if __name__ == "__main__":
    main()
