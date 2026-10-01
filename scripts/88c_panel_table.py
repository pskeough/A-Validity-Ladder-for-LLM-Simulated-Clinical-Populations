"""Table of the intermediate simulator panel for the Controls section (tables/tab_panel.tex).

Reads paper_brm/analysis_brm/intermediate_panel/88_confusion.csv (script 88b). Every cell is printed
from that file; nothing is typed by hand. Percentages to one decimal, as in 87.
"""
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(BASE, "paper_brm", "analysis_brm", "intermediate_panel", "88_confusion.csv")
OUT = os.path.join(BASE, "paper_brm", "manuscript", "tables", "tab_panel.tex")

LABELS = [
    ("REAL", "Real respondents"),
    ("ORACLE-INDEP", "Independent items"),
    ("SHIFTED-2.1", "Shifted, +2.1"),
    ("SHIFTED-4.7", "Shifted, +4.7"),
    ("STEEPENED-2x", "Income gap doubled"),
    ("COMPRESSED", "Compressed"),
    ("NONINVARIANT", "Non-invariant (income)"),
]
COLS = [("gate_min_pass", "Gate"), ("L1_pass", "L1"), ("L2_fail", "L2 fail"), ("L3_pass_2pt", "L3"),
        ("L4_R1_pass", "R1"), ("L4_R2_pass", "R2"), ("R4_e08_fail", "R4 fail"), ("R4_e08_unresolved", "R4 unres."),
        ("L4level_e08_pass", "L4")]


def pct(x):
    return f"{100 * float(x):.1f}"


d = pd.read_csv(IN).set_index("type")
assert set(t for t, _ in LABELS) <= set(d.index)
n = d.loc["REAL"]
lines = [
    "% Written by scripts/88c_panel_table.py from analysis_brm/intermediate_panel/88_confusion.csv. Do not edit by hand.",
    r"\begin{table}[tp]",
    r"\centering",
    r"\caption{Verdicts for Seven Simulator Types Built From Survey Respondents}",
    r"\label{tab:panel}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{l" + "r" * len(COLS) + "}",
    r"\toprule",
    "Simulator & " + " & ".join(h for _, h in COLS) + r" \\",
    r"\midrule",
]
for key, label in LABELS:
    r = d.loc[key]
    cells = []
    for c, _ in COLS:
        if c.startswith("R4") or c.startswith("L4level"):
            cells.append(pct(r[c]) + f" ({int(r['R4_e08_n'])})" if c == "R4_e08_fail" else pct(r[c]))
        else:
            cells.append(pct(r[c]))
    lines.append(label + " & " + " & ".join(cells) + r" \\ % R: analysis_brm/intermediate_panel/88_confusion.csv")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tabnote}",
    r"\textit{Note.} Percentage of readings that pass, except the columns marked fail or unresolved. "
    f"Gate, L2 and L3 are read per pseudo-model ({int(n['gate_n'])} readings per type); "
    f"L1, R1 and R2 on one pseudo-model per replicate ({int(n['L1_n'])} replicates). "
    "R4 is read at margin .08 on the number of replicates in parentheses; it gives no verdict without a general factor. "
    "L2 passes in no reading of any type; its remaining readings are unresolved. "
    "L4 is the level-4 pass (R1, R2 and R4 in both framings). Shifted types raise every persona by the stated points; "
    "the doubled gap raises low-income personas only; non-invariant makes three items independent within the low-income group. "
    "Construction, seeds and per-step results: \\texttt{paper\\_brm/analysis\\_brm/intermediate\\_panel} in the release.",
    r"\end{tabnote}",
    r"\end{table}",
]
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
