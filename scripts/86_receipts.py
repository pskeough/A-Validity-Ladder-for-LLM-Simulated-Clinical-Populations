"""
86_receipts.py: build the receipts table (Supplement S2) and check every number in the manuscript
against the file that is named as its receipt.

Sources
  paper_brm/manuscript/sections/*.tex, figures/*_note.tex, figures/*_caption.tex
      a line that carries a number ends with "% R: file[; file]" naming its receipt(s)
  paper_brm/manuscript/tables/*.tex
      the header comment lists the table's receipts ("receipts a.csv, b.csv"); every data row is
      checked against all of them
Receipt paths
  bare name            -> analysis/brm/<name>
  analysis_brm/<x>.md  -> paper_brm/analysis_brm/<x>.md
  ../<name>            -> analysis/<name>

Check
  Every number in the text part of a tagged line is looked for in its receipt files: CSV cells and
  numbers inside CSV strings, or every number in a .md/.txt/.json file. A number printed with d
  decimals matches a receipt value v when |v - x| <= 0.5 * 10^-d, or the same for 100 v (a
  proportion printed as a percentage). Integers from 0 to 12 are "weak": they read as words
  ("two framings", "level 3") and match almost anything, so they are listed but not scored.
  Years, model and instrument names, levels, rules (R1-R4), supplement and section numbers are
  removed before numbers are read.
  An unmatched number is not an error by itself: counts derived from rows ("10 of the 12
  readings") and differences computed in the text have no cell of their own. Each unmatched
  number is listed for a reader to re-derive.

Emits
  analysis/brm/86_receipts_check.csv      one row per number: file, line, number, status, where found
  analysis/brm/86_receipts_summary.txt    counts and the unmatched list
  paper_brm/manuscript/supplement/S2_receipts.csv   one row per tagged line: location, claim, receipts, scripts
  paper_brm/manuscript/supplement/S2_receipts.tex   the same as a longtable for the supplement
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANU = os.path.join(BASE, "paper_brm", "manuscript")
BRM = os.path.join(BASE, "analysis", "brm")
NOTES = os.path.join(BASE, "paper_brm", "analysis_brm")
SCRIPTS = os.path.join(BASE, "scripts")
SUPP = os.path.join(MANU, "supplement")

# Numbers with no cell of their own, re-derived by hand from the named receipt (1 Oct 2026).
# Keyed by (manuscript file, number as printed). Status "derived"; the reason is printed in the check.
DERIVED = {
    ("sections/01_introduction.tex", "28,800"): "sum of rows, corpus_v3_provenance.csv (14,400 clinical + 14,400 narrative)",
    ("sections/03_data.tex", "28,800"): "sum of rows, corpus_v3_provenance.csv",
    ("sections/08_limitations.tex", "28,800"): "sum of rows, corpus_v3_provenance.csv",
    ("sections/05_controls.tex", "24"): "REAL R4_e08_unresolved .96 x R4_e08_n 25, intermediate_panel/88_confusion.csv; "
                                        "also design: 24 personas per race group",
    ("sections/05_controls.tex", "25"): "R4_e08_n 25 (REAL, NONINVARIANT); NONINVARIANT R4_e08_fail 1.0 x 25, 88_confusion.csv",
    ("sections/05_controls.tex", "40"): "design: 24 personas per race group, 30 per cisgender sex, 40 per income band",
    ("sections/05_controls.tex", "30"): "design: draws per persona and framing",
    ("sections/05_controls.tex", "14.6"): "1 - STEEPENED-2x L3_pass_2pt .8536, intermediate_panel/88_confusion.csv",
    ("sections/05_controls.tex", "2"): "bound: max non-target fail among planted types is .0143 (NONINVARIANT L1_fail, COMPRESSED 1 - L4_R1_pass), 88_confusion.csv",
    ("sections/08_limitations.tex", "40"): "design: 24 personas per race group, 30 per cisgender sex, 40 per income band",
    ("sections/08_limitations.tex", "24"): "design: 24 personas per race group, 30 per cisgender sex, 40 per income band",
    ("sections/03_data.tex", "1,532"): "resent rows: DeepSeek 313 + GLM 1,032 (875 verify + 157 retry, 87a_resend_sources.csv) + 62 + 125",
    ("sections/03_data.tex", "1,219"): "GLM resent rows: 1,032 + 62 + 125",
    ("sections/03_data.tex", "120"): "design: 5 races x 4 gender identities x 3 incomes x 2 statuses",
    ("sections/03_data.tex", "30"): "design: draws per persona and framing",
    ("sections/03_data.tex", "18"): "inclusion rule: adults aged 18 or over",
    ("sections/03_data.tex", "36,274"): "sum of mec_weight_pos over cycles D-J, l1_nhanes_sample_flow.csv",
    ("sections/04_results.tex", "2.24"): "sqrt of the largest sigma2_r, 5.04 (GLM-4.7 clinical) = 2.2449, gate_dstudy.csv",
    ("sections/04_results.tex", "60"): "design constant, 2 framings x 30 draws",
    ("sections/06_external.tex", "37"): "two-region verdicts among the 85 published 'missing' readings: 17 m/a + 16 a/k + 4 r/m, l2_external_r3_receipts.csv",
    ("sections/04_results.tex", "90"): "interval level, design constant",
    ("sections/04_results.tex", "0.2"): "tolerance 0.2 SD, design constant",
    ("sections/04_results.tex", ".05"): "R4 sensitivity margin, design constant",
    ("sections/04_results.tex", "50"): "row count of the primary rows in 80d_sign_discordance_primary.csv",
    ("sections/04_results.tex", "30"): "personas per group by design (24 for race groups, 30 for sex/income)",
    ("sections/04_results.tex", "5.3"): "10.00 - 4.69, 85_worked_case.csv",
    ("sections/05_controls.tex", "303"): "303 data rows of 83a_receipt.csv, all ok = True",
    ("sections/05_controls.tex", "1.25"): "hypothetical tolerance named in the text",
    ("sections/06_external.tex", "61"): "Argyle t0.7_main family size, l2_external_r3.csv",
    ("sections/03_data.tex", "21"): "fixed_utc 11:21:30 in decoding_control_preregistration.json to the first ts 11:42:15 in decoding_control_raw.jsonl = 20.75 min",
}

# Derivations that apply only on a line containing the phrase (the same number appears elsewhere in the file).
MARITAL = "four other mappings of relationship status"
DERIVED_CTX = [
    ("sections/04_results.tex", "18", MARITAL, "per-model standardised verdicts differing from headline: ref_mar_all 4, ref_mar_never 6, ref_mar_drop1819 0, ref_no_marital 8, l2_verdicts.csv"),
    ("sections/04_results.tex", "112", MARITAL, "4 marital variants x 7 contrasts x 4 models, l2_verdicts.csv"),
    ("sections/04_results.tex", "16", MARITAL, "18 changes minus the 2 income changes under ref_no_marital, l2_verdicts.csv"),
]

NUM_RE = re.compile(r"(?<![\w.])([+-]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)(?![\w])")

# Tokens removed before numbers are read: names that contain digits, and references that are not claims.
STRIP_PATTERNS = [
    r"\\(?:cite[a-z]*|citeauthor|citeyear|ref|label|input|includegraphics|url)\*?(?:\[[^\]]*\])*\{[^}]*\}",
    r"GPT-4o-mini|Gemini-3-Flash|DeepSeek-V3|GLM-4\.7|GPT-3(?:\.5)?|PHQ-8|PHQ-9|GAD-7|AUDIT-C|PCL-5",
    r"gpt-4o-mini|gemini-3-flash-preview|deepseek-chat-v3|deepseek-chat|z-ai/glm-4\.7|glm-4\.7",
    r"\bR[1-4]\b",
    r"\b[Ll]evels?~?\s*\d(?:\s*(?:to|and|--|,)\s*\d)*",
    r"\b(?:Supplement|Supplements)~?\s*S\d(?:\.\d+)?(?:\s*(?:to|and|--)\s*S\d(?:\.\d+)?)*",
    r"\bS\d\b",
    r"\b(?:Section|Equation|Figure|Table)~?\S*",
    r"\b(?:19|20)\d{2}(?:--(?:19|20)?\d{2})?\b",  # years and year ranges
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    r"\b\d+(?:st|nd|rd|th)\b",
    r"\\(?:sigma|lambda|phi|mu|nu|gamma|rho|tau|chi)\b(?:\^\{?\d\}?|_\{?[^}$\s]*\}?)*",
    r"[_^]\{[^}]*\}|[_^]\d",
    r"\\[a-zA-Z]+",  # any remaining command name (its argument text is kept)
]


def strip_for_numbers(text):
    t = text.replace("--", " – ").replace("\u2212", "-")
    for p in STRIP_PATTERNS:
        t = re.sub(p, " ", t)
    t = t.replace("$", "").replace("{", " ").replace("}", " ")
    return t


def numbers_in(text):
    out = []
    for m in NUM_RE.finditer(strip_for_numbers(text)):
        sign, body = m.group(1), m.group(2)
        val = float(body.replace(",", ""))
        if sign == "-":
            val = -val
        dec = len(body.split(".")[1]) if "." in body else 0
        out.append((m.group(0).strip(), val, dec))
    return out


def split_comment(line):
    """Return (code, comment) split at the first unescaped %."""
    m = re.search(r"(?<!\\)%", line)
    if not m:
        return line, ""
    return line[: m.start()], line[m.start() + 1 :]


def resolve(name):
    name = name.strip()
    if name.startswith("analysis_brm/"):
        return os.path.join(NOTES, name[len("analysis_brm/"):])
    if name.startswith("../"):
        return os.path.normpath(os.path.join(BRM, name))
    return os.path.join(BRM, name)


_value_cache = {}


def receipt_values(path):
    """All numeric values in a receipt file, with a short location for each."""
    if path in _value_cache:
        return _value_cache[path]
    vals = []
    if not os.path.exists(path):
        _value_cache[path] = None
        return None
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else []
        for i, row in enumerate(rows[1:], start=2):
            for j, cell in enumerate(row):
                col = header[j] if j < len(header) else str(j)
                try:
                    vals.append((float(cell), f"r{i}:{col}"))
                    continue
                except ValueError:
                    pass
                for m in NUM_RE.finditer(cell.replace("\u2212", "-")):
                    v = float(m.group(2).replace(",", ""))
                    vals.append((-v if m.group(1) == "-" else v, f"r{i}:{col}"))
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if path.endswith(".json"):
            text = json.dumps(json.loads(text))
        for ln, line in enumerate(text.splitlines(), start=1):
            for m in NUM_RE.finditer(line.replace("\u2212", "-").replace("−", "-")):
                v = float(m.group(2).replace(",", ""))
                vals.append((-v if m.group(1) == "-" else v, f"line{ln}"))
    _value_cache[path] = vals
    return vals


def find(val, dec, paths):
    tol = 0.5 * 10 ** (-dec) + 1e-9
    for p in paths:
        vals = receipt_values(p)
        if not vals:
            continue
        for v, loc in vals:
            if abs(v - val) <= tol:
                return f"{os.path.basename(p)}:{loc}"
            if abs(100 * v - val) <= tol:
                return f"{os.path.basename(p)}:{loc} (x100)"
    # Signs: the text prints a gap as "+4.58" or a magnitude; accept the absolute value as a second try.
    for p in paths:
        vals = receipt_values(p) or []
        for v, loc in vals:
            if abs(abs(v) - abs(val)) <= tol or abs(abs(100 * v) - abs(val)) <= tol:
                return f"{os.path.basename(p)}:{loc} (abs)"
    return ""


# ---- which script writes each receipt -------------------------------------------------------
def build_writer_index():
    """Map receipt basename -> scripts that name it in an output line (to_csv, save, Emits, written)."""
    idx = defaultdict(set)
    pats = []
    for fn in sorted(os.listdir(SCRIPTS)):
        if not fn.endswith(".py") or fn == "86_receipts.py":
            continue
        with open(os.path.join(SCRIPTS, fn), encoding="utf-8", errors="replace") as f:
            src = f.read()
        for line in src.splitlines():
            if not re.search(r"to_csv|save\(|Emits|emits|written|write|savefig|\.csv\"\)|open\(", line):
                continue
            for m in re.finditer(r"([\w{}.\-]+\.(?:csv|txt|json|md))", line):
                name = m.group(1)
                if "{" in name:
                    rx = "^" + re.sub(r"\\\{[^}]*\\\}", ".*", re.escape(name)) + "$"
                    pats.append((re.compile(rx), fn))
                else:
                    idx[name].add(fn)
        # Docstring "Emits" blocks list names over several lines.
        em = re.search(r"Emits(.*?)(?:\n\s*\n|\"\"\")", src, re.S)
        if em:
            for m in re.finditer(r"([\w{}\-]+\.(?:csv|txt|json))", em.group(1)):
                name = m.group(1)
                if "{" in name:
                    rx = "^" + re.sub(r"\\\{[^}]*\\\}", ".*", re.escape(name)) + "$"
                    pats.append((re.compile(rx), fn))
                else:
                    idx[name].add(fn)
    return idx, pats


def writers(name, idx, pats):
    base = os.path.basename(name)
    if name.startswith("analysis_brm/"):
        return "analysis note"
    s = set(idx.get(base, set()))
    if not s:
        s = {fn for rx, fn in pats if rx.match(base)}
    if len(s) > 1:
        # Prefer the script whose number prefix matches the receipt prefix (80d_x.csv -> 80d_*.py).
        pre = base.split("_")[0]
        pref = {fn for fn in s if fn.startswith(pre + "_")}
        s = pref or s
    return "; ".join(sorted(s)) if s else "UNRESOLVED"


# ---- parse the manuscript --------------------------------------------------------------------
def tagged_lines():
    """Yield (relpath, lineno, text, [receipt names])."""
    files = []
    for sub in ("sections", "figures"):
        d = os.path.join(MANU, sub)
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".tex") and not f.startswith("fig1_ladder.tex")]
    for path in files:
        rel = os.path.relpath(path, MANU).replace("\\", "/")
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                code, comment = split_comment(line.rstrip("\n"))
                m = re.match(r"\s*R:\s*(.+)", comment)
                if m and code.strip():
                    names = [n.strip() for n in m.group(1).split(";") if n.strip()]
                    yield rel, ln, code.strip(), names
    tdir = os.path.join(MANU, "tables")
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".tex"):
            continue
        path = os.path.join(tdir, fn)
        rel = f"tables/{fn}"
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = " ".join(split_comment(l)[1] for l in lines if l.lstrip().startswith("%"))
        names = re.findall(r"[\w.\-]+\.csv", header)
        body = False
        for ln, line in enumerate(lines, start=1):
            code, comment = split_comment(line)
            if "\\midrule" in code:
                body = True
                continue
            if "\\bottomrule" in code:
                body = False
            m = re.match(r"\s*R:\s*(.+)", comment)
            row_names = [n.strip() for n in m.group(1).split(";")] if m else names
            if (body and "&" in code) or (m and code.strip()):
                yield rel, ln, code.strip(), row_names


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}")
            .replace("\\textbackslash\\{\\}", "\\textbackslash{}")
            .replace("^", "\\textasciicircum{}").replace("~", "\\textasciitilde{}")
            .replace("–", "--").replace("—", "---").replace("−", "$-$"))


def plain(code):
    """A readable version of a LaTeX line for the receipts table."""
    t = re.sub(r"\\(?:citet|citep|citeauthor|citeyear)\{([^}]*)\}", r"[\1]", code)
    t = re.sub(r"\\(?:ref)\{[^}]*\}", "#", t)
    t = re.sub(r"\\(?:vd|emph|textit|textbf)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\%", "%").replace("--", "–").replace("~", " ").replace("\\$", "$").replace("``", '"').replace("''", '"')
    t = re.sub(r"\$([^$]*)\$", r"\1", t)
    t = t.replace("\\", "")
    return t


def main():
    idx, pats = build_writer_index()
    check_rows, s2_rows = [], []
    missing = set()
    for rel, ln, code, names in tagged_lines():
        paths = [resolve(n) for n in names]
        for n, p in zip(names, paths):
            if not os.path.exists(p):
                missing.add(n)
        nums = numbers_in(code)
        s2_rows.append({
            "location": f"{rel}:{ln}",
            "claim": plain(code),
            "receipts": "; ".join(names),
            "scripts": "; ".join(writers(n, idx, pats) for n in names),
        })
        for shown, val, dec in nums:
            ctx = [note for (r, s, phrase, note) in DERIVED_CTX if r == rel and s == shown.lstrip("+") and phrase in code]
            if ctx:
                status, where = "derived", ctx[0]
            elif (rel, shown.lstrip("+")) in DERIVED:
                status, where = "derived", DERIVED[(rel, shown.lstrip("+"))]
            elif dec == 0 and 0 <= abs(val) <= 12:
                status, where = "weak", ""
            else:
                where = find(val, dec, paths)
                status = "matched" if where else "unmatched"
            check_rows.append({"location": f"{rel}:{ln}", "number": shown, "status": status,
                               "found_in": where, "receipts": "; ".join(names), "text": code[:160]})

    os.makedirs(SUPP, exist_ok=True)
    with open(os.path.join(BRM, "86_receipts_check.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(check_rows[0].keys()))
        w.writeheader()
        w.writerows(check_rows)
    with open(os.path.join(SUPP, "S2_receipts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(s2_rows[0].keys()))
        w.writeheader()
        w.writerows(s2_rows)

    with open(os.path.join(SUPP, "S2_receipts.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% Written by scripts/86_receipts.py. Do not edit by hand.\n")
        f.write("\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{7.2cm}"
                ">{\\raggedright\\arraybackslash}p{3.4cm}>{\\raggedright\\arraybackslash}p{2.8cm}@{}}\n")
        f.write("\\caption{Receipts: Every Number in the Article and the File That Prints It}\\label{tab:s2}\\\\\n")
        f.write("\\toprule\nLocation & Claim & Receipt file & Script \\\\\n\\midrule\n\\endfirsthead\n")
        f.write("\\toprule\nLocation & Claim & Receipt file & Script \\\\\n\\midrule\n\\endhead\n")
        for r in s2_rows:
            f.write(" & ".join(tex_escape(x) for x in (r["location"], r["claim"], r["receipts"], r["scripts"])) + " \\\\\n")
        f.write("\\bottomrule\n\\end{longtable}\n")

    counts = defaultdict(int)
    for r in check_rows:
        counts[r["status"]] += 1
    unresolved = sorted({n for r in s2_rows for n, s in zip(r["receipts"].split("; "), r["scripts"].split("; ")) if s == "UNRESOLVED"})
    lines = [
        f"tagged lines: {len(s2_rows)}",
        f"numbers: {len(check_rows)}  matched {counts['matched']}  derived {counts['derived']}  "
        f"unmatched {counts['unmatched']}  weak {counts['weak']}",
        f"receipt files missing: {len(missing)} {sorted(missing)}",
        f"receipts with no writing script found: {len(unresolved)} {unresolved}",
        "",
        "UNMATCHED (re-derive each by hand):",
    ]
    for r in check_rows:
        if r["status"] == "unmatched":
            lines.append(f"  {r['location']}  {r['number']:>10}  [{r['receipts']}]  {r['text'][:110]}")
    report = "\n".join(lines)
    with open(os.path.join(BRM, "86_receipts_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    sys.exit(main())
