"""
Supplement tables for the BRM manuscript (Supplement S1 and S3 to S6).

Every number in the supplement's tables, and every number its prose quotes through a macro, is
printed here from a receipt file. Nothing in the supplement's table cells is typed by hand. S2 is
written by scripts/86_receipts.py and is only \\input by the supplement.

Sources
    data/model_outputs_v3.csv                         corpus rows and their row_source labels
    generation/main.py, generation/narrative_main.py  released generation code
    ../original_generation_2025-12-31/main.py         recovered December code (read only, if present)
    ../original_generation_run2_2026-01/main.py       recovered January code (read only, if present)
    generation/recovery/verify_run1_refusals.py, generation/retry_failed.py,
    generation/recovery/slow_recovery.py, last_mile_recovery_opt.py   January resend scripts
    analysis/brm/87a_resend_sources.csv                                 script 87a
    generation/registries/identities_registry*.json
    analysis/prompt_control_design.json, analysis/decoding_control_design.json
    scripts/55_decoding_control_run.py                decoding-control system prompt and template
    analysis/brm/gate_dstudy.csv, gate_license_models.csv, gate_december_compare.csv
    analysis/brm/l1_summary.csv, l1_personfit_main.csv
    analysis/brm/l2_verdicts.csv
    analysis/brm/80c_l3_results.csv, 80d_headline.csv, 80d_model_verdicts.csv, 80d_december.csv,
        80d_december_summary.csv, 80d_income_anchor_permodel.csv, 80d_era.csv, 80d_era_shift_permodel.csv
    analysis/brm/l4_summary.csv, l4_summary_dec.csv, l4_invariance.csv
    analysis/brm/83c_l1.csv, 83c_l2.csv, 83c_l2_coverage.csv, 83c_l3.csv, 83c_l4.csv,
        83f_l2_parametric_power.csv
    paper_brm/external/results/opinionqa/level3_groups.csv
    paper_brm/external/results/argyle/level3_groups.csv

Rounding
    2 decimals for ratios and PHQ-8 points, 1 for percentages, 3 for dependability, congruence,
    RMSEA_D and coverage (leading zero dropped for quantities bounded by 1).

Emits
    analysis/brm/87_row_sources.csv
    paper_brm/manuscript/supplement/tables/*.tex   one fragment per table, each headed by the
                                                   receipts it reads
    paper_brm/manuscript/supplement/tables/supp_macros.tex   numbers the supplement prose quotes
"""
import difflib
import glob
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
MANU = os.path.join(BASE, "paper_brm", "manuscript")
BRM = os.path.join(BASE, "analysis", "brm")
ANALYSIS = os.path.join(BASE, "analysis")
SCRIPTS = os.path.join(BASE, "scripts")
GEN = os.path.join(BASE, "generation")
EXT = os.path.join(BASE, "paper_brm", "external", "results")
SUPP = os.path.join(MANU, "supplement")
TABLES = os.path.join(SUPP, "tables")

ORIG_CLIN = os.path.join(ROOT, "original_generation_2025-12-31", "main.py")
ORIG_NARR = os.path.join(ROOT, "original_generation_run2_2026-01", "main.py")
REL_CLIN = os.path.join(GEN, "main.py")
REL_NARR = os.path.join(GEN, "narrative_main.py")

MODEL_NAMES = {
    "deepseek/deepseek-chat-v3": "DeepSeek-V3", "deepseek-chat-v3": "DeepSeek-V3",
    "google/gemini-3-flash-preview": "Gemini-3-Flash", "gemini-3-flash-preview": "Gemini-3-Flash",
    "openai/gpt-4o-mini": "GPT-4o-mini", "gpt-4o-mini": "GPT-4o-mini",
    "z-ai/glm-4.7": "GLM-4.7", "glm-4.7": "GLM-4.7",
    "pooled": "Pooled",
}
MODEL_ORDER = ["DeepSeek-V3", "Gemini-3-Flash", "GPT-4o-mini", "GLM-4.7", "Pooled"]
GROUPS = ["Overall", "White", "Black", "Asian", "Hispanic", "Men", "Women", "Low", "Middle", "High"]
CONTRASTS = ["Black minus White", "Hispanic minus White", "Asian minus White", "Women minus Men",
             "Low minus High SES", "Middle minus High SES", "Low minus Middle SES"]

MACROS = {}


# ---------------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------------
def isnan(x):
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


def neg(s):
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def f2(x):
    """Ratios and PHQ-8 points."""
    if isnan(x):
        return "--"
    s = f"{float(x):.2f}"
    if s == "-0.00":
        s = "0.00"
    return neg(s)


def f3(x):
    """Dependability, congruence, RMSEA_D, coverage: three decimals, no leading zero."""
    if isnan(x):
        return "--"
    s = f"{float(x):.3f}"
    if s == "-0.000":
        s = "0.000"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return neg(s)


def pc(x):
    """Proportion printed as a percentage, one decimal."""
    if isnan(x):
        return "--"
    return neg(f"{100 * float(x):.1f}")


def p1(x):
    """A value already on the percentage scale, one decimal."""
    if isnan(x):
        return "--"
    s = f"{float(x):.1f}"
    if s == "-0.0":
        s = "0.0"
    return neg(s)


def ki(x):
    if isnan(x):
        return "--"
    return f"{int(round(float(x))):,}"


def ci(lo, hi, f=f2):
    if isnan(lo) or isnan(hi):
        return "--"
    return f"[{f(lo)}, {f(hi)}]"


def gate_k(s2_r, outcome, which):
    """Draws for the frozen gate (LADDER_SPEC.md): SE(k) <= 0.25 (min) or 0.125 (rec) reference SD
    on the PHQ-8 total, reference SD from 80a (NHANES 2005-2018, all adults). The rule is defined on
    the total; the >= 10 indicator keeps its stored proportion tolerances (0.10 and 0.05)."""
    if outcome == "PHQ-8 total":
        b = read(brm("80a_tolerance_basis.csv"))
        sd = float(b[(b.window == "2005-2018") & (b.population == "all adults 18+")].sd.iloc[0])
        tol = (0.25 if which == "min" else 0.125) * sd
    else:
        tol = 0.10 if which == "min" else 0.05
    return max(1, int(np.ceil(float(s2_r) / tol ** 2 - 1e-12)))


def P(w):
    """Ragged-right paragraph column."""
    return f">{{\\raggedright\\arraybackslash}}p{{{w}}}"


def R(w):
    """Ragged-left paragraph column, for numeric columns with long headers."""
    return f">{{\\raggedleft\\arraybackslash}}p{{{w}}}"


def texb(s):
    """tex() with line-break points after underscores and hyphens, for long labels."""
    return tex(s).replace("\\_", "\\_\\allowbreak{}").replace("-", "-\\allowbreak{}")


def short_l1(v):
    """Level-1 verdict without the 'fail:' prefix and the gloss on 'compressed'."""
    return v.replace("fail: ", "").replace(" (both tails in deficit)", "")


def yn(x):
    if isnan(x):
        return "--"
    if isinstance(x, str):
        x = x.strip().lower() in ("true", "1", "1.0")
    return "Yes" if bool(x) else "No"


def tex(s):
    if isnan(s):
        return "--"
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), (">=", r"$\ge$"), ("<=", r"$\le$"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}")]:
        s = s.replace(a, b)
    return s


def mname(x):
    return MODEL_NAMES.get(str(x), str(x))


def rel(path):
    """Receipt path relative to the repository base, for fragment headers."""
    return os.path.relpath(path, BASE).replace("\\", "/")


def brm(name):
    return os.path.join(BRM, name)


def read(path, **kw):
    return pd.read_csv(path, **kw)


def mean_name(m):
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else 99


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------
def header(receipts):
    names = "; ".join(rel(r) for r in receipts)
    return ("% Written by scripts/87_supplement_tables.py. Do not edit by hand.\n"
            f"% Receipts: {names}\n")


def write(name, text):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def longtable(name, receipts, caption, label, colspec, head, rows, note=None, size="footnotesize",
              tabcolsep="4pt"):
    """head: list of header lines (each ending without \\\\); rows: list of lists, or the strings
    'MID' (a \\midrule) and 'ADD' (\\addlinespace)."""
    last = re.sub(r"\\cmidrule\([lr]*\)\{[^}]*\}", "", head[-1])
    ncol = len(re.split(r"(?<!\\)&", last))
    out = [header(receipts)]
    out.append("\\begingroup\n\\singlespacing\n\\" + size + "\n")
    out.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n")
    out.append(f"\\begin{{longtable}}{{{colspec}}}\n")
    out.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
    out.append("\\toprule\n")
    out.extend(h + " \\\\\n" for h in head)
    out.append("\\midrule\n\\endfirsthead\n")
    out.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{Table \\thetable, continued}}}} \\\\\n")
    out.append("\\toprule\n")
    out.extend(h + " \\\\\n" for h in head)
    out.append("\\midrule\n\\endhead\n")
    out.append("\\bottomrule\n\\endlastfoot\n")
    for r in rows:
        if r == "MID":
            out.append("\\midrule\n")
        elif r == "ADD":
            out.append("\\addlinespace\n")
        else:
            out.append(" & ".join(r) + " \\\\\n")
    out.append("\\end{longtable}\n")
    if note:
        out.append("\\vspace{-6pt}\n\\noindent\\textit{Note.} " + note + "\n\\par\n")
    out.append("\\endgroup\n")
    write(name, "".join(out))


def verbatim_block(lines, firstline=None, title=None):
    opts = "breaklines,breakanywhere,fontsize=\\footnotesize,frame=single,framesep=4pt"
    if firstline is not None:
        opts += f",numbers=left,numbersep=6pt,firstnumber={firstline}"
    out = []
    if title:
        out.append(f"\\noindent {title}\n")
    out.append(f"\\begin{{Verbatim}}[{opts}]\n")
    out.extend(l.rstrip("\n").replace("\t", "    ") + "\n" for l in lines)
    out.append("\\end{Verbatim}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# Code extraction for S1
# ---------------------------------------------------------------------------------------------
def readlines(path):
    with open(path, encoding="utf-8") as fh:
        return fh.readlines()


def find_line(lines, pattern, start=0):
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i
    raise ValueError(f"pattern not found: {pattern}")


def span(lines, start_pat, end_pat, start=0, include_end=True):
    """0-based inclusive span from the first line matching start_pat to the next line matching
    end_pat."""
    a = find_line(lines, start_pat, start)
    b = find_line(lines, end_pat, a + (0 if include_end else 1))
    if not include_end:
        b -= 1
    return a, b


def excerpt(lines, a, b):
    return lines[a:b + 1]


def triple_string(lines, a):
    """The content of the triple-quoted string that opens on line a."""
    text = "".join(lines[a:])
    m = re.search(r'f?"""(.*?)"""', text, re.S)
    return m.group(1)


def s1():
    receipts = [REL_CLIN, REL_NARR]
    clin_path = ORIG_CLIN if os.path.exists(ORIG_CLIN) else REL_CLIN
    narr_path = ORIG_NARR if os.path.exists(ORIG_NARR) else REL_NARR
    if clin_path != REL_CLIN:
        receipts.insert(0, clin_path)
    if narr_path != REL_NARR:
        receipts.insert(1, narr_path)
    C = readlines(clin_path)
    N = readlines(narr_path)
    RC = readlines(REL_CLIN)
    RN = readlines(REL_NARR)

    # Spans in the recovered December file.
    cfg = span(C, r"^MAX_RETRIES\s*=", r"^SEMAPHORE_LIMIT\s*=")
    api = span(C, r"chat\.completions\.create\(", r"return response\.choices")
    api = (api[0] - 1, api[1])  # include the assignment line opening the call
    sysc = span(C, r'return """ROLE:', r'^\}"""')
    usrc = span(C, r'user_prompt = f"""PROFILE_ID', r'Standardized Clinical Battery\."""')
    retc = span(C, r"for attempt in range\(MAX_RETRIES \+ 1\)", r"# Process Results", include_end=False)
    while not C[retc[1]].strip():
        retc = (retc[0], retc[1] - 1)
    novel_c = find_line(C, r"fictional character for a novel")
    refus_c = find_line(C, r'"cannot" in lower_resp')

    sysn = span(N, r'return """ROLE:', r'^\}"""')
    usrn = span(N, r'user_prompt = f"""PROFILE_ID', r'Standardized Clinical Battery\."""')
    retn = span(N, r"for attempt in range\(MAX_RETRIES \+ 1\)", r"# Process Results", include_end=False)
    while not N[retn[1]].strip():
        retn = (retn[0], retn[1] - 1)
    novel_n = find_line(N, r"fictional character for a novel")
    refn = span(N, r"^def is_refusal_response", r"return any\(indicator")
    parse_n = (find_line(N, r"^def robust_json_parse"), refn[0] - 1)
    while not N[parse_n[1]].strip():
        parse_n = (parse_n[0], parse_n[1] - 1)
    MACROS["SOneParseNarr"] = f"{parse_n[0] + 1}--{parse_n[1] + 1}"
    maxr_n = find_line(N, r"^MAX_RETRIES\s*=")

    # Checks: the two system prompts are identical, the released clinical copy carries the same
    # prompt text, and the released narrative copy is the January file.
    sys_clin = triple_string(C, sysc[0])
    sys_narr = triple_string(N, sysn[0])
    assert sys_clin == sys_narr, "clinical and narrative system prompts differ"
    rsys = find_line(RC, r'return """ROLE:')
    assert triple_string(RC, rsys) == sys_clin, "released clinical prompt differs from December file"
    assert "".join(RN) == "".join(N), "released narrative_main.py differs from the January file"
    rusr = find_line(RC, r'user_prompt = f"""PROFILE_ID')
    assert triple_string(RC, rusr) == triple_string(C, usrc[0])
    rnov = find_line(RC, r"fictional character for a novel")
    assert RC[rnov].strip() == C[novel_c].strip()
    assert C[cfg[0]].strip() == N[maxr_n].strip(), "MAX_RETRIES differs between runs"

    def L(a):
        return str(a + 1)

    MACROS.update({
        "SOneClinFile": tex(rel(clin_path)), "SOneNarrFile": tex(rel(narr_path)),
        "SOneSysClin": f"{L(sysc[0])}--{L(sysc[1])}", "SOneSysNarr": f"{L(sysn[0])}--{L(sysn[1])}",
        "SOneUsrClin": f"{L(usrc[0])}--{L(usrc[1])}", "SOneUsrNarr": f"{L(usrn[0])}--{L(usrn[1])}",
        "SOneRetClin": f"{L(retc[0])}--{L(retc[1])}", "SOneRetNarr": f"{L(retn[0])}--{L(retn[1])}",
        "SOneNovelClin": L(novel_c), "SOneNovelNarr": L(novel_n), "SOneRefusClin": L(refus_c),
        "SOneRefusNarr": f"{L(refn[0])}--{L(refn[1])}",
        "SOneApiClin": f"{L(api[0])}--{L(api[1])}", "SOneCfgClin": f"{L(cfg[0])}--{L(cfg[1])}",
        "SOneRelSys": L(rsys), "SOneRelUsr": L(rusr), "SOneRelNovel": L(rnov),
        "SOneMaxRetries": C[cfg[0]].split("=")[1].split("#")[0].strip(),
        "SOneSysLines": str(len(sys_clin.split("\n"))),
    })

    # Registry example: the first persona in each registry.
    reg_c = json.load(open(os.path.join(GEN, "registries", "identities_registry.json"), encoding="utf-8"))
    reg_n = json.load(open(os.path.join(GEN, "registries", "identities_registry_narrative.json"),
                           encoding="utf-8"))
    ex_c = {k: reg_c[0][k] for k in ("id", "race", "gender", "ses", "relationship")}
    ex_n = {k: reg_n[0][k] for k in ("id", "injection_text")}
    assert ex_c["id"] == ex_n["id"]
    MACROS["SOneRegistryN"] = str(len(reg_c))

    out = [header(receipts + [os.path.join(GEN, "registries", "identities_registry.json"),
                              os.path.join(GEN, "registries", "identities_registry_narrative.json")])]
    out.append("\\subsection*{S1.1 System Prompt, Both Framings}\n")
    out.append(verbatim_block(excerpt(C, *sysc), sysc[0] + 1,
                              f"\\texttt{{{tex(rel(clin_path))}}}, lines {L(sysc[0])}--{L(sysc[1])}."))
    out.append("\\subsection*{S1.2 User Prompt, Clinical Framing}\n")
    out.append(verbatim_block(excerpt(C, *usrc), usrc[0] + 1,
                              f"\\texttt{{{tex(rel(clin_path))}}}, lines {L(usrc[0])}--{L(usrc[1])}."))
    out.append(verbatim_block([json.dumps(ex_c, indent=1, ensure_ascii=False)], None,
                              "Registry entry that fills the fields, first persona of "
                              "\\texttt{generation/registries/identities\\_registry.json}."))
    out.append("\\subsection*{S1.3 User Prompt, Narrative Framing}\n")
    out.append(verbatim_block(excerpt(N, *usrn), usrn[0] + 1,
                              f"\\texttt{{{tex(rel(narr_path))}}}, lines {L(usrn[0])}--{L(usrn[1])}."))
    out.append(verbatim_block([json.dumps(ex_n, indent=1, ensure_ascii=False)], None,
                              "Registry entry for the same persona, "
                              "\\texttt{generation/registries/identities\\_registry\\_narrative.json}."))
    out.append("\\subsection*{S1.4 API Call and Retry Settings, Clinical Run}\n")
    out.append(verbatim_block(excerpt(C, *cfg), cfg[0] + 1,
                              f"\\texttt{{{tex(rel(clin_path))}}}, lines {L(cfg[0])}--{L(cfg[1])}."))
    out.append(verbatim_block(excerpt(C, *api), api[0] + 1,
                              f"\\texttt{{{tex(rel(clin_path))}}}, lines {L(api[0])}--{L(api[1])}."))
    out.append("\\subsection*{S1.5 Retry Loop, Clinical Run}\n")
    out.append(verbatim_block(excerpt(C, *retc), retc[0] + 1,
                              f"\\texttt{{{tex(rel(clin_path))}}}, lines {L(retc[0])}--{L(retc[1])}."))
    out.append("\\subsection*{S1.6 Retry Loop and Refusal Check, Narrative Run}\n")
    out.append(verbatim_block(excerpt(N, *refn), refn[0] + 1,
                              f"\\texttt{{{tex(rel(narr_path))}}}, lines {L(refn[0])}--{L(refn[1])}."))
    out.append(verbatim_block(excerpt(N, *retn), retn[0] + 1,
                              f"\\texttt{{{tex(rel(narr_path))}}}, lines {L(retn[0])}--{L(retn[1])}."))
    write("S1_corpus_prompts.tex", "".join(out))

    # Resend prompts, in the order the scripts ran in January.
    rec_files = ["recovery/verify_run1_refusals.py", "retry_failed.py",
                 "recovery/slow_recovery.py", "recovery/last_mile_recovery_opt.py"]
    out = [header([os.path.join(GEN, f) for f in rec_files])]
    for f in rec_files:
        R = readlines(os.path.join(GEN, f))
        a = find_line(R, r'system_prompt = """ROLE:')
        b = find_line(R, r'^\}"""', a)
        c = find_line(R, r'user_prompt = f"""PROFILE_ID', b if f != "retry_failed.py" else 0)
        d = find_line(R, r'"""\s*$', c + 1)
        sub = os.path.basename(f).replace(".py", "").replace("_", "").replace("1", "one")
        MACROS[f"SOneRec{sub.title()}Sys"] = f"{a + 1}--{b + 1}"
        MACROS[f"SOneRec{sub.title()}Usr"] = f"{c + 1}--{d + 1}"
        rs = triple_string(R, a)
        MACROS[f"SOneRec{sub.title()}SameSys"] = "is identical to" if rs == sys_clin else "differs from"
        # Supplement S1.8 states that only the last-mile script changed the system prompt.
        assert (rs == sys_clin) == (f != "recovery/last_mile_recovery_opt.py"), f"system prompt check: {f}"
        out.append(verbatim_block(excerpt(R, a, b), a + 1,
                                  f"\\texttt{{generation/{tex(f)}}}, lines {a + 1}--{b + 1}."))
        out.append(verbatim_block(excerpt(R, c, d), c + 1,
                                  f"\\texttt{{generation/{tex(f)}}}, lines {c + 1}--{d + 1}."))
    write("S1_recovery_prompts.tex", "".join(out))

    # Control prompts.
    pcd_path = os.path.join(ANALYSIS, "prompt_control_design.json")
    dcd_path = os.path.join(ANALYSIS, "decoding_control_design.json")
    dec_script = os.path.join(SCRIPTS, "55_decoding_control_run.py")
    pcd = json.load(open(pcd_path, encoding="utf-8"))
    dcd = json.load(open(dcd_path, encoding="utf-8"))
    D = readlines(dec_script)
    dsys = span(D, r'^SYSTEM_PROMPT = """ROLE:', r'^\}"""')
    dusr = span(D, r'^USER_TEMPLATE = """', r'Standardized Clinical Battery\."""')
    dmax = find_line(D, r"^MAX_TOKENS\s*=")
    dpin = find_line(D, r"^PIN_PROVIDER\s*=")
    dec_sys = triple_string(D, dsys[0])

    out = [header([pcd_path, dcd_path, dec_script, clin_path])]
    out.append("\\subsection*{S1.9 Prompt Control: System Prompts of the Three Arms}\n")
    for arm in pcd["arms"]:
        out.append(verbatim_block(arm["system_prompt"].split("\n"), None,
                                  f"Arm \\texttt{{{tex(arm['name'])}}}: {tex(arm['description'])}. "
                                  f"Decoding parameters: \\texttt{{{tex(json.dumps(arm['params']))}}}."))
    out.append(verbatim_block(pcd["user_template"].split("\n"), None,
                              "User template, held identical across arms."))
    out.append("\\subsection*{S1.10 Decoding Control: Prompt and Arms}\n")
    out.append(verbatim_block(excerpt(D, *dsys), dsys[0] + 1,
                              f"\\texttt{{scripts/55\\_decoding\\_control\\_run.py}}, lines "
                              f"{dsys[0] + 1}--{dsys[1] + 1}."))
    out.append(verbatim_block(excerpt(D, *dusr), dusr[0] + 1,
                              f"\\texttt{{scripts/55\\_decoding\\_control\\_run.py}}, lines "
                              f"{dusr[0] + 1}--{dusr[1] + 1}."))
    out.append(verbatim_block([D[dmax], D[dpin]], None,
                              f"\\texttt{{scripts/55\\_decoding\\_control\\_run.py}}, lines {dmax + 1} and "
                              f"{dpin + 1}."))
    arms = "; ".join(f"\\texttt{{{tex(a['name'])}}} \\texttt{{{tex(json.dumps(a['params']))}}}"
                     for a in dcd["arms"])
    out.append(f"\\noindent Decoding arms in \\texttt{{analysis/decoding\\_control\\_design.json}}: {arms}.\n\n")
    write("S1_control_prompts.tex", "".join(out))

    # Line-by-line difference between each control system prompt and the corpus system prompt.
    corpus = sys_clin.split("\n")
    rows = []
    variants = [(f"Prompt control, \\texttt{{{tex(a['name'])}}}", a["system_prompt"]) for a in pcd["arms"]]
    variants.append(("Decoding control (both arms)", dec_sys))
    n_diff = {}
    for lab, text in variants:
        lines = text.split("\n")
        sm = difflib.SequenceMatcher(a=corpus, b=lines, autojunk=False)
        k = 0
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op == "equal":
                continue
            left = corpus[a0:a1] or [""]
            right = lines[b0:b1] or [""]
            for j in range(max(len(left), len(right))):
                lt = left[j] if j < len(left) else ""
                rt = right[j] if j < len(right) else ""
                ln = str(a0 + j + 1) if j < len(left) and left[j] else "--"
                rows.append([lab if k == 0 else "", ln,
                             "\\texttt{" + tex(lt) + "}" if lt else "(absent)",
                             "\\texttt{" + tex(rt) + "}" if rt else "(absent)"])
                k += 1
        n_diff[lab] = k
        rows.append("MID")
    rows = rows[:-1]
    longtable("S1_control_diff.tex", [pcd_path, dec_script, clin_path],
              "Lines of the Control System Prompts That Differ From the Corpus System Prompt",
              "tab:s1diff",
              ">{\\raggedright\\arraybackslash}p{2.6cm}r>{\\raggedright\\arraybackslash}p{5.9cm}"
              ">{\\raggedright\\arraybackslash}p{5.9cm}",
              ["Control prompt & Line & Corpus prompt & Control prompt"], rows,
              "Line numbers count lines of the corpus system prompt (S1.1). Lines not listed are "
              "identical. Computed with Python \\texttt{difflib} on the prompt strings.")
    MACROS["SOneDecDiffLines"] = str(n_diff["Decoding control (both arms)"])


# ---------------------------------------------------------------------------------------------
# S3: corpus provenance and sensitivity analyses
# ---------------------------------------------------------------------------------------------
def s3_row_sources():
    src = os.path.join(BASE, "data", "model_outputs_v3.csv")
    d = read(src, usecols=["prompt_condition", "model", "row_source", "narr_source", "phq8_valid"],
             low_memory=False)
    g = (d.groupby(["prompt_condition", "model", "row_source", "narr_source"], dropna=False)
         .agg(rows=("model", "size"), valid_phq8=("phq8_valid", "sum")).reset_index())
    g["valid_phq8"] = g["valid_phq8"].astype(int)
    g.to_csv(brm("87_row_sources.csv"), index=False)
    # Cross-check against the corpus provenance receipt written by script 76.
    prov = read(brm("corpus_v3_provenance.csv"))
    chk = g.groupby(["prompt_condition", "model", "row_source"], as_index=False)["rows"].sum()
    m = prov.merge(chk, on=["prompt_condition", "model", "row_source"], suffixes=("_76", "_87"))
    assert len(m) == len(prov) and (m.rows_76 == m.rows_87).all(), "row counts disagree with script 76"

    clin = g[g.prompt_condition == "clinical"]
    narr = g[g.prompt_condition == "narrative"]
    # Which January script produced each in-place row (script 87a).
    rs_src = read(brm("87a_resend_sources.csv"))
    assert rs_src.rows.sum() == clin.loc[clin.row_source == "recovery_inplace", "rows"].sum(), \
        "87a does not cover every in-place row"

    def inplace_desc(model):
        parts = [f"\\texttt{{{texb(r.script)}}} ({ki(r.rows)})" for r in
                 rs_src[rs_src.model == model].sort_values("rows", ascending=False).itertuples()]
        return "Resent in January: " + ", ".join(parts)

    def n(frame, **kw):
        sel = frame
        for k, v in kw.items():
            sel = sel[sel[k] == v]
        return int(sel.rows.sum())

    MACROS.update({
        "RowsTotal": ki(len(d)), "RowsValid": ki(int(d.phq8_valid.sum())),
        "RowsInplace": ki(n(clin, row_source="recovery_inplace")),
        "RowsInplaceDS": ki(n(clin, row_source="recovery_inplace", model="deepseek/deepseek-chat-v3")),
        "RowsInplaceGLM": ki(n(clin, row_source="recovery_inplace", model="z-ai/glm-4.7")),
        "RowsSlow": ki(n(clin, row_source="recovery_slow")),
        "RowsLastMile": ki(n(clin, row_source="recovery_last_mile")),
        "RowsRestored": ki(n(clin, row_source="dec28_restored")),
        "RowsNotDecember": ki(n(clin, row_source="recovery_inplace") + n(clin, row_source="recovery_slow")
                              + n(clin, row_source="recovery_last_mile") + n(clin, row_source="dec28_restored")),
        "RowsRecovered": ki(n(clin, row_source="recovery_inplace") + n(clin, row_source="recovery_slow")
                            + n(clin, row_source="recovery_last_mile")),
        "RowsResentVerify": ki(int(rs_src.loc[rs_src.script == "verify_run1_refusals.py", "rows"].sum())),
        "RowsResentRetry": ki(int(rs_src.loc[rs_src.script == "retry_failed.py", "rows"].sum())),
        "RowsNarrRerun": ki(n(narr, narr_source="narr_rerun_same_script")),
        "RowsNarrRerunGLM": ki(n(narr, narr_source="narr_rerun_same_script", model="z-ai/glm-4.7")),
        "RowsNarrRerunDS": ki(n(narr, narr_source="narr_rerun_same_script", model="deepseek/deepseek-chat-v3")),
    })

    desc = {
        "dec28_main": "December run, own output",
        "dec28_restored": "December run, restored from backup",
        "recovery_slow": "Resent in January: \\texttt{slow\\_recovery.py}",
        "recovery_last_mile": "Resent in January: \\texttt{last\\_mile\\_recovery\\_opt.py}",
        "narrative": "Narrative run",
    }
    ndesc = {"narr_main": "first pass", "narr_rerun_same_script": "rerun, same script"}
    rows = []
    for cond in ["clinical", "narrative"]:
        sub = g[g.prompt_condition == cond].copy()
        sub["mo"] = sub.model.map(lambda x: mean_name(mname(x)))
        sub = sub.sort_values(["mo", "row_source", "narr_source"])
        first = True
        for _, r in sub.iterrows():
            rows.append([cond.capitalize() if first else "", mname(r.model),
                         f"\\texttt{{{texb(r.row_source)}}}",
                         "--" if isnan(r.narr_source) else f"\\texttt{{{texb(r.narr_source)}}}",
                         (inplace_desc(r.model) if r.row_source == "recovery_inplace" else desc.get(r.row_source, ""))
                         + ("" if isnan(r.narr_source) else ", " + ndesc.get(r.narr_source, "")),
                         ki(r.rows), ki(r.valid_phq8)])
            first = False
        rows.append("MID")
    rows.append(["Total", "", "", "", "", ki(g.rows.sum()), ki(g.valid_phq8.sum())])
    longtable("S3_row_sources.tex", [src, brm("87_row_sources.csv"), brm("corpus_v3_provenance.csv")],
              "Corpus Rows by Framing, Model and Source", "tab:s3rows",
              "ll" + P("2.7cm") + P("2.7cm") + P("3.6cm") + "r" + R("1.2cm"),
              ["Framing & Model & \\texttt{row\\_source} & \\texttt{narr\\_source} & Description & Rows & Valid PHQ-8"],
              rows,
              "Counts from \\texttt{data/model\\_outputs\\_v3.csv}, grouped by script 87 and checked "
              "against \\texttt{corpus\\_v3\\_provenance.csv} (script 76).", size="scriptsize", tabcolsep="3pt")


def s3_december():
    # Gate.
    gc = read(brm("gate_december_compare.csv"))
    rows = []
    flips = 0
    cross30 = 0
    for c in ("min", "rec"):
        for s in ("full", "dec"):
            gc[f"k{c}_{s}"] = [gate_k(r[f"s2_e_{s}"], r.outcome, c) for _, r in gc.iterrows()]
    # The stored k at 1.0 / 0.5 points equals the frozen k at 0.98 / 0.49 on every total row.
    t = gc[gc.outcome == "PHQ-8 total"]
    assert (t.kmin_full == t.k_se_tol2_full).all() and (t.krec_full == t.k_se_tol1_full).all()
    for _, r in gc.iterrows():
        for a, b in [(r.kmin_full, r.kmin_dec), (r.krec_full, r.krec_dec)]:
            cross30 += int((a <= 30) != (b <= 30))
    MACROS["DecGateCrossThirty"] = str(cross30)
    MACROS["DecGateModels"] = ", ".join(sorted({f"{r.model} {r.framing}" for _, r in gc.iterrows()
                                                if (r.kmin_full, r.krec_full) != (r.kmin_dec, r.krec_dec)}))
    for oc in ["PHQ-8 total", "PHQ-8 >= 10"]:
        sub = gc[gc.outcome == oc]
        sub = sub.assign(mo=sub.model.map(mean_name)).sort_values(["mo", "framing"])
        first = True
        for _, r in sub.iterrows():
            kmin_f, kmin_d, krec_f, krec_d = r.kmin_full, r.kmin_dec, r.krec_full, r.krec_dec
            flips += int(kmin_f != kmin_d) + int(krec_f != krec_d)
            rows.append([tex(oc) if first else "", r.model, r.framing.capitalize(),
                         ki(r.n_persona_full), ki(r.n_persona_dec),
                         f3(r.phi_1_full), f3(r.phi_1_dec), ki(kmin_f), ki(kmin_d), ki(krec_f), ki(krec_d)])
            first = False
        rows.append("MID")
    rows = rows[:-1]
    MACROS["DecGateKChanges"] = str(flips)
    longtable("S3_dec_gate.tex", [brm("gate_december_compare.csv")],
              "Gate on the Full Corpus and on December Rows Only", "tab:s3decgate",
              "lllrrrrrrrr",
              [" & & & \\multicolumn{2}{c}{Personas} & \\multicolumn{2}{c}{$\\phi(1)$} & "
               "\\multicolumn{2}{c}{Minimum $k$} & \\multicolumn{2}{c}{Recommended $k$}",
               "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}"
               "Outcome & Model & Framing & Full & Dec. & Full & Dec. & Full & Dec. & Full & Dec."],
              rows,
              "Dec.\\ = December rows only (\\texttt{dec28\\_main}, \\texttt{dec28\\_restored} and all "
              "narrative rows). Minimum: the $k$ for SE $\\le$ 0.25 reference SD (0.98 PHQ-8 points). "
              "Recommended: the $k$ for SE $\\le$ 0.125 reference SD (0.49 points). Indicator rows: SE "
              "$\\le$ 0.10 and 0.05 in proportion.")

    # Level 1.
    pf = read(brm("l1_personfit_main.csv"))
    base = pf[(pf.reference == "2005_2018 weighted GRM") & (pf.subset == "all")]
    full = base[base["sample"] == "full"].set_index(["model", "framing"])
    dec = base[base["sample"] == "dec_only"].set_index(["model", "framing"])
    rows = []
    changes = 0
    keys = sorted(full.index, key=lambda t: (mean_name(mname(t[0])), t[1]))
    for m, fr in keys:
        r = full.loc[(m, fr)]
        q = dec.loc[(m, fr)]
        changes += int(r.verdict != q.verdict and m != "pooled")
        rows.append([mname(m), fr.capitalize(),
                     f2(r.misfit_ratio), f2(q.misfit_ratio), f2(r.overfit_ratio), f2(q.overfit_ratio),
                     tex(short_l1(r.verdict)), tex(short_l1(q.verdict))])
    MACROS["DecLOneChanges"] = str(changes)
    longtable("S3_dec_l1.tex", [brm("l1_personfit_main.csv")],
              "Level 1 on the Full Corpus and on December Rows Only", "tab:s3decl1",
              "llrrrr" + P("2.8cm") + P("2.8cm"),
              [" & & \\multicolumn{2}{c}{Misfit ratio} & \\multicolumn{2}{c}{Overfit ratio} & & ",
               "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}"
               "Model & Framing & Full & Dec. & Full & Dec. & Verdict, full & Verdict, Dec."],
              rows,
              "Reference: NHANES 2005--2018, weighted graded response model; all personas. Every verdict "
              "is a fail; the verdict column gives the failure reading, where compressed means a deficit in "
              "both tails.")

    # Level 2.
    v = read(brm("l2_verdicts.csv"))
    rows = []
    changes = 0
    changes_pooled = 0
    for est in ["standardised", "marginal"]:
        h = v[(v.analysis == "headline") & (v.estimand == est)].set_index(["contrast", "scope"])
        dd = v[(v.analysis == "dec_only") & (v.estimand == est)].set_index(["contrast", "scope"])
        first = True
        for c in CONTRASTS:
            for sc in ["deepseek/deepseek-chat-v3", "google/gemini-3-flash-preview", "openai/gpt-4o-mini",
                       "z-ai/glm-4.7", "pooled"]:
                a = h.loc[(c, sc)]
                b = dd.loc[(c, sc)]
                if a.verdict != b.verdict:
                    if sc == "pooled":
                        changes_pooled += 1
                    else:
                        changes += 1
                rows.append([est.capitalize() if first else "", tex(c) if sc.startswith("deepseek") else "",
                             mname(sc), f2(a.ratio), tex(a.verdict), f2(b.ratio), tex(b.verdict),
                             "" if a.verdict == b.verdict else "*"])
                first = False
            rows.append("ADD")
        rows[-1] = "MID"
    rows = rows[:-1]
    MACROS["DecLTwoChanges"] = str(changes)
    MACROS["DecLTwoChangesPooled"] = str(changes_pooled)
    longtable("S3_dec_l2.tex", [brm("l2_verdicts.csv")],
              "Level 2 on the Full Corpus and on December Rows Only", "tab:s3decl2",
              "l" + P("2.0cm") + "lr" + P("2.6cm") + "r" + P("2.6cm") + "c",
              [" & & & \\multicolumn{2}{c}{Full corpus} & \\multicolumn{2}{c}{December only} & ",
               "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
               "Estimand & Contrast & Model & Ratio & Verdict & Ratio & Verdict & Changed"],
              rows,
              "Rows of \\texttt{l2\\_verdicts.csv} with analysis \\texttt{headline} and \\texttt{dec\\_only}. "
              "Ratio = simulated gap over population gap. * = verdict differs.", size="scriptsize", tabcolsep="3pt")

    # Level 3.
    r3 = read(brm("80c_l3_results.csv"))
    sel = r3[(r3.spec == "S1 primary") & (r3.estimand == "PS") & (r3.framing == "combined") & (r3.outcome == "mean")]
    rows = []
    vchanges = 0
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        a = sel[(sel.corpus == "all") & (sel.model == m)].set_index("group")
        b = sel[(sel.corpus == "dec_only") & (sel.model == m)].set_index("group")
        va = {t: bool(a[f"pass_{t}"].astype(bool).all()) for t in ["0.2SD", "1pt", "2pt"]}
        vb = {t: bool(b[f"pass_{t}"].astype(bool).all()) for t in ["0.2SD", "1pt", "2pt"]}
        if m != "pooled":
            vchanges += sum(int(va[t] != vb[t]) for t in va)
        diff = (b.resid - a.resid).abs().max()
        rows.append([mname(m), f2(a.loc["Overall", "resid"]), f2(b.loc["Overall", "resid"]),
                     f"{f2(a.resid.min())} to {f2(a.resid.max())}", f"{f2(b.resid.min())} to {f2(b.resid.max())}",
                     f2(diff), yn(va["2pt"]), yn(vb["2pt"])])
    MACROS["DecLThreeChanges"] = str(vchanges)
    summ = read(brm("80d_december_summary.csv")).iloc[0]
    MACROS["DecLThreeMaxDiff"] = f2(summ.max_abs_diff)
    MACROS["DecLThreeMaxDiffNoMissing"] = f2(summ.max_abs_diff_no_missing)
    MACROS["DecLThreeRowsMissing"] = ki(summ.rows_with_missing_cells)
    longtable("S3_dec_l3.tex", [brm("80c_l3_results.csv"), brm("80d_december_summary.csv")],
              "Level 3 on the Full Corpus and on December Rows Only", "tab:s3decl3",
              "lrrrrrcc",
              [" & \\multicolumn{2}{c}{Overall residual} & \\multicolumn{2}{c}{Group residuals, range} & "
               "Largest & \\multicolumn{2}{c}{All groups pass at 2 points}",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){7-8}"
               "Model & Full & Dec. & Full & Dec. & change & Full & Dec."],
              rows,
              "Specification S1, post-stratified estimand, framings combined, PHQ-8 mean in points. "
              "Largest change = largest absolute change in a group residual across the ten groups.")

    # Level 4.
    l4 = read(brm("l4_summary.csv"))
    l4d = read(brm("l4_summary_dec.csv"))
    l4 = l4[l4.subset == "all"]
    rows = []
    changes = 0
    for _, r in l4.assign(mo=l4.model.map(mean_name)).sort_values(["mo", "framing"]).iterrows():
        q = l4d[(l4d.model == r.model) & (l4d.framing == r.framing)].iloc[0]
        changes += int(r.level4_pass != q.level4_pass)
        rows.append([r.model, r.framing.capitalize(), f2(r.ev_ratio_12), f2(q.ev_ratio_12),
                     f3(r.phi_nhanes), f3(q.phi_nhanes), tex(r.R4_verdict_e08), tex(q.R4_verdict_e08),
                     yn(r.level4_pass), yn(q.level4_pass)])
    MACROS["DecLFourChanges"] = str(changes)
    longtable("S3_dec_l4.tex", [brm("l4_summary.csv"), brm("l4_summary_dec.csv")],
              "Level 4 on the Full Corpus and on December Rows Only", "tab:s3decl4",
              "llrrrr>{\\raggedright\\arraybackslash}p{2.1cm}>{\\raggedright\\arraybackslash}p{2.1cm}cc",
              [" & & \\multicolumn{2}{c}{$\\lambda_1/\\lambda_2$} & \\multicolumn{2}{c}{Tucker $\\phi$} & "
               "\\multicolumn{2}{c}{R4 at .08} & \\multicolumn{2}{c}{Level 4 pass}",
               "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}"
               "Model & Framing & Full & Dec. & Full & Dec. & Full & Dec. & Full & Dec."],
              rows, "R1 needs $\\lambda_1/\\lambda_2 \\ge 3$ and every loading $\\ge .30$; R2 reads the "
              "lower 90\\% limit of Tucker's $\\phi$ against .95.")


def s3_nhanes2021():
    s = read(brm("l1_summary.csv"))
    pf = read(brm("l1_personfit_main.csv"))
    rows = []
    moved = s[(s.pf_verdict != s.pf_verdict_nhanes2021) & (s.model != "pooled") & (s.framing == "both")]
    MACROS["LOneEraMoved"] = "; ".join(
        f"{r.model}, {tex(short_l1(r.pf_verdict))} to {tex(short_l1(r.pf_verdict_nhanes2021))}"
        for _, r in moved.iterrows()) or "none"
    for _, r in s.assign(mo=s.model.map(mean_name)).sort_values(["mo", "framing"]).iterrows():
        q = pf[(pf.reference == "2021_2023 weighted GRM") & (pf["sample"] == "full") & (pf.subset == "all")
               & (pf.model == r.model) & (pf.framing == r.framing)].iloc[0]
        p = pf[(pf.reference == "2005_2018 weighted GRM") & (pf["sample"] == "full") & (pf.subset == "all")
               & (pf.model == r.model) & (pf.framing == r.framing)].iloc[0]
        rows.append([r.model, r.framing.capitalize(), f2(p.misfit_ratio), f2(q.misfit_ratio),
                     f2(p.overfit_ratio), f2(q.overfit_ratio),
                     tex(short_l1(r.pf_verdict)), tex(short_l1(r.pf_verdict_nhanes2021))])
    longtable("S3_l1_2021.tex", [brm("l1_summary.csv"), brm("l1_personfit_main.csv")],
              "Level 1 Against NHANES 2005--2018 and NHANES 2021--2023", "tab:s3l12021",
              "llrrrr" + P("2.6cm") + P("2.6cm"),
              [" & & \\multicolumn{2}{c}{Misfit ratio} & \\multicolumn{2}{c}{Overfit ratio} & & ",
               "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}"
               "Model & Framing & 2005--18 & 2021--23 & 2005--18 & 2021--23 & Verdict, 2005--18 & Verdict, 2021--23"],
              rows,
              "Each reference has its own weighted graded response model. Every verdict is a fail; the "
              "verdict column gives the failure reading, where compressed means a deficit in both tails.")

    # Level 3 under the 2021-2023 and 2017-2020 references, per model.
    r3 = read(brm("80c_l3_results.csv"))
    sel = r3[(r3.corpus == "all") & (r3.estimand == "PS") & (r3.framing == "combined") & (r3.outcome == "mean")]
    rows = []
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        cells = [mname(m)]
        for spec in ["S1 primary", "S6 2017-2020 pre-pandemic", "S7 2021-2023"]:
            a = sel[(sel.spec == spec) & (sel.model == m)].set_index("group")
            cells += [f2(a.loc["Overall", "resid"]), yn(a.loc["Overall", "pass_2pt"]),
                      yn(a["pass_2pt"].astype(bool).all())]
        rows.append(cells)
    shift = read(brm("80d_era_shift_permodel.csv")).iloc[0]
    MACROS["EraShiftMin"] = f2(shift.shift_min)
    MACROS["EraShiftMax"] = f2(shift.shift_max)
    MACROS["EraShiftMean"] = f2(shift.shift_mean)
    MACROS["EraResidMin"] = f2(shift.resid_min_2021)
    MACROS["EraResidMax"] = f2(shift.resid_max_2021)
    longtable("S3_l3_era.tex", [brm("80c_l3_results.csv"), brm("80d_era_shift_permodel.csv")],
              "Level 3 Overall Residual Under Three NHANES Reference Periods", "tab:s3l3era",
              "lrccrccrcc",
              [" & \\multicolumn{3}{c}{2005--2018 (S1)} & \\multicolumn{3}{c}{2017--2020 (S6)} & "
               "\\multicolumn{3}{c}{2021--2023 (S7)}",
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}"
               "Model & Overall & Row & All & Overall & Row & All & Overall & Row & All"],
              rows,
              "Overall = post-stratified overall residual (simulated minus reference, PHQ-8 points), "
              "framings combined. Row = the overall row passes at 2 points. All = all ten group rows pass "
              "at 2 points.")

    era = read(brm("80d_era.csv"))
    era = era[era.outcome == "mean"].set_index("group")
    rows = []
    for gname in GROUPS:
        if gname not in era.index:
            continue
        r = era.loc[gname]
        rows.append([gname, f2(r.ref_S1_primary), f2(r["ref_S6_2017-2020_pre-pandemic"]), f2(r["ref_S7_2021-2023"]),
                     f2(r.resid_S1_primary), f2(r["resid_S7_2021-2023"]), f2(r.resid_shift_2021_vs_primary)])
    longtable("S3_l3_era_groups.tex", [brm("80d_era.csv")],
              "Level 3 Pooled Residual by Group Under Three Reference Periods", "tab:s3l3eragroups",
              "lrrrrrr",
              [" & \\multicolumn{3}{c}{Reference mean} & \\multicolumn{2}{c}{Pooled residual} & ",
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}"
               "Group & 2005--18 & 2017--20 & 2021--23 & 2005--18 & 2021--23 & Shift"],
              rows,
              "PHQ-8 points, post-stratified, pooled over models and framings. Shift = residual under "
              "2021--2023 minus residual under 2005--2018.")


def s3_marital_income():
    r3 = read(brm("80c_l3_results.csv"))
    sel = r3[(r3.corpus == "all") & (r3.estimand == "PS") & (r3.framing == "combined") & (r3.outcome == "mean")]
    specs = ["S1 primary", "S2 cohabiting as married", "S2b cohabiting as single"]
    rows = []
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        cells = [mname(m)]
        for spec in specs:
            a = sel[(sel.spec == spec) & (sel.model == m)].set_index("group")
            cells += [f2(a.loc["Overall", "resid"]), f2(a.resid.abs().max()), yn(a["pass_2pt"].astype(bool).all())]
        rows.append(cells)
    longtable("S3_marital_l3.tex", [brm("80c_l3_results.csv")],
              "Level 3 Under Three Codings of Marital Status", "tab:s3marl3",
              "lrrcrrcrrc",
              [" & \\multicolumn{3}{c}{S1: cohabiting dropped} & \\multicolumn{3}{c}{S2: as married} & "
               "\\multicolumn{3}{c}{S2b: as single}",
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}"
               "Model & Overall & Max $|R|$ & Pass & Overall & Max $|R|$ & Pass & Overall & Max $|R|$ & Pass"],
              rows,
              "Post-stratified residual in PHQ-8 points, framings combined. Max $|R|$ = largest absolute "
              "group residual over ten groups. Pass = all ten group rows pass at 2 points.")

    v = read(brm("l2_verdicts.csv"))
    variants = [("headline", "Primary"), ("ref_mar_all", "All marital"), ("ref_mar_never", "Never married"),
                ("ref_mar_drop1819", "Drop 18--19"), ("ref_no_marital", "No marital")]
    rows = []
    changes = 0
    for c in CONTRASTS:
        for sc in ["deepseek/deepseek-chat-v3", "google/gemini-3-flash-preview", "openai/gpt-4o-mini",
                   "z-ai/glm-4.7", "pooled"]:
            cells = [tex(c) if sc.startswith("deepseek") else "", mname(sc)]
            base = None
            for an, _ in variants:
                q = v[(v.analysis == an) & (v.estimand == "standardised") & (v.contrast == c) & (v.scope == sc)].iloc[0]
                if base is None:
                    base = q.code
                elif q.code != base and sc != "pooled":
                    changes += 1
                cells.append(f"{f2(q.ratio)} \\texttt{{{tex(q.code)}}}")
            rows.append(cells)
        rows.append("ADD")
    rows = rows[:-1]
    MACROS["MarLTwoChanges"] = str(changes)
    longtable("S3_marital_l2.tex", [brm("l2_verdicts.csv")],
              "Level 2 Standardised Ratios Under Alternative Marital-Status References", "tab:s3marl2",
              P("2.2cm") + "lrrrrr",
              ["Contrast & Model & " + " & ".join(lab for _, lab in variants)],
              rows,
              "Each cell: ratio and verdict code. Codes: \\texttt{s} steepened, \\texttt{k/s} kept or "
              "steepened, \\texttt{a/k} attenuated or kept, \\texttt{m/a} missing or attenuated, "
              "\\texttt{r} reversed, \\texttt{r/m} reversed or missing, \\texttt{u} undetermined, "
              "\\texttt{X} reference too imprecise, \\texttt{N} no population gap. Rows of "
              "\\texttt{l2\\_verdicts.csv} by analysis.")

    # Income anchor, level 3.
    ip = read(brm("80d_income_anchor_permodel.csv"))
    rows = []
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        sub = ip[ip.model == m].set_index("group")
        if sub.empty:
            continue
        cells = [mname(m)]
        for gname in ["Low", "Middle", "High"]:
            if gname in sub.index:
                cells += [f2(sub.loc[gname, "S3 2007-2018 PIR (bridge)"]), f2(sub.loc[gname, "S4 persona-string income"])]
            else:
                cells += ["--", "--"]
        rows.append(cells)
    r3 = read(brm("80c_l3_results.csv"))
    sel = r3[(r3.corpus == "all") & (r3.estimand == "PS") & (r3.framing == "combined") & (r3.outcome == "mean")]
    rows2 = []
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        cells = [mname(m)]
        for spec in ["S1 primary", "S3 2007-2018 PIR (bridge)", "S4 persona-string income",
                     "S5 persona-string, high top-coded"]:
            a = sel[(sel.spec == spec) & (sel.model == m)].set_index("group")
            cells += [f"{f2(a.loc['Low', 'resid'])}, {f2(a.loc['Middle', 'resid'])}, {f2(a.loc['High', 'resid'])}",
                      yn(a["pass_2pt"].astype(bool).all())]
        rows2.append(cells)
    longtable("S3_income_l3.tex", [brm("80c_l3_results.csv")],
              "Level 3 Income-Group Residuals Under Four Income Mappings", "tab:s3incl3",
              "lrcrcrcrc",
              [" & \\multicolumn{2}{c}{S1: poverty ratio} & \\multicolumn{2}{c}{S3: PIR 2007--18} & "
               "\\multicolumn{2}{c}{S4: persona string} & \\multicolumn{2}{c}{S5: top-coded}",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}"
               "Model & Low, Mid, High & Pass & Low, Mid, High & Pass & Low, Mid, High & Pass & Low, Mid, High & Pass"],
              rows2,
              "Post-stratified residuals in PHQ-8 points, framings combined. S3 uses the poverty ratio on "
              "2007--2018 only; S4 maps the persona's dollar income and insurance onto household income "
              "and insurance type; S5 is S4 with the high band top-coded. Pass = all ten group rows pass "
              "at 2 points.", size="scriptsize", tabcolsep="3pt")
    longtable("S3_income_mg.tex", [brm("80d_income_anchor_permodel.csv")],
              "Marginal Income-Group Residuals Under Two Income Mappings", "tab:s3incmg",
              "lrrrrrr",
              [" & \\multicolumn{2}{c}{Low} & \\multicolumn{2}{c}{Middle} & \\multicolumn{2}{c}{High}",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
               "Model & S3 & S4 & S3 & S4 & S3 & S4"],
              rows,
              "Marginal residual (simulated group mean minus reference group mean), PHQ-8 points, from "
              "\\texttt{80d\\_income\\_anchor\\_permodel.csv}.")

    v = read(brm("l2_verdicts.csv"))
    rows = []
    for est in ["standardised", "marginal"]:
        first = True
        for c in ["Low minus High SES", "Middle minus High SES", "Low minus Middle SES"]:
            for sc in ["deepseek/deepseek-chat-v3", "google/gemini-3-flash-preview", "openai/gpt-4o-mini",
                       "z-ai/glm-4.7", "pooled"]:
                a = v[(v.analysis == "headline") & (v.estimand == est) & (v.contrast == c) & (v.scope == sc)].iloc[0]
                b = v[(v.analysis == "ref_income_alt") & (v.estimand == est) & (v.contrast == c) & (v.scope == sc)]
                b = b.iloc[0] if len(b) else None
                rows.append([est.capitalize() if first else "", tex(c) if sc.startswith("deepseek") else "",
                             mname(sc), f2(a.ratio), tex(a.verdict),
                             "--" if b is None else f2(b.ratio), "--" if b is None else tex(b.verdict)])
                first = False
            rows.append("ADD")
        rows[-1] = "MID"
    rows = rows[:-1]
    longtable("S3_income_l2.tex", [brm("l2_verdicts.csv")],
              "Level 2 Income Contrasts Under the Alternative Income Mapping", "tab:s3incl2",
              "l" + P("2.2cm") + "lr" + P("2.8cm") + "r" + P("2.8cm"),
              [" & & & \\multicolumn{2}{c}{Poverty ratio (primary)} & \\multicolumn{2}{c}{Persona string} ",
               "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
               "Estimand & Contrast & Model & Ratio & Verdict & Ratio & Verdict"],
              rows,
              "Rows of \\texttt{l2\\_verdicts.csv} with analysis \\texttt{headline} and \\texttt{ref\\_income\\_alt}. "
              "The alternative mapping was run for the marginal estimand on the income contrasts only.",
              tabcolsep="3pt")


# ---------------------------------------------------------------------------------------------
# S4: full per-model results
# ---------------------------------------------------------------------------------------------
def s4_gate():
    g = read(brm("gate_dstudy.csv"))
    rows = []
    rows2 = []
    for oc in ["PHQ-8 total", "PHQ-8 >= 10"]:
        sub = g[g.outcome == oc]
        sub = sub.assign(mo=sub.model.map(mean_name)).sort_values(["mo", "framing"])
        first = True
        for _, r in sub.iterrows():
            rows.append([tex(oc) if first else "", r.model, r.framing.capitalize(),
                         f"{f2(r.s2_p)} {ci(r.s2_p_lo, r.s2_p_hi)}", f"{f2(r.s2_e)} {ci(r.s2_e_lo, r.s2_e_hi)}",
                         f"{f3(r.phi_nested_1)} {ci(r.phi_nested_1_lo, r.phi_nested_1_hi, f3)}",
                         f"{f3(r.phi_nested_30)} {ci(r.phi_nested_30_lo, r.phi_nested_30_hi, f3)}"])
            kk = {c: [gate_k(r[f"s2_e{s}"], oc, c) for s in ("", "_lo", "_hi")] for c in ("min", "rec")}
            if oc == "PHQ-8 total":
                assert kk["min"][0] == r.k_se_tol2 and kk["rec"][0] == r.k_se_tol1, (r.model, r.framing)
            rows2.append([tex(oc) if first else "", r.model, r.framing.capitalize(),
                          f"{ki(r.k_phi80)} {ci(r.k_phi80_lo, r.k_phi80_hi, ki)}",
                          f"{ki(r.k_phi90)} {ci(r.k_phi90_lo, r.k_phi90_hi, ki)}",
                          f"{ki(kk['min'][0])} {ci(kk['min'][1], kk['min'][2], ki)}",
                          f"{ki(kk['rec'][0])} {ci(kk['rec'][1], kk['rec'][2], ki)}"])
            first = False
        rows.append("MID")
        rows2.append("MID")
    rows, rows2 = rows[:-1], rows2[:-1]
    longtable("S4_gate_components.tex", [brm("gate_dstudy.csv")],
              "Gate: Variance Components and Dependability With Persona-Bootstrap Intervals",
              "tab:s4gatecomp", "lll" + P("2.5cm") + P("2.5cm") + P("2.4cm") + P("2.4cm"),
              ["Outcome & Model & Framing & $\\sigma^2_p$ & $\\sigma^2_r$ & $\\phi(1)$ & $\\phi(30)$"],
              rows,
              "Brackets: 95\\% persona-bootstrap intervals. $\\sigma^2_p$ = variance between personas, "
              "$\\sigma^2_r$ = variance between draws of one persona (squared PHQ-8 points for the total, "
              "squared proportion for the indicator). $\\phi$ from the nested design.",
              size="scriptsize", tabcolsep="3pt")
    longtable("S4_gate_k.tex", [brm("gate_dstudy.csv"), brm("80a_tolerance_basis.csv")],
              "Gate: Draws Needed per Persona With Persona-Bootstrap Intervals", "tab:s4gatek",
              "lllllll",
              [" & & & \\multicolumn{2}{c}{Separation} & \\multicolumn{2}{c}{Gate rule}",
               "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
               "Outcome & Model & Framing & $\\phi \\ge .80$ & $\\phi \\ge .90$ & Min. & Rec."],
              rows2,
              "Brackets: 95\\% persona-bootstrap intervals for $k$. Min.\\ = $k$ for SE $\\le$ 0.25 "
              "reference SD (0.98 PHQ-8 points); Rec.\\ = $k$ for SE $\\le$ 0.125 reference SD (0.49 "
              "points); reference SD 3.935 (NHANES 2005--2018). Indicator rows: SE $\\le$ 0.10 and 0.05 in "
              "proportion. Separation columns: $k$ for persona separation $\\phi(k)$, required only for "
              "uses that compare personas.",
              tabcolsep="3pt")


def s4_l2():
    v = read(brm("l2_verdicts.csv"))
    h = v[v.analysis == "headline"]
    MACROS["LTwoAlpha"] = f"{h.alpha_one_sided.iloc[0]:.4f}".lstrip("0")
    MACROS["LTwoLevel"] = p1(100 * h.interval_level.iloc[0])
    MACROS["LTwoFamily"] = str(h.contrast.nunique())
    for est, tag in [("standardised", "std"), ("marginal", "marg")]:
        rows = []
        for c in CONTRASTS:
            sub = h[(h.estimand == est) & (h.contrast == c)]
            sub = sub.assign(mo=sub.scope.map(lambda s: mean_name(mname(s)))).sort_values("mo")
            first = True
            for _, r in sub.iterrows():
                interval = "unbounded" if str(r.ci_unbounded) == "True" else ci(r.ci_lo, r.ci_hi)
                rows.append([tex(c) if first else "", mname(r.scope), f2(r.population) if first else "",
                             f2(r.simulated), f2(r.ratio), interval, tex(r.verdict), tex(r.verdict_popstop)])
                first = False
            rows.append("ADD")
        rows = rows[:-1]
        longtable(f"S4_l2_{tag}.tex", [brm("l2_verdicts.csv")],
                  f"Level 2 per Model, {est.capitalize()} Estimand", f"tab:s4l2{tag}",
                  P("2.0cm") + "lrrrl" + P("2.5cm") + P("2.5cm"),
                  ["Contrast & Model & Pop.\\ gap & Sim.\\ gap & Ratio & "
                   f"{MACROS['LTwoLevel']}\\% interval & Verdict & Population-only stop"],
                  rows,
                  "Gaps in PHQ-8 points. Ratio = simulated gap over population gap, with its Fieller "
                  f"interval at the Bonferroni level for {MACROS['LTwoFamily']} contrasts (one-sided "
                  f"$\\alpha = {MACROS['LTwoAlpha']}$). Verdict: the primary rule, whose "
                  "reference-imprecision stop fires when the observed simulated gap, treated as measured "
                  "without error, would still give a ratio interval spanning three or more regions. "
                  "Population-only stop: the verdict when that stop instead fires whenever the population "
                  "gap's relative half-width exceeds 0.25, before the simulation is read (sensitivity). "
                  "Rows of \\texttt{l2\\_verdicts.csv} with analysis \\texttt{headline}.",
                  size="scriptsize", tabcolsep="3pt")


def s4_l3():
    h = read(brm("80d_headline.csv"))
    r3 = read(brm("80c_l3_results.csv"))
    sel = r3[(r3.spec == "S1 primary") & (r3.corpus == "all") & (r3.estimand == "PS") & (r3.framing == "combined")]
    rows = []
    rows2 = []
    for m in ["deepseek-chat-v3", "gemini-3-flash-preview", "gpt-4o-mini", "glm-4.7", "pooled"]:
        sub = h[h.model == m].set_index("group")
        chk = sel[(sel.model == m) & (sel.outcome == "mean")].set_index("group")
        first = True
        for gname in GROUPS:
            r = sub.loc[gname]
            assert abs(chk.loc[gname, "resid"] - r.resid) < 1e-3, f"80d_headline and 80c disagree: {m} {gname}"
            rows.append([mname(m) if first else "", gname, f2(r.sim), f2(r.ref), f2(r.resid),
                         ci(r.ci90_lo, r.ci90_hi), yn(r["pass_0.2SD"]), yn(r.pass_1pt), yn(r.pass_2pt),
                         yn(r.pass_5pt_benchmark)])
            rows2.append([mname(m) if first else "", gname, p1(r.prev_sim), p1(r.prev_ref), p1(r.prev_resid),
                          ci(r.prev_ci90_lo, r.prev_ci90_hi, p1), f2(r.ratio), ci(r.ratio_ci90_lo, r.ratio_ci90_hi),
                          yn(r["pass_2.5pp"]), yn(r.pass_5pp), yn(r["pass_ratio_0.80_1.25"])])
            first = False
        rows.append("MID")
        rows2.append("MID")
    rows, rows2 = rows[:-1], rows2[:-1]
    sd = r3["tol_0.2SD_points"].dropna().iloc[0]
    MACROS["LThreeSDTol"] = f2(sd)
    longtable("S4_l3_mean.tex", [brm("80d_headline.csv"), brm("80c_l3_results.csv")],
              "Level 3 per Model and Group: Mean PHQ-8", "tab:s4l3mean",
              "llrrrlcccc",
              [" & & & & & & \\multicolumn{4}{c}{Passes at tolerance}",
               "\\cmidrule(lr){7-10}"
               f"Model & Group & Sim. & Ref. & Residual & 90\\% CI & 0.2 SD & 1 pt & 2 pt & 5 pt (bench.)"],
              rows,
              "Post-stratified estimand, specification S1, framings combined. Residual = simulated minus "
              f"reference, PHQ-8 points. 0.2 SD = {MACROS['LThreeSDTol']} points. A row passes when its 90\\% "
              "interval lies inside the tolerance (two one-sided tests). The 5-point column is a benchmark "
              "only.")
    longtable("S4_l3_prev.tex", [brm("80d_headline.csv")],
              "Level 3 per Model and Group: Share Scoring 10 or More", "tab:s4l3prev",
              "llrrrlrlccc",
              [" & & & & & & & & \\multicolumn{3}{c}{Passes}",
               "\\cmidrule(lr){9-11}"
               "Model & Group & Sim.\\ \\% & Ref.\\ \\% & Diff.\\ (pp) & 90\\% CI & Ratio & 90\\% CI & 2.5 pp & 5 pp & 0.80--1.25"],
              rows2,
              "Post-stratified estimand, specification S1, framings combined. pp = percentage points. "
              "Ratio = simulated share over reference share, Fieller interval.", size="scriptsize", tabcolsep="3pt")


def s4_l4():
    inv = read(brm("l4_invariance.csv"))
    rows = []
    for pop in inv.population.unique():
        sub = inv[inv.population == pop]
        first = True
        for _, r in sub.iterrows():
            rows.append([tex(pop) if first else "", r.attribute, tex(r.step), ki(r.G), ki(r.df_diff),
                         f3(r.rmsea_d), ci(r.rmsea_d_lo90, r.rmsea_d_hi90, f3), f2(r.c_hat),
                         tex(r.verdict_e08), tex(r.verdict_e05)])
            first = False
        rows.append("MID")
    rows = rows[:-1]
    longtable("S4_l4_invariance.tex", [brm("l4_invariance.csv")],
              "Level 4 Invariance Steps: RMSEA of the Difference", "tab:s4l4inv",
              P("2.4cm") + "l" + P("1.6cm") + "rrrlr" + P("2.4cm") + P("2.4cm"),
              ["Population & Attribute & Step & $G$ & $d$ & RMSEA$_D$ & 90\\% CI & $\\hat c$ & At .08 & At .05"],
              rows,
              "A step holds when the upper 90\\% limit of RMSEA$_D$ is below the margin, fails when the "
              "lower limit is above it, and is indeterminate otherwise. The margin of .08 is primary and .05 "
              "a sensitivity analysis. $\\hat c$ = permutation scaling of the expected $\\Delta F$ under exact "
              "invariance. The polychoric metric rows are reported as fitted.", size="scriptsize", tabcolsep="3pt")


def s4_multiplicity():
    v = read(brm("l2_verdicts.csv"))
    h = v[v.analysis == "headline"]
    r3 = read(brm("80c_l3_results.csv"))
    pf = read(brm("l1_personfit_main.csv"))
    l4 = read(brm("l4_summary.csv"))
    inv = read(brm("l4_invariance.csv"))
    gd = read(brm("gate_dstudy.csv"))
    n_contr = h.contrast.nunique()
    n_groups = r3.group.nunique()
    tau = pf.tau.dropna().iloc[0]
    n_fram = l4.framing.nunique()
    ref_steps = inv[(inv.population == inv.population.iloc[0]) & ~inv.step.str.contains("polychoric")]
    n_steps = ref_steps.shape[0]
    assert set(l4.framing) == {"clinical", "narrative"}, "unexpected framings in l4_summary.csv"
    n_models = gd.model.nunique()
    alpha = h.alpha_one_sided.iloc[0]
    level = h.interval_level.iloc[0]
    rows = [
        ["Gate", "Draws needed per persona", f"Each model $\\times$ framing ({n_models} $\\times$ {n_fram})",
         "Estimation of $\\phi(k)$ and SE$(k)$; no test", "None; persona-bootstrap intervals reported"],
        ["Level 1", "Person-fit like people's", "Each model $\\times$ framing",
         f"2 tails: misfit ratio and overfit ratio, each against $[1/\\tau, \\tau]$, $\\tau = {tau:g}$",
         "Intersection-union: pass needs both 90\\% intervals inside the band; no adjustment"],
        ["Level 2", "Group gaps like people's", "Each model (the pooled row is description)",
         f"{n_contr} contrasts",
         f"Bonferroni: one-sided $\\alpha = .05/{n_contr} = {f'{alpha:.4f}'.lstrip('0')}$, "
         f"{p1(100 * level)}\\% Fieller intervals"],
        ["Level 3", "Group levels like people's, at a stated tolerance", "Each model $\\times$ tolerance",
         f"{n_groups} group rows", "Intersection-union: pass needs every row's 90\\% interval inside the "
         "tolerance (two one-sided tests at .05); no adjustment"],
        ["Level 4", "Structure like people's", "Each model",
         f"R1, R2 and R4 in each of {n_fram} framings; R4 reads {n_steps} invariance steps",
         "Intersection-union: pass needs every rule in every framing; R4 needs every step to match the "
         "reference's verdict; R3 is a diagnostic and not counted"],
    ]
    longtable("S4_multiplicity.tex", [brm("l2_verdicts.csv"), brm("80c_l3_results.csv"), brm("l1_personfit_main.csv"),
                                      brm("l4_summary.csv"), brm("l4_invariance.csv"), brm("gate_dstudy.csv")],
              "Multiplicity: Family and Procedure at Each Rung", "tab:s4mult",
              P("1.3cm") + P("2.8cm") + P("2.9cm") + P("3.6cm") + P("4.5cm"),
              ["Rung & Claim & Family & Members & Procedure"], rows,
              "Family sizes, $\\tau$, $\\alpha$ and the interval level are read from the receipts named in "
              "the source comment of this table.")


# ---------------------------------------------------------------------------------------------
# S5: controls
# ---------------------------------------------------------------------------------------------
def s5():
    c1 = read(brm("83c_l1.csv"))
    rows = []
    labels = {"L1 shuffle": "Shuffle item values", "L1 typical": "Most typical vector"}
    c1 = c1.assign(cond=c1.condition.fillna("Positive control"))
    for cond in ["Positive control", "L1 shuffle", "L1 typical"]:
        sub = c1[c1.cond == cond]
        first = True
        for _, r in sub.iterrows():
            rows.append([labels.get(cond, cond) if first else "", pc(r.dose), r.unit, ki(r.n),
                         f"{pc(r.expected_reading)} {ci(r.expected_reading_lo95, r.expected_reading_hi95, pc)}",
                         pc(r.share_pass), pc(r["share_fail: compressed"]), pc(r["share_fail: excess misfit"]),
                         pc(r.share_undetermined), f2(r.misfit_ratio_mean), f2(r.overfit_ratio_mean)])
            first = False
        rows.append("MID")
    rows = rows[:-1]
    pos = c1[(c1.cond == "Positive control") & (c1.unit == "both")].iloc[0]
    MACROS["CtlLOnePass"] = pc(pos.share_pass)
    MACROS["CtlLOneN"] = ki(pos.n)
    MACROS["CtlLOneMisfit"] = f2(pos.misfit_ratio_mean)
    longtable("S5_l1.tex", [brm("83c_l1.csv")],
              "Controls, Level 1: Detection by Dose", "tab:s5l1",
              ">{\\raggedright\\arraybackslash}p{2.2cm}rlr>{\\raggedright\\arraybackslash}p{2.6cm}rrrrrr",
              ["Condition & Dose (\\%) & Unit & $n$ & Expected reading (\\%) & Pass & Compressed & Excess misfit & "
               "Undet. & Misfit & Overfit"],
              rows,
              "Dose = share of draws altered. Unit: \\texttt{both} = a pseudo-model read on both framings; "
              "\\texttt{clinical} and \\texttt{narrative} = one framing. Expected reading = share of "
              "replicates returning the reading the planted failure should produce (a pass for the positive "
              "control), with a 95\\% interval. Pass, Compressed, Excess misfit and Undet.\\ are shares of "
              "replicates in percent. Misfit and Overfit are mean ratios.", size="scriptsize", tabcolsep="3pt")

    c2 = read(brm("83c_l2.csv"))
    rows = []
    for ref in c2.reference.unique():
        for scope in c2.scope.unique():
            sub = c2[(c2.reference == ref) & (c2.scope == scope)]
            first = True
            for c in CONTRASTS:
                s2 = sub[sub.contrast == c]
                firstc = True
                for _, r in s2.iterrows():
                    rows.append([f"{tex(ref)}, {scope}" if first else "", tex(c) if firstc else "",
                                 f2(r.dose), r.true_region, ki(r.n), pc(r.correct), pc(r.correct_or_compatible),
                                 pc(r.wrong), pc(r.share_undetermined), pc(r.share_stopped)])
                    first = False
                    firstc = False
            rows.append("MID")
    rows = rows[:-1]
    MACROS["CtlLTwoWrongMax"] = pc(c2.wrong.max())
    longtable("S5_l2.tex", [brm("83c_l2.csv")],
              "Controls, Level 2: Verdicts by Dose", "tab:s5l2",
              ">{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{2.6cm}rlrrrrrr",
              ["Reference, scope & Contrast & $g$ & True region & $n$ & Correct & Correct or compatible & "
               "Wrong & Undet. & Stopped"],
              rows,
              "$g$ = planted ratio of the simulated gap to the population gap. Shares in percent. Correct = "
              "the single true region named; Correct or compatible = a verdict that contains the true region; "
              "Wrong = a determinate verdict that excludes it; Stopped = either stop. Half sample: the "
              "reference is the other NHANES half. Audit precision: the reference standard error is scaled "
              "to the full NHANES sample.", size="scriptsize", tabcolsep="3pt")

    cv = read(brm("83c_l2_coverage.csv"))
    rows = []
    for scope in cv.scope.unique():
        sub = cv[cv.scope == scope]
        first = True
        for c in CONTRASTS:
            s2 = sub[sub.contrast == c]
            firstc = True
            for _, r in s2.iterrows():
                rows.append([scope if first else "", tex(c) if firstc else "", f2(r.dose), ki(r.n),
                             f3(r.coverage), f2(r.ratio_mean), f2(r.ratio_sd)])
                first = False
                firstc = False
        rows.append("MID")
    rows = rows[:-1]
    pm = cv[cv.scope == "per model"]
    MACROS["CtlCovMin"] = f3(pm.coverage.min())
    MACROS["CtlCovMax"] = f3(pm.coverage.max())
    longtable("S5_l2_coverage.tex", [brm("83c_l2_coverage.csv")],
              "Controls, Level 2: Coverage of the True Ratio", "tab:s5l2cov",
              "llrrrrr",
              ["Scope & Contrast & $g$ & $n$ & Coverage & Mean ratio & SD of ratio"], rows,
              "Coverage = share of replicates whose interval contains the planted ratio; nominal "
              f"{f3(read(brm('l2_verdicts.csv')).interval_level.iloc[0])}.")

    c3 = read(brm("83c_l3.csv"))
    rows = []
    for scope in c3.scope.unique():
        sub = c3[c3.scope == scope]
        first = True
        for _, r in sub.iterrows():
            rows.append([scope if first else "", f2(r.dose), ki(r.n),
                         f"{pc(r['pass_0.2SD'])} {ci(r['pass_0.2SD_lo95'], r['pass_0.2SD_hi95'], pc)}",
                         f"{pc(r.pass_1pt)} {ci(r.pass_1pt_lo95, r.pass_1pt_hi95, pc)}",
                         f"{pc(r.pass_2pt)} {ci(r.pass_2pt_lo95, r.pass_2pt_hi95, pc)}"])
            first = False
        rows.append("MID")
    rows = rows[:-1]
    longtable("S5_l3.tex", [brm("83c_l3.csv")],
              "Controls, Level 3: Share of Replicates Passing All Ten Groups", "tab:s5l3",
              "lrrlll",
              ["Scope & Shift $d$ & $n$ & 0.2 SD & 1 point & 2 points"], rows,
              "Shift $d$ = points added to every persona mean. Shares in percent with 95\\% intervals. At "
              "$d = 0$ the share is the pass rate of real people; at $d > 0$ a failure is the correct reading.")

    c4 = read(brm("83c_l4.csv"))
    # Frozen R2 (congruence and loading RMSD, three-way) and Pass come from the 83h replicates,
    # which rerun 83b's level-4 block with the size condition recorded.
    h = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(brm(os.path.join("83h_reps", "rep_*.csv"))))],
                  ignore_index=True)
    h["cond"] = h.condition.fillna("Positive control")
    h["passed"] = h.R1.astype(bool) & (h.R2_verdict == "pass") & h.R3.astype(bool)
    hs = h.groupby(["cond", "dose", "unit"]).agg(n=("rep", "nunique"),
                                                 r2p=("R2_verdict", lambda v: np.mean(v == "pass")),
                                                 r2u=("R2_verdict", lambda v: np.mean(v == "unresolved")),
                                                 r2f=("R2_verdict", lambda v: np.mean(v == "fail")),
                                                 passed=("passed", "mean"), rmsd=("loading_rmsd", "mean"))
    rows = []
    c4 = c4.assign(cond=c4.condition.fillna("Positive control"))
    for _, r in c4.iterrows():
        key = (r.cond, r.dose, r.unit)
        q = hs.loc[key] if key in hs.index else None
        rows.append([tex(r.cond), pc(r.dose), r.unit, ki(r.n), pc(r.R1), pc(r.R3),
                     ki(q.n) if q is not None else "--",
                     pc(q.r2p) if q is not None else "--", pc(q.r2u) if q is not None else "--",
                     pc(q.r2f) if q is not None else "--",
                     pc(q.passed) if q is not None and r.unit == "model" else "--",
                     f2(r.ev_ratio_mean), f2(q.rmsd) if q is not None and r.unit != "model" else "--",
                     f2(r.within_ev_ratio_mean)])
    longtable("S5_l4.tex", [brm("83c_l4.csv"), brm("83h_r2_size.csv")],
              "Controls, Level 4: Share of Replicates Passing Each Rule", "tab:s5l4",
              "lrlrrrrrrrrrrr",
              [" & & & \\multicolumn{3}{c}{200 replicates} & \\multicolumn{5}{c}{R2 calibration} & & & ",
               "\\cmidrule(lr){4-6}\\cmidrule(lr){7-11}"
               "Condition & Dose (\\%) & Unit & $n$ & R1 & R3 & $n$ & R2 pass & R2 unres. & R2 fail & Pass & "
               "$\\lambda_1/\\lambda_2$ & RMSD & Within $\\lambda_1/\\lambda_2$"],
              rows,
              "Dose = share of draws rebuilt item by item from other draws of the same persona. Rule columns "
              "are shares of replicates in percent. R2 is read on congruence and loading RMSD (pass, "
              "unresolved, fail) in the R2 calibration replicates (script 83h), which rerun the level-4 "
              "block with the RMSD recorded. Pass is scored on R1, R2 and R3 in both framings (unit "
              "\\texttt{model}) in those replicates. R4 was not simulated. $\\lambda_1/\\lambda_2$, RMSD "
              "(mean loading RMSD against the reference) and the within-persona ratio are means over "
              "replicates. The reference-half row reads the NHANES half used as reference.",
              size="scriptsize", tabcolsep="2pt")

    p = read(brm("83f_l2_parametric_power.csv"))
    settings = list(p.se_G_setting.unique())
    rows = []
    for c in CONTRASTS:
        sub = p[p.contrast == c]
        first = True
        for d in sorted(sub.dose.unique()):
            s2 = sub[sub.dose == d]
            cells = [tex(c) if first else "", f2(d), s2.true_region.iloc[0]]
            for st in settings:
                r = s2[s2.se_G_setting == st].iloc[0]
                cells.append(f"{pc(r.correct)} ({pc(r.correct_or_compatible)})")
            rows.append(cells)
            first = False
        rows.append("ADD")
    rows = rows[:-1]
    MACROS["CtlPowerWrongMax"] = pc(p.wrong.max())
    longtable("S5_l2_power.tex", [brm("83f_l2_parametric_power.csv")],
              "Controls, Level 2 at the Corpus's Own Precision: Parametric Power", "tab:s5l2power",
              ">{\\raggedright\\arraybackslash}p{2.6cm}rl" + "r" * len(settings),
              ["Contrast & $g$ & True region & " + " & ".join(tex(s).capitalize() for s in settings)],
              rows,
              "Each cell: percent of draws naming the true region alone, and in parentheses the percent "
              "naming a verdict that contains it. Columns: the simulated-gap standard error, set to the "
              "smallest or largest of the four real models under the draw-only (conditional) or the pairs "
              f"standard error. The largest wrong-verdict rate in any cell is {MACROS['CtlPowerWrongMax']}\\%.",
              size="scriptsize", tabcolsep="3pt")


# ---------------------------------------------------------------------------------------------
# S6: external level 3
# ---------------------------------------------------------------------------------------------
def s6():
    op_path = os.path.join(EXT, "opinionqa", "level3_groups.csv")
    ar_path = os.path.join(EXT, "argyle", "level3_groups.csv")
    o = read(op_path)
    rows = []
    for (meth, mod, st), sub in o.groupby(["method", "model", "steering"], sort=True):
        rows.append([texb(meth), texb(mod), tex(st), ki(len(sub)),
                     f"{pc(sub.share_plurality_match.median())} ({pc(sub.share_plurality_match.min())}--"
                     f"{pc(sub.share_plurality_match.max())})",
                     pc(sub.human_resample_plurality.median()) if sub.human_resample_plurality.notna().any() else "--",
                     pc(sub.share_within_strict.median()) if sub.share_within_strict.notna().any() else "--",
                     f2(sub.mean_abs_residual.median()),
                     ki(int(sub.passes_use_tolerance_90.astype(bool).sum()))])
    MACROS["ExtOqaRows"] = ki(len(o))
    MACROS["ExtOqaPass"] = ki(int(o.passes_use_tolerance_90.astype(bool).sum()))
    MACROS["ExtOqaMatchMin"] = pc(o.share_plurality_match.min())
    MACROS["ExtOqaMatchMax"] = pc(o.share_plurality_match.max())
    MACROS["ExtOqaMatchMed"] = pc(o.share_plurality_match.median())
    MACROS["ExtOqaHumanMed"] = pc(o.human_resample_plurality.median())
    longtable("S6_opinionqa.tex", [op_path],
              "External Level 3 Illustration: OpinionQA Steered Distributions Against Pew", "tab:s6oqa",
              P("2.3cm") + P("2.2cm") + "lr" + P("2.3cm") + R("1.3cm") + R("1.2cm") + R("1.1cm") + R("1.2cm"),
              ["Method & Model & Steering & Groups & Plurality match, median (range) & Human resample & "
               "Within strict & Mean $|R|$ & Groups passing"],
              rows,
              "Shares in percent, medians over groups. Plurality match = share of questions on which the "
              "model's plurality answer equals the Pew plurality answer. Human resample = the same share for "
              "a fresh Pew sample of the same size. Within strict = share of questions whose residual lies "
              "inside the human group's own 95\\% sampling half-width. Mean $|R|$ = mean absolute residual "
              "as a share of the scale range. Groups passing = groups whose plurality match is at least 90\\%, "
              "the placeholder use tolerance.", size="scriptsize", tabcolsep="3pt")

    a = read(ar_path)
    rows = []
    for run, sub in a.groupby("run", sort=True):
        rows.append([tex(run), ki(len(sub)), ki(int(sub.passes_strict.astype(bool).sum())),
                     ki(int(sub.passes_use.astype(bool).sum()))])
    main = a[a.run == "t0.7_main"]
    MACROS["ExtArgCells"] = ki(len(main))
    MACROS["ExtArgStrict"] = ki(int(main.passes_strict.astype(bool).sum()))
    MACROS["ExtArgUse"] = ki(int(main.passes_use.astype(bool).sum()))
    longtable("S6_argyle_runs.tex", [ar_path],
              "External Level 3 Illustration: Argyle et al.\\ Study 3, Passing Cells by Run", "tab:s6argruns",
              "lrrr", ["Run & Outcome $\\times$ group cells & Pass strict & Pass use"], rows,
              "Strict tolerance: the human group mean's own 95\\% sampling half-width. Use tolerance: a "
              "placeholder of 5 percentage points for proportions and one response category for ordinal "
              "scales.")
    rows = []
    for oc, sub in main.groupby("outcome", sort=False):
        allr = sub[sub.group == "all:All respondents"].iloc[0]
        rows.append([tex(oc), f2(allr.human_mean), f2(allr.silicon_mean), f2(allr.residual),
                     ci(allr.residual_ci90_lo, allr.residual_ci90_hi), f2(allr.use_tol),
                     yn(allr.passes_use), ki(len(sub)), ki(int(sub.passes_strict.astype(bool).sum())),
                     ki(int(sub.passes_use.astype(bool).sum())), pc(allr.individual_agreement)])
    longtable("S6_argyle_main.tex", [ar_path],
              "External Level 3 Illustration: Argyle et al.\\ Study 3, Main Run by Outcome", "tab:s6argmain",
              ">{\\raggedright\\arraybackslash}p{3.4cm}rrrlrcrrrr",
              [" & \\multicolumn{6}{c}{All respondents} & \\multicolumn{3}{c}{Groups} & ",
               "\\cmidrule(lr){2-7}\\cmidrule(lr){8-10}"
               "Outcome & Human & Silicon & Residual & 90\\% CI & Use tol. & Pass & $n$ & Strict & Use & Agree (\\%)"],
              rows,
              "Main run (\\texttt{t0.7\\_main}). Proportions for the first five outcomes, scale points for the "
              "rest. Groups: all respondents plus twelve age, education, gender, party and race groups. "
              "Agree = share of respondents whose silicon answer equals their own answer.",
              size="scriptsize", tabcolsep="3pt")


def write_macros():
    out = [header([brm("87_row_sources.csv")])]
    for k, v in MACROS.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    write("supp_macros.tex", "".join(out))


def main():
    os.makedirs(TABLES, exist_ok=True)
    s1()
    s3_row_sources()
    s3_december()
    s3_nhanes2021()
    s3_marital_income()
    s4_gate()
    s4_l2()
    s4_l3()
    s4_l4()
    s4_multiplicity()
    s5()
    s6()
    write_macros()
    for k in sorted(MACROS):
        print(f"{k} = {MACROS[k]}")
    print("fragments:", sorted(f for f in os.listdir(TABLES) if f.endswith(".tex")))


if __name__ == "__main__":
    main()
