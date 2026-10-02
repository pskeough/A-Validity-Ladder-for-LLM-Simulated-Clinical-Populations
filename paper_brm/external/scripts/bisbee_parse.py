"""Parse Bisbee et al. (2024) synthetic ANES thermometers into a long file for the ladder.

Source: PA_replication/data/raw/therm_ANES_RR1.csv (ChatGPT gpt-3.5-turbo, June 2023, the data
behind the paper's main results) and therm_ANES_RR1_GPT4.csv. Cleaning follows the authors'
code/2_DATA_detailed_prompt_june_prep.R line by line:
  - rows whose columns shifted left by one stray delimiter are shifted back (fix 1), and rows
    whose group field holds "group therm explanation confidence" are split by the authors' regex
    (fix 2); unfixable rows keep prompt == "1" and are dropped with it;
  - group labels normalised with the authors' substitutions; thermometer "-20" read as 20 and
    decimals truncated; rows whose thermometer is not numeric are dropped;
  - only the 11 groups the ANES asks about and only respondents in the authors' anes_simp.csv;
  - prompt_simple: "full" (demographics and politics), "demogs", "pol", from the prompt text.
Receipts against the authors' log (code/LOG/2_DATA_detailed_prompt_june_prep_LOG.txt):
  296,594 rows with prompt "1" before fixing; 290,269 rows fixed; 11,527,925 rows in resCleaned;
  248,490 respondent x prompt x group cells; minimum 10 draws in a cell.

usage: python bisbee_parse.py [rr1|gpt4]
Writes ../raw/bisbee2024/derived/{rr1,gpt4}_long.parquet and {rr1,gpt4}_cellcounts.parquet, and
../results/bisbee/parse_receipts_{rr1,gpt4}.csv (tracked in the release; raw/ is not).
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw", "bisbee2024", "PA_replication", "data", "raw")
DER = os.path.join(HERE, "..", "raw", "bisbee2024", "derived")
os.makedirs(DER, exist_ok=True)
GROUPS = ["Democratic Party", "Republican Party", "Black Americans", "White Americans", "Asian Americans",
          "Gays and Lesbians", "Muslims", "Jews", "Liberals", "Conservatives", "Christians"]
SHIFT_RE = re.compile(r"Democrat|Republic|American|Muslim|Women|Liberal|Conser|Gays|Christian|Immigrant|Jew")
FIX2_RE = re.compile(r"(.*\w+)\s+(\d+)\s+(.*)\s+(\d+)")


def norm_group(s):
    """The authors' group substitutions, innermost first."""
    if not isinstance(s, str):
        return s
    s = s.replace("Republic Party", "Republican Party").replace("Democrat Party", "Democratic Party")
    s = re.sub(r"lesbians|Lesbains", "Lesbians", s)
    s = re.sub(r"\?|^The |:|\.", "", s).strip()
    s = s.replace("party", "Party")
    s = re.sub(r"^Republican$", "Republicans", s)
    s = re.sub(r"^Democrat$", "Democrats", s)
    return s.strip()


def to_therm(x):
    if not isinstance(x, str):
        return np.nan
    x = "20" if x == "-20" else re.sub(r"\..*", "", x)
    try:
        return float(x)
    except ValueError:
        return np.nan


def prompt_simple(p):
    if not isinstance(p, str):
        return None
    if "\nYou are [IDEO" in p:
        return "full"
    if "You are a [AGE" in p:
        return "demogs"
    return "pol"


def read_chunks(path, head, names, rec):
    """Stream the file as strings. Rows with the header's field count come through pyarrow; rows
    with one extra field (the shifted rows, readr's X21) are caught by the invalid-row handler,
    parsed with the csv module and yielded with their own batch. Single-threaded."""
    import csv
    import io

    import pyarrow as pa
    from pyarrow import csv as pcsv

    extra = []

    def handler(row):
        extra.append(row.text)
        return "skip"

    ro = pcsv.ReadOptions(block_size=1 << 26, use_threads=False, column_names=head, skip_rows=1)
    po = pcsv.ParseOptions(newlines_in_values=True, invalid_row_handler=handler)
    co = pcsv.ConvertOptions(column_types={c: pa.string() for c in head}, null_values=["", "NA"],
                             strings_can_be_null=True, quoted_strings_can_be_null=True)
    for batch in pcsv.open_csv(path, read_options=ro, parse_options=po, convert_options=co):
        df = batch.to_pandas()
        df["X21"] = None
        if extra:
            rows = []
            for t in extra:
                for f in csv.reader(io.StringIO(t)):
                    f = [None if v in ("", "NA") else v for v in f]
                    if len(f) == len(names):
                        rows.append(f)
                    else:
                        rec["bad_lines"] += 1
            extra.clear()
            if rows:
                df = pd.concat([df, pd.DataFrame(rows, columns=names)], ignore_index=True)
        yield df


def parse(tag):
    fn = {"rr1": "therm_ANES_RR1.csv", "gpt4": "therm_ANES_RR1_GPT4.csv"}[tag]
    path = os.path.join(RAW, fn)
    head = pd.read_csv(path, nrows=0).columns.tolist()
    names = head + ["X21"]
    keep_ids = set(pd.read_csv(os.path.join(RAW, "anes_simp.csv"), usecols=["respID"]).respID.astype(str))
    rec = dict(rows=0, prompt_is_1=0, fixed1=0, fixed2=0, bad_lines=0)
    out = []
    t0 = time.time()
    has_expl = "explanation" in head
    for i, ch in enumerate(read_chunks(path, head, names, rec)):
        rec["rows"] += len(ch)
        rec["prompt_is_1"] += int((ch.prompt == "1").sum())
        if has_expl:
            bad = ch.group.isna() & ch.thermometer.fillna("").str.contains(SHIFT_RE)
            sh = ch.loc[bad, ["respID"]].copy()
            sh["group"] = ch.loc[bad, "thermometer"].map(norm_group)
            sh["thermometer"] = ch.loc[bad, "explanation"]
            sh["draw"] = ch.loc[bad, "temperature"]
            sh["temperature"] = ch.loc[bad, "prompt"]
            sh["prompt"] = ch.loc[bad, "index"]
            f1 = sh[sh.thermometer.notna()]
            rest = sh[sh.thermometer.isna()]
            m = rest.group.fillna("").str.match(FIX2_RE.pattern)
            f2 = rest[m].copy()
            parts = f2.group.str.extract(FIX2_RE.pattern)
            f2["group"], f2["thermometer"] = parts[0], parts[1]
            rec["fixed1"] += len(f1)
            rec["fixed2"] += len(f2)
            good = ch.loc[~ch.index.isin(f1.index) & ~ch.index.isin(f2.index),
                          ["respID", "group", "thermometer", "draw", "temperature", "prompt"]]
            good = good.assign(group=good.group.map(norm_group))
            ch = pd.concat([good.assign(fix=0), f1.assign(fix=1), f2.assign(fix=2)])
            ch["group"] = ch.group.map(norm_group)  # the authors normalise every row once more
        else:
            ch = ch[["respID", "group", "thermometer", "draw", "temperature", "prompt"]].assign(fix=0)
            ch = ch.assign(group=ch.group.map(norm_group))
        out.append(ch)
        print(f"chunk {i} rows {rec['rows']:,} {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(out, ignore_index=True)
    rec["resCleaned_rows_all_groups"] = len(res)
    res = res[res.thermometer.notna() & (res.prompt != "1") & res.group.isin(GROUPS) & res.respID.isin(keep_ids)]
    res = res.assign(prompt=res.prompt.str.replace(r",\d+$", "", regex=True))
    res["therm"] = res.thermometer.map(to_therm)
    res["framing"] = res.prompt.map(prompt_simple)
    # the authors' left_join drops the regex-repaired rows (fixed2 lacks respID), so their counts exclude them
    rec["rows_fix2_in_analysis_scope"] = int((res.fix == 2).sum())
    cells = res[res.fix != 2].groupby(["respID", "framing", "group"]).size()
    cells.rename("n").reset_index().to_parquet(os.path.join(DER, f"{tag}_cellcounts.parquet"), index=False)
    rec["rows_after_filters"] = len(res)
    rec["cells"] = len(cells)
    rec["cell_min_draws"] = int(cells.min())
    rec["therm_nonnumeric_dropped"] = int(res.therm.isna().sum())
    res = res[res.therm.notna()]
    res = res.assign(draw=pd.to_numeric(res.draw, errors="coerce"),
                     temperature=pd.to_numeric(res.temperature, errors="coerce"))
    rec["rows_numeric"] = len(res)
    rec["therm_out_of_0_100"] = int(((res.therm < 0) | (res.therm > 100)).sum())
    rec["temperature_values"] = ",".join(str(v) for v in sorted(res.temperature.dropna().unique()))
    keep = res[["respID", "group", "framing", "draw", "temperature", "therm", "fix"]].reset_index(drop=True)
    keep.to_parquet(os.path.join(DER, f"{tag}_long.parquet"), index=False)
    r = pd.DataFrame([rec]).T.reset_index()
    r.columns = ["quantity", "value"]
    res_dir = os.path.join(HERE, "..", "results", "bisbee")
    os.makedirs(res_dir, exist_ok=True)
    r.to_csv(os.path.join(res_dir, f"parse_receipts_{tag}.csv"), index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    parse(sys.argv[1] if len(sys.argv) > 1 else "rr1")
