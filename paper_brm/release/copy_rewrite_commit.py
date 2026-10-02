"""Copy the 2 Oct 2026 manuscript rewrite from UpdatedRun into the public clone.

Scope is limited to the manuscript, the figure and receipt scripts it changed, and the new
external level-2 summary; curated public files (READMEs, analysis notes) are never touched.
Excluded: manuscript/_pre_rewrite_2026-10-02 (local backup) and figures/fig2_options (sketches).
Run with --dry to list what would change."""
import filecmp
import os
import shutil
import subprocess
import sys

SRC = r"C:\Research\PsychBench\UpdatedRun"
DST = r"C:\Research\_repo_work\vl"
ROOTS = ["paper_brm/manuscript"]
FILES = ["scripts/84_fig_level2_brm.py", "scripts/84c_fig_level2_grid.py", "scripts/86_receipts.py",
         "analysis/brm/86_receipts_check.csv", "analysis/brm/86_receipts_summary.txt",
         "paper_brm/external/scripts/l2_threeway.py",
         "paper_brm/external/results/l2_threeway_summary.csv",
         "paper_brm/external/results/l2_threeway_rows.csv",
         "paper_brm/release/copy_rewrite_commit.py"]
SKIP = ("_pre_rewrite_2026-10-02", "fig2_options", "REFS_KEYMAP.md",".aux", ".log", ".blg", ".out", ".bbl", "_verify")
dry = "--dry" in sys.argv

cands = list(FILES)
for root in ROOTS:
    for dp, dn, fn in os.walk(os.path.join(SRC, root)):
        rel_dir = os.path.relpath(dp, SRC).replace("\\", "/")
        if any(s in rel_dir for s in SKIP):
            continue
        for f in fn:
            rel = f"{rel_dir}/{f}"
            if not any(s in rel for s in SKIP):
                cands.append(rel)

changed = []
for rel in sorted(set(cands)):
    s, d = os.path.join(SRC, rel), os.path.join(DST, rel)
    if not os.path.isfile(s):
        continue
    if os.path.isfile(d) and filecmp.cmp(s, d, shallow=False):
        continue
    changed.append(("new " if not os.path.isfile(d) else "mod ") + rel)
    if not dry:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
print("\n".join(changed))
print(len(changed), "files", "(dry run)" if dry else "copied")
if not dry:
    print(subprocess.run(["git", "-C", DST, "status", "--short"], capture_output=True, text=True).stdout)
