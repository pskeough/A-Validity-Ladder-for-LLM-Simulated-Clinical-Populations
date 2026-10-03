"""Build the arXiv source package for the Validity Ladder manuscript.

Copies the manuscript into arxiv/src, switches apa7 to single-spaced doc mode, strips
full-line comments and receipt tags, compiles once with tectonic to produce main.bbl
(arXiv does not run BibTeX), copies the supplement PDF to anc/, and zips the result.
Fails if any \\PATRICK{ or \\CHECK{ marker is still used in the text.

Usage: python build_arxiv.py
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parent / "manuscript"
SRC = HERE / "src"
OUT_ZIP = HERE / "validity_ladder_arxiv_2026-10-02.zip"
TECTONIC = Path.home() / ".local" / "bin" / "tectonic.exe"

FIG_FILES = ["fig1_ladder.pdf", "fig2_ladder_panels.pdf", "fig3_l2_dots.pdf",
             "fig1_ladder_caption.tex", "fig1_ladder_note.tex",
             "fig2_ladder_panels_caption.tex", "fig2_ladder_panels_note.tex",
             "fig3_l2_dots_caption.tex", "fig3_l2_dots_note.tex"]


def clean_tex(text):
    out = []
    for line in text.splitlines():
        if re.match(r"^\s*%", line):
            continue
        line = re.sub(r"\s+% R:.*$", "", line)
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / "sections").mkdir(parents=True)
    (SRC / "tables").mkdir()
    (SRC / "figures").mkdir()
    (SRC / "anc").mkdir()

    # Only files the body actually reads: 08_limitations is folded into the Discussion, and the
    # level tables moved to the supplement (tab_panel is the one table left in the body).
    for f in (MS / "sections").glob("*.tex"):
        if f.name == "08_limitations.tex":
            continue
        (SRC / "sections" / f.name).write_text(clean_tex(f.read_text(encoding="utf-8")), encoding="utf-8")
    for f in [MS / "tables" / "tab_panel.tex"]:
        (SRC / "tables" / f.name).write_text(clean_tex(f.read_text(encoding="utf-8")), encoding="utf-8")
    for name in FIG_FILES:
        f = MS / "figures" / name
        if f.suffix == ".tex":
            (SRC / "figures" / name).write_text(clean_tex(f.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(f, SRC / "figures" / name)
    shutil.copy2(MS / "refs.bib", SRC / "refs.bib")

    main_tex = clean_tex((MS / "main.tex").read_text(encoding="utf-8"))
    old = r"\documentclass[man,12pt,floatsintext,natbib]{apa7}"
    assert main_tex.count(old) == 1
    main_tex = main_tex.replace(old, r"\documentclass[doc,11pt,floatsintext,natbib]{apa7}")
    (SRC / "main.tex").write_text(main_tex, encoding="utf-8")

    # No author marker may survive into the posted text (the macro definitions in main.tex are fine).
    leftovers = []
    for f in SRC.rglob("*.tex"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\\(PATRICK|CHECK)\{", line) and not line.lstrip().startswith(r"\newcommand"):
                leftovers.append(f"{f.relative_to(SRC)}:{i}: {line[:120]}")
    if leftovers:
        print("MARKERS LEFT:\n  " + "\n  ".join(leftovers))

    r = subprocess.run([str(TECTONIC), "-X", "compile", "--keep-intermediates", "main.tex"], cwd=SRC,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = r.stdout + r.stderr
    if r.returncode != 0:
        print(log[-3000:])
        sys.exit("compile failed")
    problems = [l for l in log.splitlines() if re.search(r"warning|undefined", l, re.I) and "Fontconfig" not in l]
    print("compile warnings:", len(problems))
    for l in problems[:20]:
        print("  ", l)

    shutil.copy2(MS / "supplement" / "supplement.pdf", SRC / "anc" / "Supplement.pdf")
    shutil.copy2(SRC / "main.pdf", HERE / "main_arxiv_preview.pdf")

    # The .bbl carries the references, so refs.bib (and its process-note header) stays out of the zip.
    # With a marker left, the zip gets a NOT_READY name and no upload-named zip is left on disk.
    out_zip = OUT_ZIP.with_name(OUT_ZIP.stem + "_NOT_READY.zip") if leftovers else OUT_ZIP
    for stale in (OUT_ZIP, OUT_ZIP.with_name(OUT_ZIP.stem + "_NOT_READY.zip")):
        stale.unlink(missing_ok=True)
    keep = lambda p: p.suffix in {".tex", ".bbl", ".pdf"} and p.name != "main.pdf"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_file() and keep(p):
                z.write(p, p.relative_to(SRC).as_posix())
    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
    print(f"zip: {out_zip.name}, {len(names)} files, {out_zip.stat().st_size / 1e6:.2f} MB")
    for n in names:
        print("  ", n)
    if leftovers:
        sys.exit("markers left: not ready to upload")


if __name__ == "__main__":
    main()
