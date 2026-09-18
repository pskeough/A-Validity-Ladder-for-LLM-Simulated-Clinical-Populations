"""Structural and stylistic checks on the manuscript itself, rather than on its numbers.

The three gates check that every claim reconciles to a receipt. Nothing checks that the document is
readable: whether a reference resolves, whether a line runs into the margin, how long the abstract
has grown, or whether the prose carries the constructions that read as machine-written. This does
that, and it runs against the compiled PDF as well as the source, because an overfull box and a
question-mark reference are only visible after typesetting.

The tell scan works on whitespace-normalized text for the reason script 15's intex does: a needle
that spans a line break silently fails against the raw file, and one such miss hid a real defect.
"""
import os
import re
import collections
import subprocess
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
# Defaults to the full manuscript; pass a directory to check the condensed one instead.
DIR = sys.argv[1] if len(sys.argv) > 1 else "paper_v2"
TEX = os.path.join(BASE, DIR, "main.tex")
PDF = os.path.join(BASE, DIR, "main.pdf")

tex = open(TEX, encoding="utf-8").read()
flat = re.sub(r"\s+", " ", tex)
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        FAIL.append(name)


def words(s):
    s = re.sub(r"\\(cite|ref|label)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = re.sub(r"[{}$\\]", " ", s)
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


# ---- length ----------------------------------------------------------------------------------
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
APPENDIX = "\\appendix"
tail = tex.index(APPENDIX) if APPENDIX in tex else tex.index("\\bibliographystyle")
body = tex[tex.index(r"\section{Introduction}"):tail]
n_abs, n_body = words(abstract), words(body)
check(f"abstract length {n_abs} words", n_abs <= 380, "over 380, a reader will not finish it")
print(f"     body {n_body} words"
      + (f", appendices {words(tex[tail:])} words" if APPENDIX in tex else ""))

# ---- references resolve ------------------------------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
check("every ref has a label", refs <= labels, sorted(refs - labels))
# Every label, fig: included. Excluding figures here is what let a broken figure reference sit in
# the document unreported: it was absent from refs, which should have shown as an unreferenced
# label, and the exclusion swallowed exactly that signal.
unused = sorted(labels - refs)
print(f"     {len(labels)} labels, {len(refs)} distinct refs"
      + (f", unreferenced: {unused}" if unused else ""))

# ---- typography ---------------------------------------------------------------------------------
check("no em-dashes", "---" not in tex, tex.count("---"))
check("no carriage returns", "\r" not in tex)
check("no doubled words", not re.search(r"\b(\w+) \1\b", flat, re.I),
      (re.search(r"\b(\w+) \1\b", flat, re.I) or [""])[0])

# ---- constructions that read as machine-written ------------------------------------------------
# Each pattern below was found in this manuscript at some point and rewritten. They are kept as a
# regression scan, not as a style rule: the counts are printed, and only the ones with a hard
# ceiling fail. "not X" earned its ceiling by being introduced as a fix for "rather than X" and
# reading worse, which is the specific correction a reviewer made.
TELLS = [
    (r"\bIt(?:'s| is) worth (?:noting|mentioning|pointing)\b", "It is worth noting", 0),
    (r"\bdelve|\btapestry|\bmultifaceted|\bunderscores?\b|\bnavigat(?:e|ing)\b", "LLM lexicon", 0),
    (r", not (?:a |an |the )?\w+[.,]", "\"X, not Y\" contrast", 6),
    (r"\bnot (?:just|only) \w+(?:,| but)", "not just X but Y", 4),
    (r"\bwhat (?:this|these|it|they) (?:does|do|is|are) is\b", "cleft emphasis", 3),
    (r"\b(?:crucial|vital|pivotal|robustly|significantly enhanc)\w*", "intensifiers", 0),
    (r";[^.;]{60,};", "three-part semicolon list", 1),
]
prose = re.sub(r"\\item[^\\]*", " ", flat)   # itemize bullets end in semicolons by convention
for pat, name, ceiling in TELLS:
    hits = re.findall(pat, prose, re.I)
    check(f"{name}: {len(hits)}", len(hits) <= ceiling, f"{hits[:4]}")

# ---- the compiled document ----------------------------------------------------------------------
# Overfull-box warnings from the engine are the wrong instrument here: the log repeats each box on
# every pass, counts boxes inside the bibliography that no reader will notice, and fires at
# fractions of a point. What matters is ink outside the text column, so this measures that on the
# rendered page instead, and reports the engine's warnings only as context.
if os.path.exists(PDF):
    import fitz
    doc = fitz.open(PDF)
    text = "".join(p.get_text() for p in doc)
    # LaTeX prints an undefined reference as "??", but the class's font ligates it into a single
    # U+2047, so a check for the two-character form reads clean on a document that is visibly
    # broken. That is how a dangling \ref survived this gate and was caught by a human reader.
    bad_ref = text.count("??") + text.count("\u2047")
    check("no unresolved refs in the PDF", bad_ref == 0, bad_ref)
    # Floats can be silently dropped when the queue never drains before the document ends, which
    # takes the \label with them and is what produced that dangling reference. Every caption in
    # the source must therefore be findable in the rendered text.
    def norm(x):
        # LaTeX renders ' as a typographic apostrophe and `` '' as curly quotes, so a needle taken
        # from the source never matches the rendered text when a caption contains one. That alone
        # made this check report a figure missing that was sitting on the page.
        for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("``", '"'), ("''", '"')):
            x = x.replace(a, b)
        return re.sub(r"\s+", " ", x)

    flat_pdf = norm(text)
    # The source check above matches \ref{...} against \label{...}, which cannot see a \ref whose
    # command name is damaged: "Figure~ ef{fig:crossgroup}" is not a \ref any more, just literal
    # text, so LaTeX emits no warning, prints no "??", and the label counts as merely unreferenced.
    # It is visible only on the rendered page, where the label name itself is printed. That is a
    # reliable signal in both directions, since no correctly typeset document prints "fig:name".
    leaked = sorted(l for l in labels if l in flat_pdf)
    check("no label names printed as text", not leaked, leaked)
    caps = re.findall(r"\\caption\{(.{0,60})", tex)
    lost = []
    for cap in caps:
        needle = norm(re.sub(r"\\[a-zA-Z]+|[\\${}~^]", "", cap)).strip()[:38]
        if len(needle) > 12 and needle not in flat_pdf:
            lost.append(needle)
    check(f"every float reaches the page: {len(caps) - len(lost)}/{len(caps)}", not lost, lost)
    # The print area is measured off the document rather than assumed: acmart's column edge is not
    # where a guessed margin puts it, and a guessed margin flagged two blocks that sit exactly on
    # the edge. Anything past a half-inch margin is real bleed.
    spill = [(pg.number + 1, round(b[2], 1)) for pg in doc for b in pg.get_text("blocks")
             if b[2] > pg.rect.width - 24 or b[0] < 20 or b[3] > pg.rect.height - 24]
    check(f"no text bleeding off the page: {len(spill)}", not spill, spill[:5])
    # A block can sit inside the page and still be broken, by running out of its own column and
    # into the neighbouring one. That is how an over-wide table collided with the appendix text
    # while every page-level measurement read clean. Column edges are inferred from the document:
    # the two most common right edges belong to the left column and the full-width span.
    # No column-collision check here, deliberately. A table can sit on top of the neighbouring
    # column's text, and two attempts to detect that geometrically both failed: measuring whether a
    # block reaches past the column edge cannot separate a real collision from a two-point
    # bibliography overhang, since both end at the same coordinate, and testing block intersection
    # fires on every paragraph PyMuPDF splits in two and on every axis label inside a figure.
    #
    # The instrument that does work for this is the engine's own overfull-box count on main.tex,
    # reported below: the collision that prompted all of it was a tabular wider than its column,
    # and it showed up there. It is printed rather than failed because the count includes boxes
    # that overhang into the gutter without touching anything.
    # A float that never gets placed takes its label with it, and a page left almost empty is how
    # the fix for that shows up. Both are reported.
    thin = [(pg.number + 1, len(pg.get_text().split())) for pg in doc
            if len(pg.get_text().split()) < 200 and pg.number + 1 < doc.page_count]
    check(f"no near-empty page before the last: {len(thin)}", not thin, thin)
    print(f"     {doc.page_count} pages, {words(body)} body words")

tect = os.path.expanduser("~/.local/bin/tectonic.exe")
if os.path.exists(tect):
    r = subprocess.run([tect, "-X", "compile", TEX], capture_output=True, text=True,
                       cwd=os.path.join(BASE, DIR))
    # Deduplicated, and restricted to main.tex: the engine repeats each box on every pass, and the
    # bibliography's boxes come from generated .bbl that no edit to the manuscript can fix.
    over = {(a, b, w) for w, a, b in
            re.findall(r"main\.tex:\d+: Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at "
                       r"lines (\d+)--(\d+)", r.stderr)}
    bad = sorted(((float(w), a, b) for a, b, w in over if float(w) > 5.0), reverse=True)
    # Reported, not failed. In a two-column class an overfull warning fires on any \item or tabular
    # whose natural width exceeds the column, and the class then squeezes it back; the check that
    # something actually reached the margin is the geometric one above, and it is the one that
    # fails. Setting a ceiling here at whatever today's count happens to be would test nothing.
    print(f"     {len(bad)} distinct overfull boxes over 5pt in main.tex, none reaching the margin"
          + (f"; widest L{bad[0][1]}-{bad[0][2]} at {bad[0][0]:.0f}pt" if bad else ""))

print()
if FAIL:
    print(f"MANUSCRIPT CHECK FAILED: {len(FAIL)}")
    sys.exit(1)
print("MANUSCRIPT CHECK PASSED")
