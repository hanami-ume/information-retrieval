#!/usr/bin/env python3
"""Render the report's display equations and code listings as PNG images
for use in a Word version of the report.

    python3 report/word/make_word_assets.py

Reads report/main.tex, writes report/word/png/*.png (300 dpi). Needs xelatex
and pdftoppm. Captions and equation numbers are deliberately left out so
they can be added with Word's own caption/numbering features.
"""
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "main.tex")
OUT = os.path.join(HERE, "png")
TEXTWIDTH = "15.8cm"          # A4 minus 2.6 cm margins, as in the report

src = open(TEX, encoding="utf-8").read()

# reuse the report's colour definitions and listings style verbatim
colors = "\n".join(re.findall(r"\\definecolor\{code\w+\}\{HTML\}\{\w+\}", src))
lstset = re.search(r"\\lstset\{.*?\n\}", src, re.S).group(0)

PREAMBLE = r"""\documentclass[border=%s%s]{standalone}
\usepackage{fontspec}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{listings}
\setmonofont{JetBrains Mono}[Scale=0.85, Contextuals=NoAlternate]
%s
%s
\lstset{aboveskip=0pt, belowskip=0pt}
\begin{document}
%s
\end{document}
"""

jobs = []

# display equations (numbering stripped)
eqs = {
    "eq1_bm25": re.search(r"\\begin\{gather\}(.*?)\\end\{gather\}", src, re.S).group(1),
    "eq2_tfidf": re.search(r"\\begin\{equation\}(.*?)\\end\{equation\}", src, re.S).group(1),
}
for name, body in eqs.items():
    body = re.sub(r"\\label\{[^}]*\}|\\notag", "", body).strip()
    # gathered inside inline math crops tightly (display environments span the line)
    jobs.append((name, "4pt", "", f"$\\displaystyle\\begin{{gathered}}\n{body}\n\\end{{gathered}}$"))

# the inline query weight of the TF-IDF model, set as a display formula
jobs.append(("eq3_tfidf_query_weight", "4pt", "",
             r"$\displaystyle w_{q,u} = (1 + \log_{10}\mathrm{qtf}_u)\cdot\log_{10}\frac{N}{\mathrm{df}_u}$"))

# code listings (caption/label removed, other options such as gobble kept)
names = {"lst:analyzer": "code1_analyzer", "lst:build": "code2_index_build",
         "lst:query": "code3_query_parse", "lst:bm25": "code4_bm25",
         "lst:tfidf": "code5_tfidf"}
for opts, body in re.findall(r"\\begin\{lstlisting\}\[([^\]]*)\]\n(.*?)\\end\{lstlisting\}", src, re.S):
    label = re.search(r"label=\{([^}]*)\}", opts).group(1)
    keep = ", ".join(o.strip() for o in re.split(r",\s*(?![^{]*\})", opts)
                     if not o.strip().startswith(("caption", "label")))
    # a fixed-width box gives every listing the report's full text width
    jobs.append((names[label], "2pt", "",
                 f"\\begin{{minipage}}{{{TEXTWIDTH}}}\n\\begin{{lstlisting}}[{keep}]\n{body}\\end{{lstlisting}}\n\\end{{minipage}}"))

os.makedirs(OUT, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for name, border, width, body in jobs:
        tex = os.path.join(tmp, name + ".tex")
        with open(tex, "w", encoding="utf-8") as fh:
            fh.write(PREAMBLE % (border, width, colors, lstset, body))
        subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile",
                        os.path.join(tmp, name + ".pdf"), os.path.join(OUT, name)], check=True)
        print("wrote", os.path.relpath(os.path.join(OUT, name + ".png"), os.path.dirname(HERE)))
