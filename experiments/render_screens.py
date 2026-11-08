#!/usr/bin/env python3
"""Capture console screenshots of the five test queries.

Each query is typed into the interactive search shell (src/search.py) with
colour output enabled; the ANSI output is converted to HTML in a terminal-style
frame and photographed with headless Chromium (experiments/screenshot.js).

    python experiments/render_screens.py
"""
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
TMP = os.path.join(HERE, "screens_html")

QUERIES = [
    ("q1", "hurricane"),
    ("q2", "mitch george"),
    ("q3", "bill clinton israel"),
    ("q4", '"newt gingrich" down'),
    ("q5", "nba strike closed-door"),
]

STYLE = {"1": "font-weight:700", "2": "color:#8b949e",
         "1;36": "font-weight:700;color:#56d4dd", "1;33": "font-weight:700;color:#f0c74a"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#ffffff;padding:16px;display:inline-block}}
.win{{background:#1e2127;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.35);overflow:hidden;width:max-content}}
.bar{{background:#2c313a;height:26px;display:flex;align-items:center;padding:0 10px;gap:7px}}
.dot{{width:11px;height:11px;border-radius:50%}}
.title{{color:#9aa4b2;font:12px 'DejaVu Sans',sans-serif;margin-left:12px}}
pre{{margin:0;padding:12px 16px 14px;color:#d7dae0;font:13px/1.38 'JetBrains Mono','DejaVu Sans Mono',monospace;white-space:pre}}
.p{{color:#98c379;font-weight:700}}
</style></head><body><div class="win"><div class="bar">
<span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e"></span>
<span class="dot" style="background:#27c93f"></span><span class="title">{title}</span></div>
<pre>{body}</pre></div></body></html>"""


def ansi_to_html(text):
    out, open_span = [], False
    for part in re.split(r"(\x1b\[[\d;]*m)", text):
        m = re.fullmatch(r"\x1b\[([\d;]*)m", part)
        if m:
            if open_span:
                out.append("</span>")
                open_span = False
            code = m.group(1)
            if code in STYLE:
                out.append(f'<span style="{STYLE[code]}">')
                open_span = True
        else:
            out.append(html.escape(part))
    if open_span:
        out.append("</span>")
    return "".join(out)


def capture(qid, query, model):
    cmd = f"search --hits=10 --model={model} {query}" if model != "bm25" else f"search --hits=10 {query}"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "search.py"), "--color=always"],
        input=cmd + "\n", capture_output=True, text=True, check=True)
    lines = proc.stdout.splitlines()
    # drop the index-loading notice; echo the typed command after the prompt
    lines = [ln for ln in lines if not ln.startswith("(index loaded")]
    # the first prompt receives the typed command; the final prompt (EOF) is dropped
    body_lines = []
    prompted = False
    for ln in lines:
        if ln.startswith("#"):
            if not prompted:
                body_lines.append("__prompt__" + cmd)
                body_lines.append(ln[2:])
                prompted = True
            continue
        body_lines.append(ln)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    rendered = []
    for ln in body_lines:
        if ln.startswith("__prompt__"):
            rendered.append('<span class="p"># </span>' + html.escape(ln[len("__prompt__"):]))
        else:
            rendered.append(ansi_to_html(ln))
    return PAGE.format(title="python3 src/search.py", body="\n".join(rendered))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    jobs = []
    for model in ("bm25", "tfidf"):
        for qid, query in QUERIES:
            page = capture(qid, query, model)
            src = os.path.join(TMP, f"{qid}_{model}.html")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(page)
            jobs.append((src, os.path.join(OUT, f"screen_{qid}_{model}.png")))
    args = ["node", os.path.join(HERE, "screenshot.js")] + [x for j in jobs for x in j]
    subprocess.run(args, check=True)
    for _, png in jobs:
        print("wrote", os.path.relpath(png, ROOT))


if __name__ == "__main__":
    main()
