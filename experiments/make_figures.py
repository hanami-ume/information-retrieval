#!/usr/bin/env python3
"""Charts for the report, drawn from experiments/results/experiments.json.

    python experiments/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib import font_manager               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "report", "figures")

# reference categorical palette, fixed slot order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 180, "savefig.bbox": "tight",
})


def load():
    with open(os.path.join(HERE, "results", "experiments.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fig_doclen(R):
    from irsearch.index import InvertedIndex
    idx = InvertedIndex.load(os.path.join(ROOT, "index", "tdt3.idx"))
    lens = [x for x in idx.doc_len if x > 0]
    import numpy as np
    bins = np.logspace(0, np.log10(max(lens)) + 0.05, 50)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(lens, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    mean, med = R["index"]["stem"]["avg_doc_len"], R["index"]["stem"]["median_doc_len"]
    top = ax.get_ylim()[1]
    for x, lab, ha, k in ((med, f"中位数 {med} ", "right", 0.96), (mean, f" 均值 {mean:.0f}", "left", 1.04)):
        ax.axvline(x, color=INK2, linestyle="--", linewidth=1)
        ax.text(x * k, top * 0.985, lab, color=INK, fontsize=9.5, va="top", ha=ha,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("文档长度（词项数，对数坐标）")
    ax.set_ylabel("文档数")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "fig_doclen_hist.png"))
    plt.close(fig)


def fig_top10_len(R):
    qs = list(R["queries"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, (model, color, label, dx, marker) in enumerate(
            (("bm25", BLUE, "BM25", -0.17, "o"), ("tfidf", ORANGE, "TF-IDF (lnc.ltc)", 0.17, "s"))):
        for i, q in enumerate(qs):
            ys = [max(1, l) for _, _, l in R["queries"][q][model]["top10"]]
            ax.scatter([i + dx] * len(ys), ys, s=34, color=color, marker=marker,
                       edgecolor="white", linewidth=1.2, zorder=3,
                       label=label if i == 0 else None)
            m = R["queries"][q][model]["mean_len_top10"]
            ax.hlines(m, i + dx - 0.12, i + dx + 0.12, color=INK, linewidth=2, zorder=4)
    mean = R["index"]["stem"]["avg_doc_len"]
    ax.axhline(mean, color=INK2, linestyle="--", linewidth=1, zorder=1)
    ax.text(1.68, mean * 1.08, f"语料平均长度 {mean:.0f}", ha="right", color=INK, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(qs)))
    wrapped = {"Q4": '"newt gingrich"\ndown', "Q5": "nba strike\nclosed-door",
               "Q3": "bill clinton\nisrael", "Q2": "mitch\ngeorge"}
    ax.set_xticklabels([f"{q}\n{wrapped.get(q, R['queries'][q]['text'])}" for q in qs], fontsize=9)
    ax.set_ylabel("Top-10 文档长度（词项数，对数坐标）")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5)
    ax.text(0.995, 0.02, "黑色横线：该查询 Top-10 的平均长度", transform=ax.transAxes,
            ha="right", color=INK2, fontsize=8.5)
    fig.savefig(os.path.join(OUT, "fig_top10_doclen.png"))
    plt.close(fig)


def fig_b_sweep(R):
    sweep = R["b_sweep"]
    bs = sorted(sweep, key=float)
    qs = list(R["queries"])
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for q, c, mk in zip(qs, colors, markers):
        ys = [sweep[b][q] for b in bs]
        ax.plot([float(b) for b in bs], ys, color=c, linewidth=2, marker=mk, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, label=q)
    # direct labels at the right end, nudged apart
    ends = sorted(((sweep[bs[-1]][q], q) for q in qs))
    placed = []
    for y, q in ends:
        yy = y
        for p in placed:
            if abs(yy / p - 1) < 0.18:
                yy = p * 1.2
        placed.append(yy)
        ax.text(1.03, yy, q, va="center", color=INK, fontsize=9.5)
    mean = R["index"]["stem"]["avg_doc_len"]
    ax.axhline(mean, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.0, mean * 1.07, f"语料平均长度 {mean:.0f}", color=INK, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.04, 1.12)
    ax.set_xticks([float(b) for b in bs])
    ax.set_xlabel("BM25 长度归一化参数 b（k1 = 1.5）")
    ax.set_ylabel("Top-10 平均文档长度（对数坐标）")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9.5)
    fig.savefig(os.path.join(OUT, "fig_b_sweep.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = load()
    fig_doclen(R)
    fig_top10_len(R)
    fig_b_sweep(R)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
