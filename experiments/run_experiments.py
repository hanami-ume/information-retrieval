#!/usr/bin/env python3
"""Reproduce every number used in the report.

    python experiments/run_experiments.py

Writes experiments/results/experiments.json and the plain-text top-10 listings
experiments/results/<Qn>_<model>.txt. Builds index/tdt3.idx (stemmed) and
index/tdt3_nostem.idx (ablation) if they do not exist.
"""
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from irsearch.analyzer import Analyzer                  # noqa: E402
from irsearch.corpus import read_text                   # noqa: E402
from irsearch.engine import SearchEngine                # noqa: E402
from irsearch.index import InvertedIndex                # noqa: E402
from irsearch.porter import stem as my_stem             # noqa: E402
from irsearch.query import parse_query                  # noqa: E402
from irsearch.scoring import phrase_postings            # noqa: E402
from irsearch.snippet import render                     # noqa: E402

DATA = os.path.join(ROOT, "tdt3")
OUT = os.path.join(HERE, "results")
QUERIES = [
    ("Q1", "hurricane"),
    ("Q2", "mitch george"),
    ("Q3", "bill clinton israel"),
    ("Q4", '"newt gingrich" down'),
    ("Q5", "nba strike closed-door"),
]
MODELS = ["bm25", "tfidf"]


def get_index(path, stemming):
    if not os.path.exists(path):
        idx = InvertedIndex.build(DATA, Analyzer(stemming=stemming))
        idx.save(path)
    t0 = time.perf_counter()
    idx = InvertedIndex.load(path, data_root=DATA)
    load = time.perf_counter() - t0
    return idx, load


def index_stats(idx, path, load):
    lens = sorted(idx.doc_len)
    n_postings = sum(p.df for p in idx.postings.values())
    singletons = sum(1 for p in idx.postings.values() if sum(p.tfs) == 1)
    return {
        "documents": idx.num_docs,
        "vocabulary": len(idx.postings),
        "tokens": idx.num_tokens,
        "postings": n_postings,
        "hapax_terms": singletons,
        "avg_doc_len": round(idx.avg_doc_len, 1),
        "median_doc_len": lens[len(lens) // 2],
        "max_doc_len": lens[-1],
        "empty_docs": sum(1 for x in lens if x == 0),
        "build_seconds": round(idx.build_seconds, 1),
        "load_seconds": round(load, 2),
        "file_mib": round(os.path.getsize(path) / 2**20, 1),
    }


def doc_contains_units(idx, d, q):
    """True if document d contains every term unit and phrase of q."""
    for t in q.terms:
        p = idx.get(t)
        if p is None or d not in set(p.docs):
            return False
    for ph in q.phrases:
        if d not in phrase_postings(idx, ph):
            return False
    return True


def timed(engine, text, model, runs=20, **kw):
    ts = []
    res = None
    for _ in range(runs):
        res = engine.search(text, model=model, hits=10, **kw)
        ts.append(res.seconds)
    return res, statistics.median(ts)


def listing(res, label):
    lines = [f"{label} = {res.query.raw} (took {res.seconds:.3f} seconds)", ""]
    for h in res.hits:
        lines.append(f"{h.rank:02d} [{h.score:.4f}] {h.docid}")
        lines.append(render(h.snippet, width=80))
        lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {}
    R["environment"] = {"python": platform.python_version(), "platform": platform.platform()}

    # ---------------------------------------------------------------- index
    stem_path = os.path.join(ROOT, "index", "tdt3.idx")
    nostem_path = os.path.join(ROOT, "index", "tdt3_nostem.idx")
    idx, load = get_index(stem_path, True)
    idx_ns, load_ns = get_index(nostem_path, False)
    R["index"] = {"stem": index_stats(idx, stem_path, load),
                  "nostem": index_stats(idx_ns, nostem_path, load_ns)}
    R["doctypes"] = dict(Counter(idx.doc_types))
    src_of = [re.match(r"[A-Z]+", n).group(0) for n in idx.doc_names]
    R["sources"] = dict(Counter(src_of))
    by_src = {}
    for s, l in zip(src_of, idx.doc_len):
        by_src.setdefault(s, []).append(l)
    R["source_len"] = {s: {"docs": len(v), "median": statistics.median(v),
                           "mean": round(statistics.mean(v), 1)} for s, v in sorted(by_src.items())}
    print(json.dumps(R["index"], indent=1))

    # ------------------------------------------- Porter vs NLTK (original)
    try:
        from nltk.stem.porter import PorterStemmer
        ref = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
        words = [w for w in idx_ns.postings if w.isalpha()]
        diff = [(w, my_stem(w), ref.stem(w)) for w in words if my_stem(w) != ref.stem(w)]
        R["porter_check"] = {"words": len(words), "mismatches": len(diff), "examples": diff[:20]}
        print("porter check:", len(words), "words,", len(diff), "mismatches")
    except ImportError:
        R["porter_check"] = None

    # --------------------------------------------------- the five queries
    engine = SearchEngine(idx)
    engine_ns = SearchEngine(idx_ns)
    corpus_avg = idx.avg_doc_len
    R["queries"] = {}
    for qid, text in QUERIES:
        q = parse_query(text, idx.analyzer)
        entry = {"text": text, "terms": dict(q.terms), "phrases": [list(p) for p in q.phrases],
                 "df": {t: idx.get(t).df if idx.get(t) else 0 for t in q.terms},
                 "phrase_df": {" ".join(p): len(phrase_postings(idx, p)) for p in q.phrases}}
        for model in MODELS:
            res, med = timed(engine, text, model)
            _, med_ns = timed(engine, text, model, snippets=False)
            with open(os.path.join(OUT, f"{qid}_{model}.txt"), "w") as fh:
                fh.write(listing(res, qid.lower()))
            docs = [h.doc for h in res.hits]
            entry[model] = {
                "matched": res.matched,
                "latency_ms": round(med * 1000, 2),
                "latency_ms_no_snippet": round(med_ns * 1000, 2),
                "top10": [(h.docid, round(h.score, 4), idx.doc_len[h.doc]) for h in res.hits],
                "mean_len_top10": round(statistics.mean(idx.doc_len[d] for d in docs), 1),
                "all_units_top10": sum(doc_contains_units(idx, d, q) for d in docs),
                "sources_top10": dict(Counter(src_of[d] for d in docs)),
            }
        a = {d for d, _, _ in entry["bm25"]["top10"]}
        b = {d for d, _, _ in entry["tfidf"]["top10"]}
        entry["overlap_bm25_tfidf"] = len(a & b)
        # stemming ablation (BM25)
        r_ns = engine_ns.search(text, model="bm25", hits=10, snippets=False)
        q_ns = parse_query(text, idx_ns.analyzer)
        entry["nostem"] = {
            "matched": r_ns.matched,
            "df": {t: idx_ns.get(t).df if idx_ns.get(t) else 0 for t in q_ns.terms},
            "overlap_with_stem": len(a & {h.docid for h in r_ns.hits}),
            "top10": [h.docid for h in r_ns.hits],
        }
        R["queries"][qid] = entry
        print(qid, {m: (entry[m]["matched"], entry[m]["latency_ms"]) for m in MODELS},
              "overlap", entry["overlap_bm25_tfidf"])
    R["corpus_avg_len"] = round(corpus_avg, 1)

    # surface variants that stemming conflates for the query words
    variants = {}
    for qid, text in QUERIES:
        for w in re.findall(r"[a-z]+", text.lower()):
            s = my_stem(w)
            forms = sorted(t for t in idx_ns.postings if t.isalpha() and my_stem(t) == s)
            variants[w] = {"stem": s, "forms": [(f, idx_ns.get(f).df) for f in forms][:12]}
    R["stem_variants"] = variants

    # ------------------------------------------------------------- BM25 b
    R["b_sweep"] = {}
    for bval in [0.0, 0.25, 0.5, 0.75, 1.0]:
        per_q = {}
        for qid, text in QUERIES:
            res = engine.search(text, model="bm25", hits=10, snippets=False, k1=1.5, b=bval)
            per_q[qid] = round(statistics.mean(idx.doc_len[h.doc] for h in res.hits), 1)
        R["b_sweep"][str(bval)] = per_q
    print("b sweep:", R["b_sweep"])

    # ----------------------------------------------- soft vs strict phrase
    R["phrase_mode"] = {}
    text = QUERIES[3][1]
    q = parse_query(text, idx.analyzer)
    for model in MODELS:
        soft = engine.search(text, model=model, hits=10, snippets=False)
        strict = engine.search(text, model=model, hits=10, snippets=False, strict_phrase=True)
        ph = set(phrase_postings(idx, q.phrases[0]))
        R["phrase_mode"][model] = {
            "phrase_docs": len(ph),
            "soft_top10_with_phrase": sum(h.doc in ph for h in soft.hits),
            "soft_matched": soft.matched, "strict_matched": strict.matched,
            "overlap": len({h.doc for h in soft.hits} & {h.doc for h in strict.hits}),
            "soft_top10_with_down": sum(h.doc in set(idx.get("down").docs) for h in soft.hits),
        }
    # documents that contain "newt" and "gingrich" but not adjacent
    nd = set(idx.get("newt").docs) & set(idx.get("gingrich").docs)
    R["phrase_mode"]["newt_and_gingrich_docs"] = len(nd)

    # -------------------------------------------------- hyphen handling Q5
    R["hyphen_mode"] = {}
    text = QUERIES[4][1]
    nba = set(idx.get("nba").docs)
    cd = set(phrase_postings(idx, ("close", "door")))
    for model in MODELS:
        for mode in ["phrase+parts", "phrase", "parts"]:
            res = engine.search(text, model=model, hits=10, snippets=False, hyphen_mode=mode)
            R["hyphen_mode"][f"{model}/{mode}"] = {
                "top10_nba": sum(h.doc in nba for h in res.hits),
                "top10_closed_door": sum(h.doc in cd for h in res.hits),
                "top3": [h.docid for h in res.hits[:3]],
            }
    R["hyphen_mode"]["nba_and_closed_door_docs"] = [idx.doc_names[d] for d in sorted(nba & cd)]
    R["hyphen_mode"]["nba_docs"] = len(nba)
    R["hyphen_mode"]["nba_and_strike_docs"] = len(nba & set(idx.get("strike").docs))
    R["hyphen_mode"]["nba_and_lockout_docs"] = len(nba & set(idx.get("lockout").docs))
    print("hyphen:", json.dumps(R["hyphen_mode"], indent=1))

    with open(os.path.join(OUT, "experiments.json"), "w") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
    print("written", os.path.join(OUT, "experiments.json"))


if __name__ == "__main__":
    main()
