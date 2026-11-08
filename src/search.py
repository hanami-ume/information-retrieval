#!/usr/bin/env python3
"""TDT3 command-line search engine.

Build the index once:
    python src/search.py --build                 # Porter stemming (default)
    python src/search.py --build --no-stem       # ablation index

Search (one-shot; quote phrases for the shell with single quotes):
    python src/search.py --hits=27 new york san francisco
    python src/search.py '"new york city"' apple cider

Interactive mode (no query given), mirroring the assignment's syntax:
    # search --hits=27 new york san francisco
    # search "newt gingrich" down
    # quit
"""
import argparse
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from irsearch.analyzer import Analyzer           # noqa: E402
from irsearch.engine import SearchEngine          # noqa: E402
from irsearch.index import InvertedIndex          # noqa: E402
from irsearch.snippet import render               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tdt3")
DEFAULT_INDEX = os.path.join(ROOT, "index", "tdt3.idx")


def build_parser():
    ap = argparse.ArgumentParser(prog="search", description="TDT3 search engine")
    ap.add_argument("--build", action="store_true", help="build the index and exit")
    ap.add_argument("--no-stem", action="store_true", help="disable Porter stemming (with --build)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="TDT3 root directory")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="index file")
    ap.add_argument("--hits", type=int, default=10, help="number of results (top-N)")
    ap.add_argument("--model", default="bm25", choices=["bm25", "tfidf"], help="scoring function")
    ap.add_argument("--k1", type=float, default=1.5, help="BM25 k1")
    ap.add_argument("--b", type=float, default=0.75, help="BM25 b")
    ap.add_argument("--strict-phrase", action="store_true",
                    help="require quoted phrases to occur (default: soft matching)")
    ap.add_argument("--hyphen", default="phrase", choices=["phrase", "phrase+parts", "parts"],
                    help="how a hyphenated query word is matched (default: phrase)")
    ap.add_argument("--color", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("query", nargs="*", help="query; omit for interactive mode")
    return ap


def do_build(args):
    analyzer = Analyzer(stemming=not args.no_stem)
    print(f"indexing {args.data} (stemming={'on' if analyzer.stemming else 'off'}) ...")
    idx = InvertedIndex.build(args.data, analyzer,
                              progress=lambda n: print(f"  {n} documents", flush=True))
    t0 = time.perf_counter()
    idx.save(args.index)
    print(f"documents   : {idx.num_docs}")
    print(f"vocabulary  : {len(idx.postings)} terms")
    print(f"tokens      : {idx.num_tokens}")
    print(f"avg doc len : {idx.avg_doc_len:.1f}")
    print(f"build time  : {idx.build_seconds:.1f} s (+{time.perf_counter() - t0:.1f} s to save)")
    print(f"index file  : {args.index} ({os.path.getsize(args.index) / 2**20:.1f} MiB)")


def print_result(res, label, model, color):
    def c(code, s):
        return f"\033[{code}m{s}\033[0m" if color else s

    print(c("1", f"{label} = {res.query.raw} (took {res.seconds:.3f} seconds)"))
    print(c("2", f"[{model}, {res.matched} matching documents]"))
    print()
    width = max(2, len(str(len(res.hits))))
    for h in res.hits:
        print(f"{h.rank:0{width}d} [{h.score:.4f}] {c('1;36', h.docid)}")
        print(render(h.snippet, width=80, color=color))
        print()
    if not res.hits:
        print("no matching documents\n")


def run_query(engine, args, text, label, color):
    params = {"k1": args.k1, "b": args.b} if args.model == "bm25" else {}
    res = engine.search(text, model=args.model, hits=args.hits,
                        strict_phrase=args.strict_phrase, hyphen_mode=args.hyphen, **params)
    print_result(res, label, args.model, color)


def split_options(line):
    """Split leading --options from the raw query text, keeping quotes intact."""
    opts, rest = [], line.strip()
    while rest.startswith("--"):
        head, _, tail = rest.partition(" ")
        opts.append(head)
        rest = tail.lstrip()
        if "=" not in head and head in ("--hits", "--model", "--k1", "--b", "--hyphen") and rest:
            val, _, tail = rest.partition(" ")
            opts.append(val)
            rest = tail.lstrip()
    return opts, rest


def interactive(engine, args, color):
    print(f"TDT3 search engine, {engine.index.num_docs} documents indexed.")
    print("usage: search [--hits=N] [--model=bm25|tfidf] <query>   (quit to exit)")
    n = 0
    while True:
        try:
            line = input("# ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line.split()[0] == "search":
            line = line[len("search"):]
        opts, text = split_options(line)
        try:
            local = build_parser().parse_args(opts, namespace=argparse.Namespace(**vars(args)))
        except SystemExit:
            continue
        if not text:
            continue
        n += 1
        run_query(engine, local, text, f"q{n}", color)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.build:
        do_build(args)
        return
    if not os.path.exists(args.index):
        sys.exit(f"index not found: {args.index}\nbuild it first: python src/search.py --build")
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    t0 = time.perf_counter()
    index = InvertedIndex.load(args.index, data_root=args.data)
    load_s = time.perf_counter() - t0
    engine = SearchEngine(index)
    if args.query:
        run_query(engine, args, " ".join(args.query), "q1", color)
    else:
        print(f"(index loaded in {load_s:.1f} s)")
        interactive(engine, args, color)


if __name__ == "__main__":
    main()
