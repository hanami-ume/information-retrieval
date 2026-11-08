"""Relevance scoring functions, implemented by hand on top of the index.

A query is evaluated term-at-a-time over "units". A unit is either a single
term or a phrase; a phrase behaves like a pseudo-term whose document frequency
and in-document frequency are obtained by positional intersection.

BM25 (Robertson & Zaragoza):
    score(d,q) = sum_u  qtf_u * IDF(u) * tf(u,d)*(k1+1) / (tf(u,d) + k1*(1-b+b*|d|/avgdl))
    IDF(u)     = ln(1 + (N - df_u + 0.5) / (df_u + 0.5))

TF-IDF vector space model, SMART lnc.ltc (Manning et al., IIR ch. 6):
    query  weight  w_q(u)   = (1 + log10 qtf_u) * log10(N / df_u),  cosine-normalised
    doc    weight  w_d(u)   = (1 + log10 tf(u,d)) / ||d||,  ||d|| over all terms of d
    score(d,q)              = sum_u  w_q(u) * w_d(u)
"""
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .index import InvertedIndex
from .query import Query

Unit = Tuple[str, ...]          # ("term",) or ("w1", "w2", ...)


def phrase_postings(index: InvertedIndex, phrase: Tuple[str, ...]) -> Dict[int, int]:
    """doc id -> number of occurrences of the exact phrase."""
    plist = [index.get(t) for t in phrase]
    if any(p is None for p in plist):
        return {}
    # candidate documents: those containing every word (start from rarest)
    order = sorted(range(len(plist)), key=lambda i: plist[i].df)
    cand = set(plist[order[0]].docs)
    for i in order[1:]:
        cand.intersection_update(plist[i].docs)
        if not cand:
            return {}
    pos_maps = [p.positions_by_doc() for p in plist]
    result = {}
    for d in cand:
        first = pos_maps[0][d]
        others = [set(pos_maps[i][d]) for i in range(1, len(phrase))]
        n = 0
        for p in first:
            if all((p + i) in others[i - 1] for i in range(1, len(phrase))):
                n += 1
        if n:
            result[d] = n
    return result


def _units(index: InvertedIndex, query: Query, phrase_weight: float):
    """Yield (unit, qtf_weight, docs, tfs) for each unit with non-zero df."""
    for term, qtf in query.terms.items():
        p = index.get(term)
        if p is not None:
            yield (term,), float(qtf), p.docs, p.tfs
    for phrase in query.phrases:
        pp = phrase_postings(index, phrase)
        if pp:
            docs = sorted(pp)
            yield phrase, phrase_weight, docs, [pp[d] for d in docs]


class Scorer:
    name = "base"

    def __init__(self, phrase_weight: float = 1.0):
        self.phrase_weight = phrase_weight

    def score(self, index: InvertedIndex, query: Query,
              restrict: Optional[Set[int]] = None) -> Dict[int, float]:
        raise NotImplementedError


class BM25(Scorer):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, phrase_weight: float = 1.0):
        super().__init__(phrase_weight)
        self.k1, self.b = k1, b
        self._norm_cache = {}

    def _length_norm(self, index: InvertedIndex) -> List[float]:
        key = id(index)
        if key not in self._norm_cache:
            avg = index.avg_doc_len
            k1, b = self.k1, self.b
            self._norm_cache[key] = [k1 * (1 - b + b * dl / avg) for dl in index.doc_len]
        return self._norm_cache[key]

    def score(self, index, query, restrict=None):
        N = index.num_docs
        K = self._length_norm(index)
        k1p1 = self.k1 + 1
        acc = defaultdict(float)
        for unit, qw, docs, tfs in _units(index, query, self.phrase_weight):
            df = len(docs)
            w = qw * math.log(1 + (N - df + 0.5) / (df + 0.5))
            for d, tf in zip(docs, tfs):
                if restrict is None or d in restrict:
                    acc[d] += w * tf * k1p1 / (tf + K[d])
        return acc


class TfIdf(Scorer):
    name = "tfidf"

    def score(self, index, query, restrict=None):
        N = index.num_docs
        norms = index.doc_norm
        units = list(_units(index, query, self.phrase_weight))
        wq = []
        for unit, qw, docs, tfs in units:
            idf = math.log10(N / len(docs))
            wq.append((1 + math.log10(qw)) * idf if qw >= 1 else qw * idf)
        qnorm = math.sqrt(sum(w * w for w in wq)) or 1.0
        acc = defaultdict(float)
        for (unit, qw, docs, tfs), w in zip(units, wq):
            w /= qnorm
            if w == 0.0:
                continue
            for d, tf in zip(docs, tfs):
                if restrict is None or d in restrict:
                    acc[d] += w * (1 + math.log10(tf)) / norms[d]
        return acc


def make_scorer(model: str, **kw) -> Scorer:
    model = model.lower()
    if model == "bm25":
        return BM25(k1=kw.get("k1", 1.5), b=kw.get("b", 0.75),
                    phrase_weight=kw.get("phrase_weight", 1.0))
    if model in ("tfidf", "tf-idf", "vsm"):
        return TfIdf(phrase_weight=kw.get("phrase_weight", 1.0))
    raise ValueError(f"unknown model: {model}")
