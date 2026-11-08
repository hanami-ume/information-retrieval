"""Search facade: parse -> score -> rank -> snippets."""
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .corpus import read_text
from .index import InvertedIndex
from .query import Query, parse_query
from .scoring import make_scorer, phrase_postings
from .snippet import make_snippet


@dataclass
class Hit:
    rank: int
    doc: int
    docid: str
    score: float
    snippet: list = field(default_factory=list)


@dataclass
class SearchResult:
    query: Query
    hits: List[Hit]
    matched: int          # documents with a non-zero score
    seconds: float


class SearchEngine:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self._scorers = {}

    def scorer(self, model: str, **kw):
        key = (model, tuple(sorted(kw.items())))
        if key not in self._scorers:
            self._scorers[key] = make_scorer(model, **kw)
        return self._scorers[key]

    def search(self, text: str, model: str = "bm25", hits: int = 10,
               strict_phrase: bool = False, snippets: bool = True,
               hyphen_mode: str = "phrase", **params) -> SearchResult:
        t0 = time.perf_counter()
        q = parse_query(text, self.index.analyzer, hyphen_mode=hyphen_mode)
        if q.is_empty:
            return SearchResult(q, [], 0, time.perf_counter() - t0)
        restrict: Optional[set] = None
        if strict_phrase and q.phrases:
            for ph in q.phrases:
                docs = set(phrase_postings(self.index, ph))
                restrict = docs if restrict is None else restrict & docs
        scores = self.scorer(model, **params).score(self.index, q, restrict)
        # deterministic order: higher score first, then lower doc id
        top = heapq.nsmallest(hits, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        result = []
        for r, (d, s) in enumerate(top, 1):
            h = Hit(r, d, self.index.doc_names[d], s)
            if snippets:
                h.snippet = make_snippet(read_text(self.index.doc_path(d)),
                                         self.index.analyzer, q)
            result.append(h)
        return SearchResult(q, result, len(scores), time.perf_counter() - t0)
