"""Positional inverted index built from scratch.

Postings for a term t are stored as three parallel unsigned-int arrays:
    docs[k]  internal id of the k-th document containing t (ascending)
    tfs[k]   frequency of t in that document
    pos      all positions of t, concatenated in document order; the positions
             for docs[k] are pos[sum(tfs[:k]) : sum(tfs[:k+1])]
Arrays keep the on-disk pickle compact and fast to load compared with nested
Python lists.
"""
import math
import os
import pickle
import time
from array import array
from collections import defaultdict
from typing import Dict, List, Tuple

from .analyzer import Analyzer
from .corpus import iter_files, read_doc

FORMAT_VERSION = 1


class Postings:
    __slots__ = ("docs", "tfs", "pos", "_offsets")

    def __init__(self):
        self.docs = array("I")
        self.tfs = array("I")
        self.pos = array("I")
        self._offsets = None

    def __getstate__(self):
        return (self.docs, self.tfs, self.pos)

    def __setstate__(self, state):
        self.docs, self.tfs, self.pos = state
        self._offsets = None

    @property
    def df(self) -> int:
        return len(self.docs)

    def positions_by_doc(self) -> Dict[int, array]:
        """doc id -> positions (only computed for phrase evaluation)."""
        out = {}
        start = 0
        pos = self.pos
        for d, tf in zip(self.docs, self.tfs):
            out[d] = pos[start:start + tf]
            start += tf
        return out


class InvertedIndex:
    def __init__(self, analyzer: Analyzer):
        self.analyzer = analyzer
        self.postings: Dict[str, Postings] = {}
        self.doc_names: List[str] = []
        self.doc_paths: List[str] = []      # relative to data_root
        self.doc_types: List[str] = []
        self.doc_len = array("I")           # number of terms (BM25 |d|)
        self.doc_norm = array("d")          # lnc vector length (TF-IDF)
        self.data_root = ""
        self.build_seconds = 0.0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, data_root: str, analyzer: Analyzer, progress=None) -> "InvertedIndex":
        idx = cls(analyzer)
        idx.data_root = os.path.abspath(data_root)
        t0 = time.perf_counter()
        postings = defaultdict(Postings)
        for n, path in enumerate(iter_files(data_root)):
            docid, doctype, text = read_doc(path)
            d = len(idx.doc_names)
            idx.doc_names.append(docid)
            idx.doc_paths.append(os.path.relpath(path, data_root))
            idx.doc_types.append(doctype)

            local: Dict[str, List[int]] = defaultdict(list)
            for p, term in enumerate(analyzer.terms(text)):
                local[term].append(p)
            length = sum(len(v) for v in local.values())
            norm_sq = 0.0
            for term, plist in local.items():
                pst = postings[term]
                pst.docs.append(d)
                pst.tfs.append(len(plist))
                pst.pos.extend(plist)
                w = 1.0 + math.log10(len(plist))
                norm_sq += w * w
            idx.doc_len.append(length)
            idx.doc_norm.append(math.sqrt(norm_sq))
            if progress and (n + 1) % 5000 == 0:
                progress(n + 1)
        idx.postings = dict(postings)
        idx.build_seconds = time.perf_counter() - t0
        return idx

    # ------------------------------------------------------------ statistics
    @property
    def num_docs(self) -> int:
        return len(self.doc_names)

    @property
    def avg_doc_len(self) -> float:
        return sum(self.doc_len) / max(1, self.num_docs)

    @property
    def num_tokens(self) -> int:
        return sum(self.doc_len)

    def get(self, term: str):
        return self.postings.get(term)

    def doc_path(self, d: int) -> str:
        return os.path.join(self.data_root, self.doc_paths[d])

    # ----------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "version": FORMAT_VERSION,
            "stemming": self.analyzer.stemming,
            "postings": self.postings,
            "doc_names": self.doc_names,
            "doc_paths": self.doc_paths,
            "doc_types": self.doc_types,
            "doc_len": self.doc_len,
            "doc_norm": self.doc_norm,
            "data_root": self.data_root,
            "build_seconds": self.build_seconds,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, data_root: str = None) -> "InvertedIndex":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("version") != FORMAT_VERSION:
            raise ValueError("index format mismatch; rebuild with --build")
        idx = cls(Analyzer(stemming=state["stemming"]))
        idx.postings = state["postings"]
        idx.doc_names = state["doc_names"]
        idx.doc_paths = state["doc_paths"]
        idx.doc_types = state["doc_types"]
        idx.doc_len = state["doc_len"]
        idx.doc_norm = state["doc_norm"]
        idx.data_root = os.path.abspath(data_root) if data_root else state["data_root"]
        idx.build_seconds = state["build_seconds"]
        return idx
