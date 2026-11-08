"""Text analysis: tokenization and normalization shared by indexing and querying.

Decisions (see report, section on preprocessing):
  * case folding: everything is lowercased;
  * numbers are kept; thousands separators are removed ("1,000" -> "1000"),
    decimals keep their point ("3.6");
  * dotted abbreviations are joined ("U.S." -> "us");
  * punctuation is dropped; possessive "'s" is removed, other apostrophes are
    deleted ("don't" -> "dont");
  * hyphenated words are split into their parts at consecutive positions
    ("closed-door" -> "closed", "door"), so that a hyphenated query word can be
    matched as a phrase;
  * Porter stemming (can be disabled for ablation);
  * stopwords are NOT removed.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, List, Tuple

from . import porter

_TOKEN_RE = re.compile(
    r"(?P<num>\d+(?:[.,]\d+)+)"          # 1,000  3.6  1,234.5
    r"|(?P<abbr>(?:[a-z]\.){2,})"         # u.s.  u.n.
    r"|(?P<word>[a-z0-9]+(?:['’\-][a-z0-9]+)*)"  # words, clinton's, closed-door
)


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return porter.stem(word)


@dataclass(frozen=True)
class Analyzer:
    stemming: bool = True

    def _normalize_part(self, part: str) -> str:
        if self.stemming and not part.isdigit():
            return _stem(part)
        return part

    def surface_tokens(self, text: str) -> Iterator[Tuple[List[str], int, int]]:
        """Yield (terms, start, end) for every surface token of `text`.

        `terms` has more than one element only for hyphenated words. Offsets
        refer to the original string and are used for snippet highlighting.
        """
        lowered = text.lower()
        for m in _TOKEN_RE.finditer(lowered):
            kind = m.lastgroup
            tok = m.group(kind)
            if kind == "num":
                if "." in tok and "," in tok:
                    tok = tok.replace(",", "")
                elif "," in tok:
                    tok = tok.replace(",", "")
                yield [tok], m.start(), m.end()
            elif kind == "abbr":
                yield [tok.replace(".", "")], m.start(), m.end()
            else:
                tok = tok.replace("’", "'")
                if tok.endswith("'s"):
                    tok = tok[:-2]
                tok = tok.replace("'", "")
                parts = [p for p in tok.split("-") if p]
                if parts:
                    yield [self._normalize_part(p) for p in parts], m.start(), m.end()

    def terms(self, text: str) -> List[str]:
        """Flat list of index terms in positional order."""
        out: List[str] = []
        for terms, _, _ in self.surface_tokens(text):
            out.extend(terms)
        return out
