"""Query parsing: free text mixed with double-quoted phrases.

    hurricane                 -> terms {hurrican}
    "newt gingrich" down      -> phrase (newt, gingrich) + terms {newt, gingrich, down}
    nba strike closed-door    -> phrase (close, door)   + terms {nba, strike}

Quoted phrases are matched *softly*: the member words of a phrase are also added
as ordinary terms, so a document lacking the exact phrase can still be retrieved
on the individual words, while documents containing the phrase receive the
additional phrase contribution.

A hyphenated free-text word is by default a single phrase unit: it is one word
of the query, and giving it three units (phrase + both parts) let it outweigh
the other query words (see report, Q5).
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from .analyzer import Analyzer

_PHRASE_RE = re.compile(r"[\"“”]([^\"“”]*)[\"“”]")


@dataclass
class Query:
    raw: str
    terms: Counter = field(default_factory=Counter)          # term -> query tf
    phrases: List[Tuple[str, ...]] = field(default_factory=list)
    phrase_terms: Set[str] = field(default_factory=set)        # words inside phrases

    @property
    def highlight(self) -> Set[str]:
        return set(self.terms) | self.phrase_terms

    @property
    def is_empty(self) -> bool:
        return not self.terms and not self.phrases


HYPHEN_MODES = ("phrase", "phrase+parts", "parts")


def parse_query(text: str, analyzer: Analyzer, hyphen_mode: str = "phrase") -> Query:
    """Parse a query string.

    hyphen_mode controls a hyphenated free-text word such as "closed-door":
      "phrase"        one phrase unit only (default: one query word = one unit)
      "phrase+parts"  phrase unit plus each part as a term (same as quoted phrases)
      "parts"         only the parts as independent terms
    """
    if hyphen_mode not in HYPHEN_MODES:
        raise ValueError(f"hyphen_mode must be one of {HYPHEN_MODES}")
    q = Query(raw=text.strip())

    def add_group(terms, hyphenated=False):
        if not terms:
            return
        if len(terms) == 1:
            q.terms.update(terms)
            return
        mode = hyphen_mode if hyphenated else "phrase+parts"
        if mode in ("phrase+parts", "parts"):
            q.terms.update(terms)
        if mode in ("phrase", "phrase+parts"):
            q.phrases.append(tuple(terms))
            q.phrase_terms.update(terms)

    # quoted phrases first, then whatever is left is free text
    rest = []
    last = 0
    for m in _PHRASE_RE.finditer(text):
        rest.append(text[last:m.start()])
        add_group(analyzer.terms(m.group(1)))
        last = m.end()
    rest.append(text[last:])
    free = re.sub(r"[\"“”]", " ", " ".join(rest))  # unbalanced quotes
    for terms, _, _ in analyzer.surface_tokens(free):
        add_group(terms, hyphenated=True)
    return q
