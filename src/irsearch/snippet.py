"""Query-biased summary snippets.

The snippet is the window of `window` consecutive surface tokens that covers
the largest number of distinct query terms (ties: more matched tokens, then a
phrase occurrence, then the earliest window). Matched words are marked so the
console can highlight them.
"""
import re
from typing import List, Tuple

from .analyzer import Analyzer
from .query import Query

Piece = Tuple[str, bool]          # (text, is_match)
_WS = re.compile(r"\s+")


def make_snippet(text: str, analyzer: Analyzer, query: Query,
                 window: int = 36) -> List[Piece]:
    toks = list(analyzer.surface_tokens(text))
    if not toks:
        return [("(empty document)", False)]
    wanted = query.highlight
    matched = [set(t) & wanted for t, _, _ in toks]

    # token index at which a full query phrase starts
    flat, owner = [], []
    for i, (terms, _, _) in enumerate(toks):
        for t in terms:
            flat.append(t)
            owner.append(i)
    phrase_start = [0] * len(toks)
    for ph in query.phrases:
        L = len(ph)
        for j in range(len(flat) - L + 1):
            if tuple(flat[j:j + L]) == ph:
                phrase_start[owner[j]] = 1

    n = len(toks)
    w = min(window, n)
    best, best_key = 0, None
    for s in range(0, n - w + 1):
        seen = set()
        hits = 0
        ph = 0
        for i in range(s, s + w):
            if matched[i]:
                seen |= matched[i]
                hits += 1
            ph += phrase_start[i]
        key = (len(seen), min(ph, 1), hits, -s)
        if best_key is None or key > best_key:
            best, best_key = s, key
    # a little left context when the window does not start the document
    start = max(0, best - 3)
    end = min(n, best + w)          # keep the whole chosen window

    a = toks[start][1] if start > 0 else 0
    z = toks[end - 1][2]
    pieces: List[Piece] = []
    if start > 0:
        pieces.append(("...", False))
    cur = a
    for i in range(start, end):
        terms, ts, te = toks[i]
        if matched[i]:
            if ts > cur:
                pieces.append((text[cur:ts], False))
            pieces.append((text[ts:te], True))
            cur = te
    if z > cur:
        pieces.append((text[cur:z], False))
    if end < n:
        pieces.append((" ...", False))
    # collapse whitespace inside pieces (documents are hard-wrapped)
    out = []
    for i, (s, m) in enumerate(pieces):
        s = _WS.sub(" ", s)
        if i == 0 or (i == 1 and pieces[0][0] == "..."):
            s = s.lstrip() if i == 0 else s
        out.append((s, m))
    return out


def render(pieces: List[Piece], width: int = 80, color: bool = False) -> str:
    """Word-wrap pieces to `width` visible columns; highlight matches in color mode."""
    words: List[Tuple[str, bool, bool]] = []      # (word, is_match, glue_to_prev)
    prev_space = True
    for text, is_match in pieces:
        if not text:
            continue
        parts = re.split(r"(\s+)", text)
        # glue to the previous word only if no whitespace separates them
        glue = not prev_space and not text[:1].isspace()
        prev_space = text[-1:].isspace()
        for p in parts:
            if not p:
                continue
            if p.isspace():
                glue = False
                continue
            words.append((p, is_match, glue and bool(words)))
            glue = True
    lines, line, vis = [], [], 0
    for word, is_match, glue in words:
        shown = f"\033[1;33m{word}\033[0m" if (color and is_match) else word
        if glue and line:
            line[-1] += shown
            vis += len(word)
            continue
        add = len(word) + (1 if line else 0)
        if line and vis + add > width:
            lines.append(" ".join(line))
            line, vis = [shown], len(word)
        else:
            line.append(shown)
            vis += add
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)
