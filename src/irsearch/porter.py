"""Porter stemmer, implemented from M. F. Porter, "An algorithm for suffix
stripping", Program 14(3), 1980.

The implementation follows the published algorithm (steps 1a-5b) without the
later extensions of the reference C version. Within each step only the longest
matching suffix is considered; if its condition fails, the step leaves the word
unchanged.
"""

_VOWELS = frozenset("aeiou")


def _is_cons(w: str, i: int) -> bool:
    """A consonant is a letter other than a/e/i/o/u, and other than y
    preceded by a consonant."""
    ch = w[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_cons(w, i - 1)
    return True


def _measure(stem: str) -> int:
    """m in the form [C](VC)^m[V]."""
    m = 0
    prev_vowel = False
    for i in range(len(stem)):
        cons = _is_cons(stem, i)
        if cons and prev_vowel:
            m += 1
        prev_vowel = not cons
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_cons(stem, i) for i in range(len(stem)))


def _ends_double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_cons(w, len(w) - 1)


def _ends_cvc(w: str) -> bool:
    """*o: stem ends cvc, where the second c is not w, x or y."""
    n = len(w)
    return (n >= 3 and _is_cons(w, n - 3) and not _is_cons(w, n - 2)
            and _is_cons(w, n - 1) and w[-1] not in "wxy")


def _apply(word: str, rules, cond):
    """Apply the longest matching (suffix, replacement) rule if cond(stem)."""
    for suffix, repl in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            return stem + repl if cond(stem) else word
    return word


def _m_gt0(stem):
    return _measure(stem) > 0


def _m_gt1(stem):
    return _measure(stem) > 1


# Suffix lists are ordered so that a longer suffix precedes any shorter
# suffix it contains, which makes "first match" equal to "longest match".
_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
_STEP2.sort(key=lambda r: -len(r[0]))

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
_STEP3.sort(key=lambda r: -len(r[0]))

_STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
          "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
_STEP4.sort(key=lambda s: -len(s))


def _step1a(w):
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w):
    if w.endswith("eed"):
        return w[:-1] if _m_gt0(w[:-3]) else w
    for suffix in ("ed", "ing"):
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if not _has_vowel(stem):
                return w
            w = stem
            if w.endswith(("at", "bl", "iz")):
                return w + "e"
            if _ends_double_cons(w) and w[-1] not in "lsz":
                return w[:-1]
            if _measure(w) == 1 and _ends_cvc(w):
                return w + "e"
            return w
    return w


def _step1c(w):
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


def _step4(w):
    for suffix in _STEP4:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if suffix == "ion":
                ok = _m_gt1(stem) and stem.endswith(("s", "t"))
            else:
                ok = _m_gt1(stem)
            return stem if ok else w
    return w


def _step5(w):
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            w = stem
    if w.endswith("ll") and _measure(w) > 1:
        w = w[:-1]
    return w


def stem(word: str) -> str:
    """Return the Porter stem of a lowercase word."""
    if len(word) <= 2:
        return word
    w = _step1a(word)
    w = _step1b(w)
    w = _step1c(w)
    w = _apply(w, _STEP2, _m_gt0)
    w = _apply(w, _STEP3, _m_gt0)
    w = _step4(w)
    w = _step5(w)
    return w
