"""Reading the TDT3 collection.

Each file holds one SGML document of the form
    <DOC><DOCNO>..</DOCNO><DOCTYPE>..</DOCTYPE><TXTTYPE>..</TXTTYPE>
    <TEXT> ... </TEXT></DOC>
The docID shown to the user is the file name without extension, e.g.
NYT19981231_0439.
"""
import os
import re
from typing import Iterator, Tuple

_TEXT_RE = re.compile(r"<TEXT>(.*?)</TEXT>", re.S)
_TYPE_RE = re.compile(r"<DOCTYPE>\s*(.*?)\s*</DOCTYPE>", re.S)


def iter_files(root: str) -> Iterator[str]:
    """All .txt documents under root, in a deterministic order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".txt") and not name.startswith("."):
                yield os.path.join(dirpath, name)


def read_text(path: str) -> str:
    """Body text of a document (content of <TEXT>), whitespace preserved."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    m = _TEXT_RE.search(raw)
    return m.group(1) if m else ""


def read_doc(path: str) -> Tuple[str, str, str]:
    """(docid, doctype, text)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    m = _TEXT_RE.search(raw)
    t = _TYPE_RE.search(raw)
    docid = os.path.splitext(os.path.basename(path))[0]
    return docid, (t.group(1) if t else ""), (m.group(1) if m else "")
