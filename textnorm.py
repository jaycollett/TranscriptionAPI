"""Word normalisation shared by transcribe.py, app.py and the unit tests.

Both sides of any comparison (Whisper words against MFA words-tier labels, the tail
of one segment against the head of the next) go through `norm_words`, so punctuation,
case and curly quotes never count as a difference.

Kept byte-identical to tools/quality_harness/textnorm.py below the docstring: the
harness measured the pipeline through this rule, and the two copies drifting apart
would silently invalidate every agreement number in
tools/quality_harness/results/.
"""

import re

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
# Keep letters, digits and internal apostrophes; everything else is a separator.
_STRIP = re.compile(r"[^a-z0-9']+")


def norm_token(token):
    """Normalise a single token: lowercase, ascii quotes, edge punctuation removed."""
    token = token.translate(_QUOTES).lower()
    token = _STRIP.sub(" ", token).strip()
    return token.strip("'")


def norm_words(text):
    """Split `text` into normalised words, dropping tokens that normalise to nothing."""
    out = []
    for raw in text.translate(_QUOTES).lower().split():
        for piece in _STRIP.sub(" ", raw).split():
            piece = piece.strip("'")
            if piece:
                out.append(piece)
    return out
