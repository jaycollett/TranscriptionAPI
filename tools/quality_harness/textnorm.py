"""Word normalisation shared by every metric and by the MFA word matcher.

Both sides of any comparison (Whisper words, MFA words-tier labels, transcripts of
two configs) go through `norm_words` so punctuation, case and curly quotes never
count as disagreement.
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
