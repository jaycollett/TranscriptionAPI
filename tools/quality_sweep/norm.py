"""Text normalisation for the sweep, reused verbatim from the quality harness.

Both sides of every comparison (a legacy transcript from the orchestrator database
and a 0.6.0 transcript from the sweep service) go through `norm_words`, so case,
punctuation and curly quotes never count as a difference. The harness module is the
single source of truth; this shim only puts it on the path.
"""

import os
import sys

_HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quality_harness")
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

from textnorm import norm_token, norm_words  # noqa: E402

__all__ = ["norm_token", "norm_words"]
