"""Isolated observed-change instruction granularity experiment.

The package is an empirical screening implementation.  It is not a
commutativity proof and does not execute LLVM tools or Phasebatch workers.
"""

from .models import ExtractionLevel

__all__ = ["ExtractionLevel"]

