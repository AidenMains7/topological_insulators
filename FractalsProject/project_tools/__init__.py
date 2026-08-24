"""project_tools — computation library for the Chapter 7 fractal-patterned-mass
(Cantor / carpet / sponge) results.

Compute building blocks live here.  Orchestration lives in the root-level
``compute_data.py`` (data computation) and ``plot_data.py`` (figures).  See ``README.md``.
"""

from . import lattice, model, observables, witten, io

__all__ = ["lattice", "model", "observables", "witten", "io"]
