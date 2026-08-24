"""Unified persistence and canonical case naming (Layer 1 support).

Data layout
-----------
One ``data/`` root, one subdirectory per observable family, then nested by the
parameters most useful for browsing, one self-describing ``.npz`` per run named
by :func:`case_tag`::

    data/
        phase_diagram/<fractal>/L<L>/<tag>.npz       # grid, M_vals, M_alt_vals
        witten/<geometry>/<method>/L<L>/<tag>.npz
        ldos/<geometry>/<method>/L<L>/<tag>.npz

For witten/ldos, ``<geometry>`` folds the fractal with the tiling choice
(``tiled_sponge`` / ``untiled_sponge`` / ``cube``); the ``<method>`` level is
omitted when it merely repeats the fractal name (e.g. the solid ``cube``
reference).  Every file embeds its full parameter set as JSON under the reserved
key ``__meta__``, so each file is fully self-describing without any external index.
"""

import json
from pathlib import Path

import numpy as np

# Data lives at the project root, one level above this package.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
KINDS = ("phase_diagram", "witten", "ldos")
_META_KEY = "__meta__"


def case_tag(**params):
    """Canonical filename stem from the parameters that identify a run.

    Only the keys that are present are emitted, in a fixed order, so tags are
    stable and greppable, e.g.::

        case_tag(fractal='sponge', n=1, L=6, M=2.0, M_alt=-0.1,
                 pasted=True, method='substituted')
        -> 'sponge_n1_L6_M+2.00_Malt-0.10_pasted_substituted'
    """
    order = ["fractal", "n", "L", "M", "M_alt",
             "M_prime", "t", "B", "gauge", "method"]
    parts = []
    for key in order:
        if key not in params or params[key] is None:
            continue
        val = params[key]
        if key == "n":
            parts.append(f"n{val}")
        elif key == "L":
            parts.append(f"L{int(val)}")
        elif key in ("M", "M_alt", "M_prime"):
            short = {"M": "M", "M_alt": "Malt", "M_prime": "Mp"}[key]
            parts.append(f"{short}{val:+.2f}")
        elif key in ("t", "B"):
            parts.append(f"{key}{val:g}")
        elif key == "gauge":
            parts.append(f"g{val}")
        else:
            parts.append(str(val))
    for flag in ("pasted", "pbc"):
        if params.get(flag):
            parts.append(flag)
    return "_".join(parts)


def save_npz(path, **arrays):
    """Save ``arrays`` to ``path`` (``.npz`` appended if missing); returns the Path."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_npz(path):
    """Load an ``.npz`` into a plain dict."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    return dict(np.load(path, allow_pickle=False))


# ── canonical data layout ────────────────────────────────────────────────────

def _result_subdir(kind, meta):
    """Nested directory parts (under ``data/<kind>/``) for a run's metadata.

    Layout::

        phase_diagram/<fractal>/L<L>/
        witten/<geometry>/<method>/L<L>/
        ldos/<geometry>/<method>/L<L>/

    For witten/ldos the ``<geometry>`` label folds the fractal together with the
    tiling choice: ``tiled_sponge`` / ``untiled_sponge`` when ``pasted`` is a
    meaningful axis, otherwise just the fractal name (e.g. ``cube``).  The
    ``<method>`` level is omitted when it merely repeats the fractal name (so the
    solid ``cube`` reference lands in ``cube/L<L>/`` rather than ``cube/cube/L<L>/``).
    """
    if kind in ("witten", "ldos"):
        fractal = str(meta.get("fractal", "unknown"))
        if "pasted" in meta:
            geometry = ("tiled_" if meta.get("pasted") else "untiled_") + fractal
        else:
            geometry = fractal
        parts = [geometry]
        method = meta.get("method")
        if method is not None and method != meta.get("fractal"):
            parts.append(str(method))
    else:                                   # phase_diagram (and any future kind)
        parts = [str(meta.get("fractal", "unknown"))]
    L = meta.get("L")
    if L is not None:
        parts.append(f"L{int(L)}")
    return parts


def result_path(kind, meta, *, root=None):
    """Path for a run of the given ``kind`` (see :data:`KINDS`), named by ``meta``.

    The file lands in a nested subdirectory derived from ``meta`` (see
    :func:`_result_subdir`); the filename stem is :func:`case_tag`.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}.")
    root = Path(root) if root is not None else DATA_ROOT
    parts = _result_subdir(kind, meta)
    return root.joinpath(kind, *parts, f"{case_tag(**meta)}.npz")


def save_result(kind, meta, *, root=None, **arrays):
    """Save arrays + JSON metadata for one run; returns the Path.

    ``meta`` fully identifies the run (used both for the filename and embedded
    under ``__meta__`` for self-description).
    """
    path = result_path(kind, meta, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{_META_KEY: json.dumps(meta)}, **arrays)
    return path


def load_result(path):
    """Load a run saved by :func:`save_result`; returns ``(arrays, meta)``."""
    d = load_npz(path)
    meta = json.loads(str(d.pop(_META_KEY))) if _META_KEY in d else {}
    return d, meta
