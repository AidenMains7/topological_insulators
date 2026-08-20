"""Fractal lattice geometry (Layer 0a).

Pure NumPy: no physics, no ``hypercubic`` dependency.  Produces the 0/1 sector
arrays consumed by :mod:`model`.

Two orthogonal size knobs live here and should stay explicit everywhere:

    generation ``n``   -> fractal detail (via ``fractal_lattice`` / ``upscale_to_n``)
    linear size ``L``  -> ``block_scale`` (each site becomes a block_scale**D block),
                          with ``upscale_to_n`` and ``pasted`` also scaling L; see
                          :func:`system_length` for the exact relation.
"""

import numpy as np



def _seed(name):
    name = name.lower()
    if name == "cantor":
        return np.array([1, 0, 1], dtype=np.uint8)
    if name == "carpet":
        s = np.ones((3, 3), dtype=np.uint8)
        s[1, 1] = 0
        return s
    if name == "sponge":
        s = np.ones((3, 3, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i == 1) + (j == 1) + (k == 1) >= 2:
                        s[i, j, k] = 0
        return s
    raise ValueError(f"unknown fractal {name!r}; choose 'cantor', 'carpet', or 'sponge'.")


def substitution_system(seed, n):
    """Generation-``n`` substitution of a binary hypercubic ``seed``."""
    seed = np.asarray(seed, dtype=np.uint8)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    rules = np.stack((np.zeros_like(seed), seed), axis=0)
    arr = np.ones((1,) * seed.ndim, dtype=np.uint8)
    D, b = seed.ndim, rules.shape[1]
    perm = np.empty(2 * D, dtype=np.intp)
    perm[0::2] = np.arange(D)
    perm[1::2] = np.arange(D, 2 * D)
    for _ in range(n):
        arr = rules[arr].transpose(perm).reshape(tuple(s * b for s in arr.shape))
    return (arr != 0).astype(np.uint8)


def fractal_lattice(name, n, *, upscale_to_n=None):
    """Order-``n`` fractal sector array, optionally cell-replicated to a higher order."""
    lat = substitution_system(_seed(name), n)
    if upscale_to_n is not None and upscale_to_n > n:
        scale = 3 ** (upscale_to_n - n)
        for axis in range(lat.ndim):
            lat = np.repeat(lat, scale, axis=axis)
    return lat


def block_upscale(lat, block_scale):
    """Replace every site with a ``block_scale``-sided hypercubic block (the ``L`` knob)."""
    arr = np.asarray(lat)
    if not isinstance(block_scale, (int, np.integer)) or block_scale < 1:
        raise ValueError(f"block_scale must be a positive integer; got {block_scale!r}.")
    if block_scale == 1:
        return arr
    for axis in range(arr.ndim):
        arr = np.repeat(arr, block_scale, axis=axis)
    return arr


def system_length(n, *, upscale_to_n=None, block_scale=2, pasted=False):
    """Linear size ``L`` (sites along one axis) of the built lattice.

    Mirrors :func:`build_lattice`: the order-``n`` fractal is repeated to
    ``max(n, upscale_to_n)`` and by ``block_scale`` (both plain ``np.repeat``,
    so they combine into a single factor), then doubled per axis if ``pasted``.
    """
    up = n if upscale_to_n is None else max(n, upscale_to_n)
    L = int(block_scale) * 3 ** up
    if pasted:
        L *= 2
    return int(L)


def build_lattice(name, n, *, upscale_to_n=None, block_scale=2, pasted=False):
    """One call: fractal_lattice -> block_upscale -> (2x..x2 tile if ``pasted``).

    This is the full geometry chain: fractal pattern, block expansion, tiling.
    """
    lat = fractal_lattice(name, n, upscale_to_n=upscale_to_n)
    lat = block_upscale(lat, block_scale)
    if pasted:
        lat = np.tile(lat, (2,) * lat.ndim)
    return np.asarray(lat, dtype=int)

