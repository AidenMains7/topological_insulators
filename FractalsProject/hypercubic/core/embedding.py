import numpy as np


class Embedding:
    __slots__ = ("_coords", "_nd", "_n")

    def __init__(self, coords):
        arr = np.asarray(coords)
        if arr.ndim != 2:
            raise ValueError(f"coords must be 2D (nd, n_sites), got shape {arr.shape}")
        self._coords = arr
        self._nd, self._n = arr.shape

    @property
    def nd(self):
        return self._nd

    @property
    def n(self):
        return self._n

    def coords(self):
        return self._coords

    def coords_of(self, site_ids):
        return self._coords[:, site_ids]

    def transformed(self, origin, scales, shift_first):
        o = _broadcast_vec(origin, self._nd)
        s = _broadcast_vec(scales, self._nd)
        c = self._coords
        if shift_first:
            new = (c - o[:, None]) * s[:, None]
        else:
            new = c * s[:, None] - o[:, None]
        return Embedding(new)


class GridEmbedding(Embedding):
    __slots__ = ("_shape", "_grid_to_site", "_site_to_grid")

    def __init__(self, coords, grid_shape, grid_to_site, site_to_grid):
        super().__init__(coords)
        self._shape = tuple(int(x) for x in grid_shape)
        self._grid_to_site = grid_to_site
        self._site_to_grid = site_to_grid

    @property
    def grid_shape(self):
        return self._shape

    def grid_to_site(self):
        return self._grid_to_site

    def site_to_grid(self):
        return self._site_to_grid

    def transformed(self, origin, scales, shift_first):
        e = super().transformed(origin, scales, shift_first)
        return GridEmbedding(e.coords(), self._shape, self._grid_to_site, self._site_to_grid)


def _broadcast_vec(v, nd):
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full(nd, float(arr))
    if arr.shape != (nd,):
        raise ValueError(f"expected scalar or shape ({nd},), got {arr.shape}")
    return arr


def grid_embedding_from_shape(shape):
    shape = tuple(int(x) for x in shape)
    nd = len(shape)
    n = 1
    for s in shape:
        n *= s
    coords = np.indices(shape).reshape(nd, -1, order="F").astype(np.intp)
    site_to_grid = coords.T.copy()
    grid_to_site = np.arange(n, dtype=np.intp).reshape(shape, order="F")
    return GridEmbedding(coords.astype(float), shape, grid_to_site, site_to_grid)


def grid_embedding_from_active(shape, active_mask):
    shape = tuple(int(x) for x in shape)
    nd = len(shape)
    flat_active = np.asarray(active_mask, dtype=bool).ravel(order="F")
    n_full = flat_active.size
    n_active = int(flat_active.sum())
    all_coords = np.indices(shape).reshape(nd, -1, order="F")
    coords = all_coords[:, flat_active].astype(float)
    site_to_grid = coords.T.astype(np.intp)
    grid_to_site = np.full(n_full, -1, dtype=np.intp)
    grid_to_site[flat_active] = np.arange(n_active, dtype=np.intp)
    grid_to_site = grid_to_site.reshape(shape, order="F")
    return GridEmbedding(coords, shape, grid_to_site, site_to_grid)

