from typing import NamedTuple, Tuple, Hashable
import numpy as np


class ChannelEdges(NamedTuple):
    src: np.ndarray
    dst: np.ndarray
    meta: dict


def empty_edges(meta_keys=()):
    return ChannelEdges(
        src=np.empty(0, dtype=np.intp),
        dst=np.empty(0, dtype=np.intp),
        meta={k: np.empty(0) for k in meta_keys},
    )


class HopGraph:
    def channels(self):
        raise NotImplementedError

    def has_channel(self, channel):
        raise NotImplementedError

    def edges(self, channel):
        raise NotImplementedError


class ExplicitHopGraph(HopGraph):
    __slots__ = ("_n", "_edges")

    def __init__(self, n_sites, edges):
        self._n = int(n_sites)
        self._edges = {tuple(k) if isinstance(k, (list, tuple)) else k: _normalize_channel_edges(v)
                       for k, v in edges.items()}

    @property
    def n(self):
        return self._n

    def channels(self):
        return tuple(self._edges.keys())

    def has_channel(self, channel):
        return channel in self._edges

    def edges(self, channel):
        e = self._edges.get(channel)
        if e is None:
            return empty_edges()
        return e


class HypercubicHopGraph(HopGraph):
    __slots__ = ("_shape", "_nd", "_n", "_pbc", "_grid_to_site", "_site_coords", "_cache")

    def __init__(self, grid_shape, pbc_flags, grid_to_site, site_coords):
        self._shape = tuple(int(x) for x in grid_shape)
        self._nd = len(self._shape)
        self._pbc = tuple(bool(x) for x in pbc_flags)
        self._grid_to_site = grid_to_site
        self._site_coords = site_coords.astype(np.intp, copy=False)
        self._n = self._site_coords.shape[1]
        self._cache = {}

    @property
    def n(self):
        return self._n

    @property
    def grid_shape(self):
        return self._shape

    @property
    def nd(self):
        return self._nd

    @property
    def pbc_flags(self):
        return self._pbc

    def channels(self):
        return tuple(self._cache.keys())

    def has_channel(self, channel):
        return True

    def edges(self, channel):
        key = tuple(int(x) for x in channel)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        built = self._build(key)
        self._cache[key] = built
        return built

    def _build(self, signed):
        nd = self._nd
        s = np.asarray(signed, dtype=np.intp)
        if s.shape != (nd,):
            raise ValueError(f"channel {signed} has wrong dimensionality (expected {nd})")
        c0 = self._site_coords
        L = np.asarray(self._shape, dtype=np.intp)
        cf = c0 + s[:, None]

        cross = np.zeros(self._n, dtype=np.uint16)
        for a in range(nd):
            cross |= (((cf[a] < 0) | (cf[a] >= L[a])).astype(np.uint16) << a)

        pbc_mask = np.uint16(0)
        for a, p in enumerate(self._pbc):
            if p:
                pbc_mask |= np.uint16(1 << a)

        keep = (cross & np.uint16(~pbc_mask & 0xFFFF)) == 0
        if not keep.any():
            return empty_edges(meta_keys=("cross", "winding"))

        idx = np.nonzero(keep)[0]
        cross_keep = cross[idx]
        cf_w = cf[:, idx].copy()
        for a in range(nd):
            mask_a = ((cross_keep >> a) & 1).astype(bool)
            if mask_a.any():
                cf_w[a, mask_a] %= L[a]

        flat = np.ravel_multi_index(tuple(cf_w), self._shape, order="F")
        dst = self._grid_to_site.ravel(order="F")[flat]

        valid = dst >= 0
        if not valid.all():
            idx = idx[valid]
            cross_keep = cross_keep[valid]
            dst = dst[valid]
            cf_w = cf_w[:, valid]

        src = idx
        delta = self._site_coords[:, dst] - c0[:, src]
        winding = ((s[:, None] - delta) // L[:, None]).astype(np.int16)

        return ChannelEdges(
            src=src.astype(np.intp, copy=False),
            dst=dst.astype(np.intp, copy=False),
            meta={"cross": cross_keep.astype(np.uint16, copy=False),
                  "winding": winding},
        )


class OverlayHopGraph(HopGraph):
    __slots__ = ("_base", "_overlay", "_suppress")

    def __init__(self, base, overlay_edges, suppress_channels):
        self._base = base
        self._overlay = {tuple(k) if isinstance(k, (list, tuple)) else k: _normalize_channel_edges(v)
                         for k, v in overlay_edges.items()}
        self._suppress = frozenset(suppress_channels)

    @property
    def n(self):
        return self._base.n

    def channels(self):
        ch = set(self._base.channels())
        ch.update(self._overlay.keys())
        ch -= self._suppress
        return tuple(ch)

    def has_channel(self, channel):
        if channel in self._suppress:
            return channel in self._overlay
        return channel in self._overlay or self._base.has_channel(channel)

    def edges(self, channel):
        ov = self._overlay.get(channel)
        if channel in self._suppress:
            return ov if ov is not None else empty_edges()
        base = self._base.edges(channel)
        if ov is None:
            return base
        if base.src.size == 0:
            return ov
        meta = {}
        keys = set(base.meta.keys()) | set(ov.meta.keys())
        for k in keys:
            if k in base.meta and k in ov.meta:
                meta[k] = np.concatenate([base.meta[k], ov.meta[k]], axis=-1)
            elif k in base.meta:
                meta[k] = base.meta[k]
            else:
                meta[k] = ov.meta[k]
        return ChannelEdges(
            src=np.concatenate([base.src, ov.src]),
            dst=np.concatenate([base.dst, ov.dst]),
            meta=meta,
        )


def _normalize_channel_edges(v):
    if isinstance(v, ChannelEdges):
        return v
    if isinstance(v, tuple) and len(v) == 2:
        src, dst = v
        return ChannelEdges(src=np.asarray(src, dtype=np.intp),
                            dst=np.asarray(dst, dtype=np.intp), meta={})
    if isinstance(v, dict):
        src = np.asarray(v["src"], dtype=np.intp)
        dst = np.asarray(v["dst"], dtype=np.intp)
        meta = {k: np.asarray(val) for k, val in v.items() if k not in ("src", "dst")}
        return ChannelEdges(src=src, dst=dst, meta=meta)
    raise TypeError(f"cannot normalize edges of type {type(v).__name__}")

