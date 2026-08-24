import numpy as np

from ..core.sites import SiteRegistry
from ..core.embedding import (Embedding, GridEmbedding,
                               grid_embedding_from_shape, grid_embedding_from_active)
from ..core.hopgraph import HypercubicHopGraph
from ..core.internal import clifford_basis, trivial_internal
from ..core.model import Model
from ..dsl.dvector import make_operator_term


def _normalize_pbc(pbc, nd):
    if isinstance(pbc, bool):
        return tuple([pbc] * nd)
    flags = tuple(bool(x) for x in pbc)
    if len(flags) != nd:
        raise ValueError(f"pbc_flags has length {len(flags)}, expected {nd}")
    return flags


def _normalize_dim_symbols(symbols, nd):
    if symbols == "default" or symbols is None:
        if nd <= 3:
            return ("kx", "ky", "kz")[:nd]
        return tuple(f"kx{i+1}" for i in range(nd))
    if len(symbols) != nd:
        raise ValueError(f"dimension_symbols has length {len(symbols)}, expected {nd}")
    return tuple(str(x) for x in symbols)


def _detect_modifier_signature(fn):
    if fn is None:
        return None, (), False
    import inspect
    sig = inspect.signature(fn)
    keys = []
    wants_ctx = False
    for i, (name, param) in enumerate(sig.parameters.items()):
        if i < 3:
            continue
        if name == "ctx":
            wants_ctx = True
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        keys.append(name)
    return fn, tuple(keys), wants_ctx


def _required_clifford_pairs(d_keys):
    max_idx = 0
    for k in d_keys:
        for s in k.split("_")[1:]:
            v = int(s)
            if v > max_idx:
                max_idx = v
    if max_idx == 0:
        return 1
    pairs = 1
    while 2 * pairs + 1 < max_idx:
        pairs += 1
    return pairs


def hypercubic_grid_model(lattice_shape=None, lattice=None, pbc_flags=False,
                          origin=0.0, scales=1.0, shift_first=True,
                          dimension_symbols="default",
                          real_space_functions=(),
                          hopping_modifier=None,
                          internal=None,
                          display_lattice=None,
                          default_params=None,
                          **d_components):
    if lattice_shape is None and lattice is None:
        raise ValueError("must supply lattice_shape or lattice")
    if lattice_shape is not None and lattice is not None:
        raise ValueError("supply only one of lattice_shape, lattice")

    if lattice is not None:
        lattice_arr = np.asarray(lattice, dtype=int)
        shape = lattice_arr.shape
        active = lattice_arr >= 0
        sector_labels_full = lattice_arr.ravel(order="F")
        active_flat = active.ravel(order="F")
        sector_labels = sector_labels_full[active_flat]
    else:
        if isinstance(lattice_shape, int):
            shape = (lattice_shape,)
        else:
            shape = tuple(int(x) for x in lattice_shape)
        active = np.ones(shape, dtype=bool)
        active_flat = active.ravel(order="F")
        n_full = active_flat.size
        sector_labels = np.zeros(n_full, dtype=int)

    nd = len(shape)
    pbc = _normalize_pbc(pbc_flags, nd)
    dim_symbols = _normalize_dim_symbols(dimension_symbols, nd)

    grid_emb = grid_embedding_from_active(shape, active)
    n_sites = grid_emb.n
    physics_emb = grid_emb.transformed(origin, scales, shift_first)

    sites = SiteRegistry(n_sites, {"sector": sector_labels})

    hop_graph = HypercubicHopGraph(grid_shape=shape, pbc_flags=pbc,
                                   grid_to_site=grid_emb.grid_to_site(),
                                   site_coords=grid_emb.coords().astype(np.intp))

    fn_registry = {fn.__name__: fn for fn in real_space_functions}

    if internal is None:
        pairs = _required_clifford_pairs(d_components.keys())
        internal_obj = clifford_basis(pairs)
    elif isinstance(internal, int):
        internal_obj = clifford_basis(internal) if internal > 0 else trivial_internal()
    else:
        internal_obj = internal

    mod_fn, mod_keys, mod_wants_ctx = _detect_modifier_signature(hopping_modifier)

    terms = []
    for d_key in sorted(d_components.keys()):
        value = d_components[d_key]
        term = make_operator_term(d_key, value, dim_symbols, fn_registry,
                                  edge_modifier=mod_fn, edge_modifier_keys=mod_keys,
                                  edge_modifier_wants_ctx=mod_wants_ctx,
                                  selector_mask=None)
        if term is not None:
            terms.append(term)

    display_emb = None
    if display_lattice is not None:
        display_emb = _build_display_embedding(display_lattice, sites, grid_emb)

    return Model(sites=sites, hop_graph=hop_graph, physics_embedding=physics_emb,
                 display_embedding=display_emb, internal=internal_obj,
                 terms=terms, vacancy_mask=None,
                 default_params=default_params or {})


def _build_display_embedding(display_lattice, sites, build_grid_emb):
    arr = np.asarray(display_lattice, dtype=int)
    if arr.ndim < 2:
        raise ValueError("display_lattice must be at least D+1 dimensional")
    grid_shape = arr.shape[:-1]
    nd = arr.shape[-1]
    n_full = int(np.prod(grid_shape))
    flat = arr.reshape(n_full, nd, order="F")
    sentinel = (flat < 0).any(axis=1)

    build_coords_T = build_grid_emb.site_to_grid()
    build_to_site = {tuple(c.tolist()): i for i, c in enumerate(build_coords_T)}

    site_for_cell = np.full(n_full, -1, dtype=np.intp)
    for cell, c in enumerate(flat):
        if sentinel[cell]:
            continue
        key = tuple(int(x) for x in c)
        if key not in build_to_site:
            raise ValueError(f"display_lattice cell {cell} references missing build coordinate {key}")
        site_for_cell[cell] = build_to_site[key]

    n_sites = sites.n
    coords = np.zeros((nd, n_sites), dtype=float)
    site_to_grid = np.zeros((n_sites, nd), dtype=np.intp)
    grid_indices = np.indices(grid_shape).reshape(nd, -1, order="F")

    seen = np.zeros(n_sites, dtype=bool)
    for cell in range(n_full):
        s = site_for_cell[cell]
        if s < 0:
            continue
        if seen[s]:
            raise ValueError(f"display_lattice maps multiple cells to site {s}")
        seen[s] = True
        coords[:, s] = grid_indices[:, cell]
        site_to_grid[s] = grid_indices[:, cell]

    if not seen.all():
        missing = np.nonzero(~seen)[0]
        raise ValueError(f"display_lattice does not cover all build sites; missing site(s) {missing.tolist()[:8]}...")

    grid_to_site = site_for_cell.reshape(grid_shape, order="F")
    return GridEmbedding(coords, grid_shape, grid_to_site, site_to_grid)

