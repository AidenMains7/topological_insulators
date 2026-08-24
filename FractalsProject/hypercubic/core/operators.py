from typing import NamedTuple
import numpy as np
import scipy.sparse as sps


class HopContext(NamedTuple):
    channel: tuple
    src: np.ndarray
    dst: np.ndarray
    src_coords: np.ndarray
    dst_coords: np.ndarray
    meta: dict


class OperatorTerm:
    __slots__ = ("hops_factory", "site_terms", "edge_modifier", "edge_modifier_keys",
                 "edge_modifier_wants_ctx", "gamma_indices", "name", "selector_mask")

    def __init__(self, hops_factory, site_terms, edge_modifier, edge_modifier_keys,
                 edge_modifier_wants_ctx, gamma_indices, name, selector_mask):
        self.hops_factory = hops_factory
        self.site_terms = site_terms
        self.edge_modifier = edge_modifier
        self.edge_modifier_keys = edge_modifier_keys
        self.edge_modifier_wants_ctx = edge_modifier_wants_ctx
        self.gamma_indices = gamma_indices
        self.name = name
        self.selector_mask = selector_mask


def assemble_term(term, hop_graph, embedding, internal, params, n_sites):
    coords = embedding.coords()
    rows_list, cols_list, vals_list = [], [], []

    if term.hops_factory is not None:
        shift_coefs = term.hops_factory(params)
        for channel, coef in shift_coefs.items():
            if coef == 0:
                continue
            edges = hop_graph.edges(channel)
            if edges.src.size == 0:
                continue
            src_idx = edges.src
            dst_idx = edges.dst

            if term.selector_mask is not None:
                sel = term.selector_mask
                keep = sel[src_idx] & sel[dst_idx]
                if not keep.any():
                    continue
                if not keep.all():
                    src_idx = src_idx[keep]
                    dst_idx = dst_idx[keep]
                    edges_meta = {k: (v[..., keep] if hasattr(v, 'ndim') and v.ndim > 0 else v)
                                  for k, v in edges.meta.items()}
                else:
                    edges_meta = edges.meta
            else:
                edges_meta = edges.meta

            if term.edge_modifier is not None:
                sc = coords[:, src_idx]
                dc = coords[:, dst_idx]
                mod_kwargs = {k: params[k] for k in term.edge_modifier_keys if k in params}
                if term.edge_modifier_wants_ctx:
                    ctx = HopContext(channel=channel, src=src_idx, dst=dst_idx,
                                     src_coords=sc, dst_coords=dc, meta=edges_meta)
                    v = term.edge_modifier(coef, sc, dc, ctx=ctx, **mod_kwargs)
                else:
                    v = term.edge_modifier(coef, sc, dc, **mod_kwargs)
                if v is None:
                    continue
                v = np.asarray(v, dtype=np.complex128)
                if v.shape != src_idx.shape:
                    raise ValueError("edge_modifier must return a 1D array matching the number of hops")
            else:
                v = np.full(src_idx.shape, coef, dtype=np.complex128)

            rows_list.append(src_idx)
            cols_list.append(dst_idx)
            vals_list.append(v)

    diag = None
    if term.site_terms:
        diag = np.zeros(n_sites, dtype=np.complex128)
        for fn, kwargs_factory, coef_factory in term.site_terms:
            coef = coef_factory(params)
            if coef == 0:
                continue
            kwargs = kwargs_factory(params)
            raw = fn(coords, **kwargs)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.complex128)
            if arr.shape != (n_sites,):
                raise ValueError(f"site term '{getattr(fn,'__name__',fn)}' must return shape ({n_sites},)")
            if term.selector_mask is not None:
                arr = arr * term.selector_mask
            diag += coef * arr

    if rows_list:
        spatial = sps.coo_matrix(
            (np.concatenate(vals_list),
             (np.concatenate(rows_list), np.concatenate(cols_list))),
            shape=(n_sites, n_sites),
        ).tocsr()
    else:
        spatial = sps.csr_matrix((n_sites, n_sites), dtype=np.complex128)

    if diag is not None and np.any(diag):
        spatial = spatial + sps.diags(diag, format="csr")

    if internal.dim == 1:
        gamma = internal.product(term.gamma_indices)
        return spatial * gamma[0, 0]
    gamma = internal.product(term.gamma_indices)
    return sps.kron(spatial, sps.csr_matrix(gamma), format="csr")

