import numpy as np
import scipy.linalg as la

from ..core.embedding import GridEmbedding


def build_LDOS(site_prob, model):
    return _project_to_grid(site_prob, model, model.active_site_indices)


def build_LDOS_partial(site_prob, model, kept_full_indices):
    return _project_to_grid(site_prob, model, kept_full_indices)


def build_LDOS_biortho(site_prob, model):
    return _project_to_grid(site_prob, model, model.active_site_indices, dtype=np.float64)


def _project_to_grid(site_prob, model, full_site_indices, dtype=np.float64):
    n_active, k = site_prob.shape
    emb = model.display_embedding
    if not isinstance(emb, GridEmbedding):
        out = np.full((k, model.sites.n), np.nan, dtype=dtype)
        out[:, full_site_indices] = site_prob.T
        return out

    shape = emb.grid_shape
    site_to_grid = emb.site_to_grid()
    out = np.full((k,) + shape, np.nan, dtype=dtype)
    grid_coords = site_to_grid[full_site_indices]
    idx = tuple(grid_coords.T)
    out[(slice(None),) + idx] = site_prob.T
    return out


def build_IPR(site_prob):
    s = site_prob.sum(axis=0)
    s = np.where(s > 0, s, 1.0)
    return np.sum((site_prob / s) ** 2, axis=0)


def build_IPR_biortho(site_prob):
    return np.sum(site_prob ** 2, axis=0)


def biorthonormalize(vl, vr):
    M = vl.conj().T @ vr
    P, L, U = la.lu(M)
    Linv = la.solve_triangular(L, np.eye(L.shape[0], dtype=L.dtype), lower=True)
    Uinv = la.solve_triangular(U, np.eye(U.shape[0], dtype=U.dtype), lower=False)
    vl_new = vl @ (P @ Linv.conj().T)
    vr_new = vr @ Uinv
    return vl_new, vr_new

