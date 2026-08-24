import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .eigensolve import _solve_hermitian, _solve_non_hermitian, is_hermitian, _site_prob


def schur_solve(model, eliminate_label, eliminate_value, energy=0.0,
                k=None, sigma=None, which=None,
                return_eigenvalues=True, return_eigenvectors=True,
                hermitian=None, herm_rtol=1e-8, herm_atol=1e-10,
                return_LDOS=False, return_IPR=False,
                solver_kwargs=None, apply_vacancies=True,
                params=None, **extra_params):
    from ..report.observables import build_LDOS_partial, build_IPR
    sk = dict(solver_kwargs) if solver_kwargs else {}
    merged_params = {}
    if params:
        merged_params.update(params)
    merged_params.update(extra_params)
    H = model.assemble(apply_vacancies=apply_vacancies, **merged_params)
    d = model.internal.dim

    label_arr = model.sites.labels(eliminate_label)
    eliminate_site_mask = np.isin(label_arr, eliminate_value)
    if apply_vacancies and model.vacancy_mask.any():
        active = ~model.vacancy_mask
        eliminate_site_mask = eliminate_site_mask[active]
        kept_site_indices = np.nonzero(active)[0]
    else:
        kept_site_indices = np.arange(model.sites.n)

    keep_site_mask = ~eliminate_site_mask

    A_h = np.repeat(keep_site_mask, d)
    B_h = np.repeat(eliminate_site_mask, d)

    H_AA = H[A_h][:, A_h]
    H_BB = H[B_h][:, B_h]
    H_AB = H[A_h][:, B_h]
    H_BA = H[B_h][:, A_h]

    nB = H_BB.shape[0]
    if nB == 0:
        H_eff = H_AA.toarray() if sps.issparse(H_AA) else H_AA
    else:
        if energy == 0.0:
            M = -H_BB
        else:
            M = energy * sps.identity(nB, dtype=np.complex128, format="csr") - H_BB
        lu = spla.splu(M.tocsc())
        X = lu.solve(H_BA.toarray() if sps.issparse(H_BA) else H_BA)
        H_AA_dense = H_AA.toarray() if sps.issparse(H_AA) else H_AA
        H_eff = H_AA_dense - (H_AB @ X)

    if hermitian is None:
        hermitian = is_hermitian(H_eff, herm_rtol, herm_atol)

    result = {"hermitian": bool(hermitian), "eliminated_count": int(eliminate_site_mask.sum())}

    if hermitian:
        w, v = _solve_hermitian(H_eff, k, sigma, which, return_eigenvectors or return_LDOS or return_IPR, sk)
        if return_eigenvalues:
            result["eigenvalues"] = w
        if return_eigenvectors and v is not None:
            result["eigenvectors"] = v
        if (return_LDOS or return_IPR) and v is not None:
            site_prob = _site_prob(v, d)
            kept_full_indices = kept_site_indices[keep_site_mask]
            if return_LDOS:
                result["LDOS"] = build_LDOS_partial(site_prob, model, kept_full_indices)
            if return_IPR:
                result["IPR"] = build_IPR(site_prob)
    else:
        w, vl, vr = _solve_non_hermitian(H_eff, k, sigma, which, False, return_eigenvectors or return_LDOS or return_IPR, sk)
        if return_eigenvalues:
            result["eigenvalues"] = w
        if return_eigenvectors and vr is not None:
            result["eigenvectors_right"] = vr
        if (return_LDOS or return_IPR) and vr is not None:
            pr = _site_prob(vr, d)
            kept_full_indices = kept_site_indices[keep_site_mask]
            if return_LDOS:
                result["LDOS_right"] = build_LDOS_partial(pr, model, kept_full_indices)
            if return_IPR:
                result["IPR_right"] = build_IPR(pr)

    return result

