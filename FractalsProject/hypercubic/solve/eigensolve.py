import inspect
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def is_hermitian(H, rtol, atol):
    if sps.issparse(H):
        diff = (H - H.getH()).tocoo()
        if diff.nnz == 0:
            return True
        ref = np.maximum(np.abs(H.diagonal()).max(), np.abs(H.data).max() if H.nnz else 0.0)
        tol = atol + rtol * ref
        return np.all(np.abs(diff.data) <= tol)
    return np.allclose(H, H.conj().T, rtol=rtol, atol=atol)


def _filter_kwargs(fn, kwargs):
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _solve_hermitian(H, k, sigma, which, return_eigenvectors, solver_kwargs):
    if k is None:
        if sps.issparse(H):
            H = H.toarray()
        if return_eigenvectors:
            return la.eigh(H, **_filter_kwargs(la.eigh, solver_kwargs))
        return la.eigvalsh(H, **_filter_kwargs(la.eigvalsh, solver_kwargs)), None
    if not sps.issparse(H):
        H = sps.csr_matrix(H)
    kw = dict(k=k, which=which if which is not None else ("SA" if sigma is None else "LM"),
              sigma=sigma, return_eigenvectors=return_eigenvectors)
    kw = _filter_kwargs(spla.eigsh, {**kw, **solver_kwargs})
    out = spla.eigsh(H, **kw)
    if return_eigenvectors:
        w, v = out
    else:
        w, v = out, None
    order = np.argsort(w)
    if v is not None:
        return w[order], v[:, order]
    return w[order], None


def _solve_non_hermitian(H, k, sigma, which, left, right, solver_kwargs):
    if k is None:
        if sps.issparse(H):
            H = H.toarray()
        kw = dict(left=left, right=right)
        kw = _filter_kwargs(la.eig, {**kw, **solver_kwargs})
        out = la.eig(H, **kw)
        if left and right:
            w, vl, vr = out
            return w, vl, vr
        if right:
            w, vr = out
            return w, None, vr
        if left:
            w, vl = out
            return w, vl, None
        return out, None, None
    if not sps.issparse(H):
        H = sps.csr_matrix(H)
    kw = dict(k=k, which=which if which is not None else ("SR" if sigma is None else "LM"),
              sigma=sigma, return_eigenvectors=right)
    kw = _filter_kwargs(spla.eigs, {**kw, **solver_kwargs})
    out = spla.eigs(H, **kw)
    if right:
        w, vr = out
    else:
        w, vr = out, None
    vl = None
    if left:
        kwL = dict(kw)
        out2 = spla.eigs(H.getH(), **kwL)
        if right:
            wL, vlL = out2
        else:
            wL, vlL = out2, None
        # match ordering by sorting both sets the same way
        vl = vlL.conj() if vlL is not None else None
    return w, vl, vr


def solve_model(model, k=None, sigma=None, which=None,
                return_eigenvalues=True, return_eigenvectors=True,
                left=False, right=True,
                hermitian=None, herm_rtol=1e-8, herm_atol=1e-10,
                return_LDOS=False, return_IPR=False,
                biortho=False, solver_kwargs=None, apply_vacancies=True,
                params=None,
                **extra_params):
    from ..report.observables import (build_LDOS, build_IPR,
                                       build_LDOS_biortho, build_IPR_biortho)
    sk = dict(solver_kwargs) if solver_kwargs else {}
    merged_params = {}
    if params:
        merged_params.update(params)
    merged_params.update(extra_params)
    H = model.assemble(apply_vacancies=apply_vacancies, **merged_params)

    if hermitian is None:
        hermitian = is_hermitian(H, herm_rtol, herm_atol)

    result = {"hermitian": bool(hermitian)}

    if hermitian:
        w, v = _solve_hermitian(H, k, sigma, which, return_eigenvectors or return_LDOS or return_IPR, sk)
        if return_eigenvalues:
            result["eigenvalues"] = w
        if return_eigenvectors and v is not None:
            result["eigenvectors"] = v
        if (return_LDOS or return_IPR) and v is not None:
            site_prob = _site_prob(v, model.internal.dim)
            if return_LDOS:
                result["LDOS"] = build_LDOS(site_prob, model)
            if return_IPR:
                result["IPR"] = build_IPR(site_prob)
    else:
        w, vl, vr = _solve_non_hermitian(H, k, sigma, which, left or biortho, right or biortho, sk)
        if return_eigenvalues:
            result["eigenvalues"] = w
        if return_eigenvectors:
            if vr is not None:
                result["eigenvectors_right"] = vr
            if vl is not None:
                result["eigenvectors_left"] = vl
        if return_LDOS or return_IPR:
            if vr is not None:
                pr = _site_prob(vr, model.internal.dim)
                if return_LDOS:
                    result["LDOS_right"] = build_LDOS(pr, model)
                if return_IPR:
                    result["IPR_right"] = build_IPR(pr)
            if vl is not None:
                pl = _site_prob(vl, model.internal.dim)
                if return_LDOS:
                    result["LDOS_left"] = build_LDOS(pl, model)
                if return_IPR:
                    result["IPR_left"] = build_IPR(pl)
            if biortho and vl is not None and vr is not None:
                from ..report.observables import biorthonormalize
                vl_n, vr_n = biorthonormalize(vl, vr)
                pb = _biorth_site_prob(vl_n, vr_n, model.internal.dim)
                if return_LDOS:
                    result["LDOS_biortho"] = build_LDOS_biortho(pb, model)
                if return_IPR:
                    result["IPR_biortho"] = build_IPR_biortho(pb)

    return result


def _site_prob(v, dim):
    nh, k = v.shape
    ns = nh // dim
    arr = v.reshape(ns, dim, k)
    return np.sum(np.abs(arr) ** 2, axis=1)


def _biorth_site_prob(vl, vr, dim):
    nh, k = vr.shape
    ns = nh // dim
    L = vl.reshape(ns, dim, k)
    R = vr.reshape(ns, dim, k)
    return np.sum(L.conj() * R, axis=1).real

