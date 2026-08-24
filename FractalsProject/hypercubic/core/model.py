import numpy as np
import scipy.sparse as sps

from .operators import assemble_term

_SPARSE_FORMATS = frozenset({'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'})


class Model:
    __slots__ = ("sites", "hop_graph", "physics_embedding", "display_embedding",
                 "internal", "_terms", "_vacancy_mask", "_default_params",
                 "_term_index", "_assemble_cache_key", "_assemble_cache_value")

    def __init__(self, sites, hop_graph, physics_embedding, display_embedding,
                 internal, terms, vacancy_mask, default_params):
        self.sites = sites
        self.hop_graph = hop_graph
        self.physics_embedding = physics_embedding
        self.display_embedding = display_embedding if display_embedding is not None else physics_embedding
        self.internal = internal
        self._terms = list(terms)
        self._term_index = {t.name: i for i, t in enumerate(self._terms) if t.name is not None}
        self._vacancy_mask = vacancy_mask if vacancy_mask is not None else np.zeros(sites.n, dtype=bool)
        self._default_params = dict(default_params)
        self._assemble_cache_key = None
        self._assemble_cache_value = None

    # ---------------------------------------------------------------- properties

    @property
    def n_sites(self):
        return self.sites.n

    @property
    def n_active_sites(self):
        return int((~self._vacancy_mask).sum())

    @property
    def internal_dim(self):
        return self.internal.dim

    @property
    def hilbert_dim(self):
        return self.sites.n * self.internal.dim

    @property
    def active_hilbert_dim(self):
        return self.n_active_sites * self.internal.dim

    @property
    def vacancy_mask(self):
        return self._vacancy_mask

    @property
    def active_mask(self):
        return ~self._vacancy_mask

    @property
    def active_site_indices(self):
        return np.nonzero(~self._vacancy_mask)[0]

    @property
    def active_hilbert_indices(self):
        d = self.internal.dim
        sites = np.nonzero(~self._vacancy_mask)[0]
        return (sites[:, None] * d + np.arange(d)[None, :]).ravel()

    @property
    def terms(self):
        return tuple(self._terms)

    # ---------------------------------------------------------------- vacancies

    def set_vacancies(self, mask):
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.sites.n,):
            raise ValueError(f"vacancy mask must have shape ({self.sites.n},)")
        self._vacancy_mask = m
        self._invalidate_cache()

    def add_vacancies(self, mask):
        self.set_vacancies(self._vacancy_mask | np.asarray(mask, dtype=bool))

    def clear_vacancies(self):
        self._vacancy_mask = np.zeros(self.sites.n, dtype=bool)
        self._invalidate_cache()

    def vacancies_from_label(self, **eq):
        self.set_vacancies(self.sites.mask(**eq))

    # ---------------------------------------------------------------- assembly

    def assemble(self, apply_vacancies=True, format='csr', **params):
        if format != 'dense' and format not in _SPARSE_FORMATS:
            raise ValueError(
                f"unknown format '{format}'; expected one of "
                f"{sorted(_SPARSE_FORMATS)} or 'dense'"
            )
        merged = dict(self._default_params)
        merged.update(params)
        cache_key = (apply_vacancies, tuple(sorted(merged.items(), key=lambda kv: kv[0])),
                     self._vacancy_mask.tobytes() if apply_vacancies else None)
        if self._assemble_cache_key == cache_key:
            H = self._assemble_cache_value
        else:
            n = self.sites.n
            H = sps.csr_matrix((n * self.internal.dim, n * self.internal.dim), dtype=np.complex128)
            for term in self._terms:
                H = H + assemble_term(term, self.hop_graph, self.physics_embedding,
                                      self.internal, merged, n)
            H = H.tocsr()

            if apply_vacancies and self._vacancy_mask.any():
                keep = ~self._vacancy_mask
                keep_h = np.repeat(keep, self.internal.dim)
                H = H[keep_h][:, keep_h]

            self._assemble_cache_key = cache_key
            self._assemble_cache_value = H

        if format == 'dense':
            return H.toarray()
        return H.asformat(format)

    def _invalidate_cache(self):
        self._assemble_cache_key = None
        self._assemble_cache_value = None

    # ---------------------------------------------------------------- coordinates

    def coordinates(self, frame="display", active_only=True):
        emb = self._embedding_for(frame)
        c = emb.coords()
        if active_only and self._vacancy_mask.any():
            c = c[:, ~self._vacancy_mask]
        return c.T

    def _embedding_for(self, frame):
        if frame == "display":
            return self.display_embedding
        if frame == "physics":
            return self.physics_embedding
        raise ValueError(f"unknown frame '{frame}'")

    # ---------------------------------------------------------------- solve hooks

    def solve(self, **kwargs):
        from ..solve.eigensolve import solve_model
        return solve_model(self, **kwargs)

    def solve_schur(self, eliminate_label, eliminate_value, **kwargs):
        from ..solve.schur import schur_solve
        return schur_solve(self, eliminate_label, eliminate_value, **kwargs)

