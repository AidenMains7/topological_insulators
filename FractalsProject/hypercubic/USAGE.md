# Hypercubic — Usage Guide

A library for building tight-binding Hamiltonians on hypercubic-grid lattices and arbitrary site graphs, with first-class support for:

- multi-sector / multi-label site spaces
- vacancies (deferred, toggleable)
- periodic / open / mixed boundary conditions
- custom hop topologies (seams, identifications, defect bridges)
- per-site real-space functions
- edge-by-edge hopping modifiers with hop-context
- separation between *physics* and *display* coordinate frames
- Hermitian and non-Hermitian eigensolvers (dense and ARPACK)
- Schur-complement reduction

The package exposes two layers:

- a low-level **kernel** (`hypercubic.core`, `hypercubic.solve`, `hypercubic.report`, `hypercubic.dsl`) for full control, and
- high-level **builders** (`hypercubic.builders`) that recreate the conventional hypercubic-grid workflow with an ergonomic constructor.

Most users will start with `hypercubic_grid_model` and only drop into the kernel when needed.

---

## 1. Quickstart

```python
import numpy as np
import hypercubic as hc

# 1D chain, OBC, two-band Dirac-like
m = hc.hypercubic_grid_model(lattice_shape=10,
                              d_1="cos(kx)",
                              d_2="sin(kx)")
res = m.solve()
print(res["eigenvalues"])
```

---

## 2. The high-level builder: `hypercubic_grid_model`

```python
m = hc.hypercubic_grid_model(
    lattice_shape=None,        # int or tuple of ints
    lattice=None,              # ndarray of sector labels (alt to lattice_shape)
    pbc_flags=False,           # bool or per-axis tuple
    origin=0.0,                # scalar or per-axis vector
    scales=1.0,                # scalar or per-axis vector
    shift_first=True,          # (c-o)*s if True, c*s-o if False
    dimension_symbols="default",
    real_space_functions=(),   # callables for on-site potentials
    hopping_modifier=None,     # global edge transformer
    internal=None,             # InternalStructure or int (clifford pairs)
    display_lattice=None,      # optional D+1 array, see §6
    default_params=None,       # dict of expression parameters
    **d_components,            # d_0=..., d_1=..., d_1_2=..., etc.
)
```

Provide *exactly one* of `lattice_shape` or `lattice`.

When `lattice` is supplied, every entry is a sector label; entries `< 0` mark "no site here" (vacuum, not a vacancy — see §7).

The returned object is a `Model`. It is intentionally mutable in only one way: you can toggle vacancies. All other configuration is fixed at construction.

---

## 3. d-vector components

Each d-component contributes a term

$$d_{\text{key}}(\hat r,\hat k)\,\Gamma_{\text{key}}$$

to the full Hamiltonian. The key encodes a product of gamma matrices: `d_0` → identity, `d_1` → Γ₁, `d_1_2` → Γ₁Γ₂, etc. Higher indices auto-extend the Clifford algebra.

A value can be a string (most common), a dict, or a `(momentum_expr, real_terms)` tuple.

### Momentum strings

Supported tokens:

- `cos(...)`, `sin(...)`, `exp(...)` — `exp(i·…)` semantics for `k`-arguments
- `kx`, `ky`, `kz`, or `kx1`, `kx2`, ... (≥4D)
- arithmetic `+ - * / **`
- parameter substitution via `{name}`

Examples:

```python
d_0="[disorder, strength={W}]"
d_1="sin(kx)"
d_2="{m} - cos(kx)"
d_1_2="cos(kx) + 0.3*cos(2*kx)"
d_3="sin(kx)*sin(ky)"
```

### Real-space terms in strings

Use bracket syntax `coef[func_name, kw1=val1, ...]`:

```python
d_0="[disorder, strength={W}]"
d_2="{m} - cos(kx) + {V}*[harmonic, k=0.1]"
```

The function must be passed in `real_space_functions=(...)` and matched by `__name__`.

### Parameters

Substitute values at solve/assemble time:

```python
res = m.solve(params={"m": 0.5, "V": 1.0})
```

Or pass them as kwargs (avoid names that collide with `solve` kwargs like `k`, `sigma`, `which`):

```python
res = m.solve(m=0.5, V=1.0)
```

---

## 4. Real-space functions and hopping modifiers

### Real-space functions

```python
def harmonic(coords, k=1.0):
    return k * np.sum(coords**2, axis=0)
```

`coords` has shape `(nd, n_sites)` in the **physics** frame (after `origin`, `scales`, `shift_first`).

### Hopping modifiers

```python
def flux(val, src, dst, ctx=None, phi=0.0):
    return val * np.exp(1j * phi * (dst[1] - src[1]))
```

`src`, `dst` have shape `(nd, n_hops)` in the physics frame. Adding a `ctx` keyword opts into receiving a `HopContext` (see kernel section).

Extra keyword arguments after the third positional are forwarded from `params`.

---

## 5. Vacancies

Vacancies are toggled on the model after construction; they are not declared up front.

```python
mask = np.zeros(m.n_sites, dtype=bool)
mask[0] = mask[5] = True
m.set_vacancies(mask)              # rebuild internal cache invalidated
m.add_vacancies(other_mask)        # union with current
m.clear_vacancies()                # reset
m.vacancies_from_label(sector=1)   # convenience: query sites by label
```

When vacancies are active:

- `m.assemble()` returns the Hamiltonian with vacancy rows/columns removed.
- `m.solve()` operates on the reduced Hamiltonian.
- `LDOS` arrays put `NaN` at vacancy sites.
- `m.coordinates(active_only=True)` (default) returns only kept sites.

Pass `apply_vacancies=False` to either `assemble` or `solve` to ignore the mask without clearing it.

---

## 6. Direct assembly

`Model.assemble` returns the Hamiltonian matrix directly, without solving.

```python
H = model.assemble(
    apply_vacancies=True,   # strip vacancy rows/columns
    format='csr',           # output format (see below)
    **params,               # expression parameters, same as solve
)
```

### Output format

The `format` keyword controls the return type:

| `format` | Return type |
|---|---|
| `'csr'` *(default)* | `scipy.sparse.csr_matrix` |
| `'csc'` | `scipy.sparse.csc_matrix` |
| `'coo'` | `scipy.sparse.coo_matrix` |
| `'bsr'` | `scipy.sparse.bsr_matrix` |
| `'dia'` | `scipy.sparse.dia_matrix` |
| `'dok'` | `scipy.sparse.dok_matrix` |
| `'lil'` | `scipy.sparse.lil_matrix` |
| `'dense'` | `numpy.ndarray` (via `.toarray()`) |

```python
H_csr   = model.assemble()                       # default CSR
H_csc   = model.assemble(format='csc')
H_dense = model.assemble(format='dense')
H_dense = model.assemble(format='dense', m=1.5)  # with params
```

The internal assembly cache always stores a CSR matrix; format conversion is applied after cache retrieval. For `format='csr'` the cached object is returned directly with no copy.

---

## 7. Display lattice (the "second lattice")

A model can carry two lattice embeddings:

- the **physics embedding** (coordinates seen by callbacks and used to build hops),
- the **display embedding** (coordinates used for LDOS, plots, and reported positions).

If you do not supply one, the display embedding equals the physics embedding.

To supply one, pass a `(D+1)-dimensional` integer array `display_lattice` of shape `(*display_shape, D)`. Each cell of the display grid stores the *integer construction-grid coordinate* of the site it should represent. Use any negative integer in any component to mark a cell as empty.

Constraints:

- Every active build site must appear in exactly one display cell (bijection).
- Each display cell must reference an existing build site, or be marked empty.

Example (4×4 identity remap):

```python
disp = np.zeros((4, 4, 2), dtype=int)
for j in range(4):
    for i in range(4):
        disp[j, i] = (i, j)
m = hc.hypercubic_grid_model(lattice_shape=(4,4), display_lattice=disp,
                              d_1="cos(kx)", d_2="cos(ky)")
```

LDOS arrays are then placed on the display grid, so `res["LDOS"]` has shape `(k, *display_shape)`.

---

## 8. Solving

```python
res = m.solve(
    k=None,                 # ARPACK count; None → dense
    sigma=None,             # shift-invert target
    which=None,             # ARPACK selection criterion
    return_eigenvalues=True,
    return_eigenvectors=True,
    left=False, right=True, # non-Hermitian only
    hermitian=None,         # None → auto-detect
    herm_rtol=1e-8, herm_atol=1e-10,
    return_LDOS=False,
    return_IPR=False,
    biortho=False,          # non-Hermitian biorthogonal observables
    solver_kwargs=None,
    apply_vacancies=True,
    params=None,            # dict; preferred to avoid kwarg collisions
    **extra_params,         # only safe names
)
```

### Result keys

Hermitian:

- `"eigenvalues"`, `"eigenvectors"`, `"LDOS"`, `"IPR"`, `"hermitian"`.

Non-Hermitian:

- `"eigenvalues"`, `"eigenvectors_left"`, `"eigenvectors_right"`,
  `"LDOS_left"`, `"LDOS_right"`, `"IPR_left"`, `"IPR_right"`,
  `"LDOS_biortho"`, `"IPR_biortho"`, `"hermitian"`.

`LDOS` arrays have shape `(num_states, *display_grid_shape)` with `NaN` at vacancies and cells that have no site.

---

## 9. Schur-complement reduction

```python
res = m.solve_schur(
    eliminate_label="sector",
    eliminate_value=1,        # int or list of values
    energy=0.0,               # E in (E·I − H_BB)^{-1}
    k=None, sigma=None, which=None,
    return_eigenvalues=True, return_eigenvectors=True,
    return_LDOS=False, return_IPR=False,
    hermitian=None,
    apply_vacancies=True,
    params=None,
)
```

The set of eliminated sites is a **label query**, not a hard-coded sector list — any registered label can be used. LDOS arrays are mapped onto kept sites only; eliminated and vacant sites both appear as `NaN`.

`exclude_sectors`-style behavior is recovered by combining `set_vacancies` with `solve_schur` or `solve`.

---

## 10. Coordinates and properties

```python
m.n_sites               # total registered sites
m.n_active_sites        # ignoring vacancies
m.internal_dim
m.hilbert_dim
m.active_hilbert_dim
m.active_site_indices
m.active_hilbert_indices
m.vacancy_mask
m.coordinates(frame="display", active_only=True)   # (n, nd)
m.terms                 # tuple of OperatorTerm
m.sites.labels("sector")
m.sites.select(sector=0)
m.sites.mask(sector=0)
```

---

## 11. Custom hop topologies (low-level)

For grain-boundary stitching, identified surfaces, defect bridges, twisted boundary conditions beyond standard PBC, etc., wrap the default hop graph in an `OverlayHopGraph`:

```python
from hypercubic import OverlayHopGraph, ChannelEdges

extra = {
    (1, 0): ChannelEdges(src=np.array([12, 24]), dst=np.array([93, 81]),
                          meta={}),
    # Hermitian conjugate goes in the (-1, 0) channel:
    (-1, 0): ChannelEdges(src=np.array([93, 81]), dst=np.array([12, 24]),
                           meta={}),
}
m.hop_graph = OverlayHopGraph(m.hop_graph, extra, suppress_channels=())
m._invalidate_cache()
```

Whatever channel keys you add must match channels referenced by your d-components. The momentum DSL emits signed integer tuples — `cos(kx)` references `(1,0,...)` and `(-1,0,...)`, `cos(2*kx)` references `(2,0,...)` and `(-2,0,...)`, and so on.

---

## 12. The kernel API

If you need to bypass the high-level builder entirely, build a model from primitives:

```python
from hypercubic import (SiteRegistry, GridEmbedding, HypercubicHopGraph,
                        clifford_basis, OperatorTerm, Model,
                        grid_embedding_from_active)
```

A `Model` accepts:

- `sites` — `SiteRegistry`
- `hop_graph` — any `HopGraph` (hypercubic, explicit, overlay, …)
- `physics_embedding` — `Embedding`
- `display_embedding` — optional `Embedding`
- `internal` — `InternalStructure`
- `terms` — list of `OperatorTerm`
- `vacancy_mask` — optional bool array of length `n_sites`
- `default_params` — dict of expression parameters

This is the path to take when implementing new high-level builders or unusual constructions.

---

## 13. Biorthonormalization utility

```python
from hypercubic import biorthonormalize
vl_n, vr_n = biorthonormalize(vl, vr)   # vl_n.conj().T @ vr_n ≈ I
```

