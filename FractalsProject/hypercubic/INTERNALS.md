# Hypercubic — Internals

Architecture, data flow, and conventions for developers (human or AI).

---

## Layered architecture

```
hypercubic/
    core/        # primitive data types
        sites.py          # SiteRegistry
        embedding.py      # Embedding, GridEmbedding
        hopgraph.py       # HopGraph + variants (ChannelEdges)
        internal.py       # InternalStructure, clifford_basis
        operators.py      # OperatorTerm, assemble_term, HopContext
        model.py          # Model
    solve/       # eigensolvers, reductions
        eigensolve.py
        schur.py
    report/      # observables on top of solver output
        observables.py
    dsl/         # user-facing DSL (only entrypoint with parsing)
        expr.py           # safe arithmetic + {param} substitution
        momentum.py       # momentum string -> {channel: coef}
        dvector.py        # d_* keys -> OperatorTerm
    builders/    # high-level convenience constructors
        hypercubic.py
    USAGE.md
    INTERNALS.md
    STANDARDS.md
```

Dependency direction is strictly downward: `dsl` and `builders` depend on `core`; `solve` and `report` depend only on `core`. Nothing in `core` imports from above.

---

## Layer 0: `SiteRegistry`

A flat list of `n` sites with named label arrays.

- `_n: int`
- `_labels: dict[str, ndarray (n,)]`

The only universal site identity is the integer index `0..n-1`. All filtering, override, exclusion, and elimination uses **label queries** via `mask(**eq)` / `select(**eq)`. A query supports either scalar equality or membership in a sequence of values (vectorized via `np.isin`).

Sectors are no longer special-cased anywhere in the library — they are simply a label called `"sector"` populated by the high-level builder. New labels can be attached freely.

---

## Layer 1: Embeddings

`Embedding` is a thin wrapper around an `(nd, n_sites)` float coordinate array. It supports the canonical transform

```
shifted = (c - origin) * scales       # shift_first=True
shifted = c * scales - origin         # shift_first=False
```

via `transformed(...)`, returning a new immutable embedding.

`GridEmbedding` extends `Embedding` with a regular grid layout:

- `_shape: tuple[int]`             — display/lattice grid shape
- `_grid_to_site: ndarray (shape, dtype=intp)` — `-1` for cells with no site
- `_site_to_grid: ndarray (n, nd, dtype=intp)` — inverse map

`GridEmbedding.transformed(...)` preserves grid metadata. Two helpers build grid embeddings from a shape (`grid_embedding_from_shape`) or from a boolean active-mask (`grid_embedding_from_active`).

The high-level builder always constructs both:

- `physics_embedding` — float coords after `(origin, scales, shift_first)`, used by callbacks
- `display_embedding` — either inherited from physics or built from a user `display_lattice` (D+1 array)

`Model.display_embedding` is what `report.observables._project_to_grid` writes onto.

---

## Layer 2: `HopGraph`

A `HopGraph` is a shift-channel-labeled directed multigraph. It exposes a single read interface:

```python
edges(channel) -> ChannelEdges(src, dst, meta)
```

`ChannelEdges` is a `NamedTuple`:

- `src, dst`: `ndarray (n_hops,)` of site indices
- `meta`: `dict[str, ndarray]` — arbitrary per-edge metadata

Channels are arbitrary hashable keys at the kernel level; the high-level DSL emits `tuple[int]` signed shifts.

Three concrete graphs:

| Class | Behavior |
|---|---|
| `ExplicitHopGraph` | dict-backed; user supplies edges per channel |
| `HypercubicHopGraph` | lazy: builds shift-grouped edges from `(shape, pbc_flags, grid_to_site, coords)` on first access; caches per channel |
| `OverlayHopGraph` | composes a base graph with overlays / suppressions per channel |

`HypercubicHopGraph._build(signed)`:

1. Computes `cf = coords + signed`.
2. Computes a `cross_bits` mask per site (which axes wrap).
3. Uses the boundary mask to filter out edges that cross OBC axes.
4. Wraps remaining `cf` modulo `L` and looks up destination site via `grid_to_site`.
5. Computes winding numbers and stores them under `meta["winding"]`.

The result is cached in `_cache[signed]`.

`OverlayHopGraph` concatenates `src/dst/meta` arrays. If a channel is in `suppress_channels`, the base graph contribution for that channel is dropped.

---

## Layer 3: `InternalStructure`

Holds the (possibly trivial) internal Hilbert space.

- `_basis: ndarray (m, dim, dim)`
- `product(indices)` — caches Γ-products keyed by tuple of indices
- `clifford_basis(num_pairs)` builds `2n+1` anticommuting matrices via iterated Pauli Kronecker products

`trivial_internal()` provides a 1×1 identity for cases without internal structure.

---

## Layer 4: `OperatorTerm` and `assemble_term`

`OperatorTerm` is a flat data container:

| Field | Meaning |
|---|---|
| `hops_factory` | `params -> dict[channel, complex]` (lazy reparse of momentum expr each assemble) |
| `site_terms` | list of `(fn, kwargs_factory, coef_factory)` |
| `edge_modifier` | optional callable, signature `(val, src_coords, dst_coords, [ctx], **extras)` |
| `edge_modifier_keys` | tuple of param names to forward |
| `edge_modifier_wants_ctx` | bool, detected from signature |
| `gamma_indices` | tuple of internal-basis indices to multiply |
| `name` | str, typically the d-key |
| `selector_mask` | optional bool array of length `n_sites` for sector-pair-style restrictions |

`assemble_term(term, hop_graph, embedding, internal, params, n_sites)`:

1. Calls `hops_factory(params)` to get a fresh `{channel: coef}` map.
2. For each non-zero channel, queries `hop_graph.edges(channel)` for cached `(src, dst, meta)`.
3. Optionally applies `selector_mask` to filter src/dst pairs.
4. Optionally applies `edge_modifier` (with or without `ctx`).
5. Accumulates COO entries.
6. Evaluates each `site_term` to add to a diagonal vector (also masked).
7. Builds a CSR spatial matrix.
8. Tensors with `internal.product(gamma_indices)` (skipping the `kron` when `internal.dim == 1`).

The `hops_factory` indirection is what allows the Fourier coefficients to be re-evaluated cheaply on every parameter change while the underlying `hop_graph.edges(channel)` arrays stay cached.

---

## Layer 5: `Model`

`Model` is the only stateful kernel object. State is limited to:

- `_vacancy_mask: ndarray (n_sites,)` — toggleable
- `_default_params: dict` — convenience defaults
- `_assemble_cache_(key|value)` — single-slot memoization keyed by `(apply_vacancies, sorted params, vacancy bytes)`

Everything else is immutable.

`assemble(apply_vacancies=True, format='csr', **params)`:

1. Validates `format` against `_SPARSE_FORMATS ∪ {'dense'}`.
2. Merges `_default_params` and `params`.
3. Computes a cache key. Returns the cached H (post-converted to `format`) if it matches.
4. Builds `H = sum(assemble_term(...) for term in terms)` over the **full** site set.
5. If `apply_vacancies` and any vacancies are flagged, slices to `keep_h = repeat(active, internal_dim)` rows/columns. CSR slicing is cheap.
6. Caches the CSR matrix, then converts: `H.toarray()` for `format='dense'`, `H.asformat(format)` otherwise (returns `self` for `format='csr'`).
7. Returns the converted matrix.

`set_vacancies` / `add_vacancies` / `clear_vacancies` invalidate the cache.

`coordinates(frame, active_only)`:

- `frame="display"` (default) → `display_embedding` coords.
- `frame="physics"` → `physics_embedding` coords.
- `active_only=True` filters out vacancy sites.

---

## Solve layer

`solve_model(model, ...)`:

1. `H = model.assemble(...)`.
2. Hermiticity auto-detected via `is_hermitian(H, rtol, atol)` if `hermitian is None`.
   - Sparse path computes `(H - H.getH()).tocoo()` and compares element-wise against `atol + rtol * max(|H|)`.
3. Dispatches to `_solve_hermitian` (uses `eigh`/`eigvalsh`/`eigsh`) or `_solve_non_hermitian` (`eig`/`eigs`).
4. `_filter_kwargs` introspects each scipy function and forwards only accepted keys.
5. If LDOS / IPR requested, computes site probabilities by reshaping `(n_active * internal_dim, k)` → `(n_active, internal_dim, k)` and summing the internal axis.
6. `report.observables.build_LDOS` calls `_project_to_grid`, which uses `display_embedding.site_to_grid()` and a single fancy-index assignment (no F-order reshapes — those return copies on C-contiguous arrays and silently lose writes).

`schur_solve(model, eliminate_label, eliminate_value, energy, ...)`:

1. Assembles the (vacancy-reduced) Hamiltonian.
2. Computes `eliminate_site_mask = isin(model.sites.labels(eliminate_label), eliminate_value)` over **active** sites (so vacancy semantics compose).
3. Builds A/B Hilbert masks via `np.repeat(mask, internal.dim)`.
4. Slices `H_AA`, `H_BB`, `H_AB`, `H_BA`. `H_AA` becomes dense (output is dense anyway), the others stay sparse.
5. Factorizes `M = E·I − H_BB` (or `−H_BB` for `E=0`) with `splu` and computes `X = M^{-1} H_BA`.
6. `H_eff = H_AA − H_AB X`.
7. Re-detects Hermiticity on `H_eff` and dispatches to the same eigensolvers.
8. `build_LDOS_partial` maps eigenvectors back onto kept site coordinates, leaving eliminated and vacant sites as `NaN`.

---

## DSL layer

### `expr.safe_eval`

AST-based evaluator supporting `+ - * / ** unary` on numeric literals and named variables only. No function calls, attribute access, subscript, or comprehension. `j` resolves to `1j` if not shadowed.

### `expr.safe_format`

Replaces `{...}` placeholders by stringifying `safe_eval(inner, params)` results. Used for both momentum-string preprocessing and site-term kwargs.

### `momentum.parse_momentum(expr, dim_symbols, params)`

A small recursive-descent parser. Internal representation is one of:

- a **shift dict** `dict[tuple[int], complex]` — Fourier expansion already realized
- a **k-marker** `{"_k": [(dim_idx, multiplier), ...], "const": complex}` — a linear combination of momentum coordinates that has not yet been wrapped by `cos/sin/exp`

Arithmetic between these two representations is restricted: a k-marker may only multiply or divide by scalars (zero-shift dicts), and may only combine with shift dicts via addition of scalars. Wrapping a k-marker in `cos`, `sin`, or `exp` produces shift entries (Euler decomposition); `exp` is interpreted as `exp(i·…)` so that `exp(i*kx)` produces `{(1,): 1}` rather than a non-Fourier expression.

This intentionally rejects ambiguous expressions like `kx + cos(kx)` early.

### `dvector.split_dstring`

Splits a d-value string into a momentum part and a list of `(coef_expr, body)` real-space-term entries via the bracket regex.

### `dvector.make_operator_term(d_key, value, dim_symbols, registered_fns, ...)`

Normalizes the three accepted value formats (string, dict, `(momentum, real_terms)` tuple) into the structured `OperatorTerm`. Site-term coefficient and kwargs expressions are wrapped in `make_coef_fn` / `make_kwargs_fn` closures, so they re-evaluate per assemble.

### High-level builder

`builders.hypercubic.hypercubic_grid_model`:

1. Resolves `lattice_shape` vs `lattice` to a `(shape, active_mask, sector_labels)`.
2. Builds a `GridEmbedding` from the active mask, then derives the physics embedding via `transformed(origin, scales, shift_first)`.
3. Constructs `SiteRegistry` with a single `"sector"` label.
4. Builds a `HypercubicHopGraph` from the shape and PBC flags.
5. Computes the required Clifford-pair count from the d-keys (or accepts an explicit `internal`).
6. Detects the hopping modifier signature: extra named kwargs become `edge_modifier_keys` (forwarded from params), and a `ctx` keyword turns on context delivery.
7. For each d-key, calls `make_operator_term` to produce an `OperatorTerm`.
8. Optionally validates and builds a `display_embedding` from a `display_lattice` D+1 array.

The returned `Model` is the only object the user interacts with after this point.

---

## Caching strategy

| Cached | Where | Invalidation |
|---|---|---|
| Hop edges per channel | `HypercubicHopGraph._cache` | never (graph is immutable) |
| Γ-matrix products | `InternalStructure._product_cache` | never |
| Last-assembled Hamiltonian | `Model._assemble_cache_*` | `set_vacancies`, `add_vacancies`, `clear_vacancies` |

The cache always stores CSR; `format` conversion is applied after retrieval and is not part of the cache key.

What is **not** cached:

- Fourier coefficient evaluations (cheap; depend on params).
- Real-space function evaluations (depend on params).
- Vacancy-mask slicing of the assembled Hamiltonian (cheap CSR slice).

This split exists so that parameter sweeps reuse expensive structural work but always reflect the current parameter values.

---

## Conventions and contracts

- Sites are integer-indexed `0..n-1`; index identity is permanent.
- Channels are arbitrary hashable keys; signed integer tuples are the convention.
- Vacancies are state on the model, never on the graph or embedding.
- Embeddings never alter graph topology; they only define coordinates.
- Real-space functions receive **physics-frame** coordinates.
- Hopping modifiers receive **physics-frame** source/target coordinates.
- LDOS is reported in the **display-frame** grid.
- Kernel functions assume canonical inputs; validation is the responsibility of `dsl`, `builders`, and the entry-point methods of user-facing classes.

---

## Adding new builders

A new builder typically:

1. Constructs a `SiteRegistry` with appropriate labels.
2. Constructs an `Embedding` (probably a `GridEmbedding` if any LDOS will be reported).
3. Constructs a `HopGraph` (often `ExplicitHopGraph` or `OverlayHopGraph` over a base `HypercubicHopGraph`).
4. Constructs `OperatorTerm`s, possibly via `dsl.dvector.make_operator_term` for ergonomic momentum + real-space-term parsing.
5. Returns a `Model`.

The grain-boundary stitching from earlier work, for example, becomes a builder that:

- builds a hypercubic grid with a vacancy strip masked from the active set,
- constructs an `OverlayHopGraph` adding the across-seam hops on the channels referenced by the desired d-vector momentum components,
- supplies a `display_lattice` mapping the build coordinates onto the visually connected layout.

---

## Adding new solvers / observables

Solvers operate on `Model` and respect its public contract (`assemble`, `vacancy_mask`, `internal_dim`, `sites`, `display_embedding`). New observables should accept an already-computed eigenvector matrix and the model — see `report.observables` for the pattern.

