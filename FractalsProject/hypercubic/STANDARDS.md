# Hypercubic — Coding Standards

Project-wide conventions. Exceptions are listed at the bottom.

---

## 1. Layered architecture and dependency direction

The package is split into orthogonal layers:

```
core   →   solve, report, dsl, builders
                 ↘             ↗
                  builders may use dsl
```

Dependencies run **downward only**. `core` imports nothing from `solve`, `report`, `dsl`, or `builders`. `solve` and `report` import only from `core`. `dsl` imports only from `core`. `builders` may import from `core` and `dsl`.

The `Model` class breaks the rule one place: its `solve` and `solve_schur` shortcuts import `solve.*` lazily (function-body imports) to avoid an import cycle without forcing solver code on every model load.

---

## 2. Validation and type checking

**Validate at API boundaries; assume canonical inputs everywhere else.**

- Public class `__init__`s, public methods, and any function in `builders` or `dsl` that accepts user input may validate, normalize, and convert types.
- Internal helpers (anything in `core/*` called by other internals, plus all `_`-prefixed functions) take pre-validated inputs and do **not** re-check them. They are deterministic on every branch with no fallback paths for malformed data.

The pragmatic implication: if you find yourself adding a defensive type check or `np.asarray(...)` deep inside the kernel, push it up to the boundary that produced the bad value.

---

## 3. Defaults

**Defaults appear only on user-facing signatures.** Internal helpers, factories, and constructors of internal-only types take all arguments positionally with no default values. This forces every internal call site to be explicit.

User-facing signatures (`Model.solve`, `hypercubic_grid_model`, `Embedding.__init__`, `SiteRegistry.__init__`, etc.) carry defaults where they are ergonomic — and only there.

---

## 4. Documentation

**No docstrings or block comments inside source modules**, except:

- short single-line annotations explaining a non-obvious local choice, or
- region separators (e.g. `# ---------------- assembly` style headers).

All user documentation lives in `USAGE.md`. All architectural documentation lives in `INTERNALS.md`. This file (`STANDARDS.md`) holds conventions.

When a comment is the right call, it explains *why*, not *what*.

---

## 5. Optimization conventions

### Vectorize

Prefer numpy vectorized operations over Python loops whenever the inner work is uniform. Loops are acceptable for outer iteration over a small heterogeneous list (e.g. iterating `OperatorTerm`s).

### Cache structural artifacts, not parameter-dependent ones

The general rule: cache anything whose construction depends only on geometry / topology / structure. Do **not** cache things whose value depends on user parameters that may change between calls.

Examples:

- `HypercubicHopGraph._cache` — hop edges per channel: cached forever.
- `InternalStructure._product_cache` — Γ-products: cached forever.
- `Model._assemble_cache_*` — last-assembled Hamiltonian: cached but invalidated on vacancy changes; keyed by `(apply_vacancies, params, vacancy bytes)`.
- Fourier coefficients per shift (depend on params): not cached, re-evaluated per `assemble`.
- Real-space-function values (depend on params): not cached.

### Lazy evaluation

Build expensive objects on first use, not at construction. The hop graph builds shift groups lazily; coordinates are computed once per embedding and then reused.

### Defer rather than bake in

When a transformation is cheap to apply at the edge of the pipeline, prefer storing the *flag* and applying the transformation lazily.

The canonical example is vacancies: `Model._vacancy_mask` is just a boolean array. The reduced Hamiltonian, the active-coordinates view, and the LDOS NaN-fill are all computed at the moment of access. This means a single model can be reused across many vacancy configurations without rebuilding.

The same applies to `apply_vacancies=False` toggles in `assemble` and `solve` — the user can opt out per call.

---

## 6. Mutability

Kernel objects are effectively immutable values. The single intentional exception is `Model._vacancy_mask`, which is meant to be toggled.

To change configuration, build a new model. There is no `update_*` API by design — that is the kind of patchwork that the rewrite was meant to eliminate.

If a future feature genuinely requires mutation (e.g. dynamic addition of operator terms), it should be implemented as a builder helper that returns a new `Model`, not as in-place mutation.

---

## 7. Naming

- Public types: `CamelCase`.
- Public functions and methods: `snake_case`.
- Private functions and module attributes: leading underscore.
- Acronyms: `LDOS`, `IPR` (kept in upper case for physics literacy, and for direct mapping onto literature).

Channel keys are signed integer tuples by convention; arbitrary hashable keys are accepted at the kernel.

Label names are arbitrary lowercase strings; the high-level builder uses `"sector"`.

---

## 8. Error handling

Raise the most specific built-in exception:

- `ValueError` for malformed but well-typed inputs (wrong shape, unknown key).
- `TypeError` for genuinely wrong types.
- `KeyError` only when looking up something the user clearly named (e.g. an unregistered real-space function).

Internal code should not raise on conditions that are the responsibility of the boundary to prevent.

---

## 9. Testing philosophy

(Tests will live under a top-level `tests/` directory once added.)

- Unit-test each kernel module against fully synthetic inputs.
- Integration-test by exercising builders end-to-end and comparing against analytical expectations on small lattices.
- Avoid relying on external reference data; if regression checks are useful, generate them in-tree from a known-good builder.

---

## 10. Documented exceptions to the standards

- `Model.solve` and `Model.solve_schur` use lazy in-function imports to avoid an `core ↔ solve` cycle. Acceptable because they are user-facing convenience shortcuts.
- `OperatorTerm.assemble_term` raises a few `ValueError`s on bad return shapes from user-supplied callbacks. Strictly speaking these are inputs from the *user*, but they pass through internal code, so we make a defensive check at the boundary between user code and the kernel pipeline.
- `Model._invalidate_cache` is a private method on a public class but is sometimes called by builders that swap `model.hop_graph` out for an overlay (see USAGE §10). This is a deliberate escape hatch; document any such mutation in the calling builder.
- The momentum DSL accepts `exp(...)` and interprets it as `exp(i·...)` for momentum arguments. This is not standard Python semantics but is the convention in the literature; documented in `USAGE.md`.
- Single-element 1×1 internal structures (`trivial_internal()`) skip the `sps.kron` call and use scalar multiplication. This is a micro-optimization that breaks the otherwise-uniform Kronecker pipeline; flagged in `assemble_term`.

