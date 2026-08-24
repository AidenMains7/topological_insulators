"""Data-computation orchestration — run this file in your IDE.

Set the compute flags at the top of ``main()``, adjust the parameter blocks for
whichever sections are enabled, and run.  This file performs NO plotting; see
``plot_data.py`` for figures.  Results are saved under the root-level ``data/``
directory.

Load results back with:
    arrays, meta = io.load_result(io.result_path('witten', dict(...)))
    arrays, meta = io.load_result(io.result_path('phase_diagram', dict(...)))
"""

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from project_tools import model, observables, io, lattice


# ─────────────────────────────────────────────────────────────────────────────
# Phase-diagram sweep (cantor polarization / carpet Bott index)
#
# Grid convention: grid[i, j]  <->  M = M_vals[j],  M_alt = M_alt_vals[i]
# Sentinel for failed points: NaN (cantor) or 100 (carpet).
# ─────────────────────────────────────────────────────────────────────────────

def phase_diagram(fractal, n, M_range, M_alt_range, *,
                  M_res, M_alt_res, block_scale=2, upscale_to_n=None,
                  pasted=False, n_jobs=-1, t=1.0, B=1.0):
    """Parallel (M, M_alt) sweep; returns (grid, M_vals, M_alt_vals)."""
    if fractal == "cantor":
        index_fn = observables.polarization
        sentinel  = np.nan
        dtype     = float
    else:                           # carpet
        index_fn = observables.bott_index
        sentinel  = 100
        dtype     = np.int8

    M_vals     = np.linspace(M_range[0],     M_range[1],     M_res)
    M_alt_vals = np.linspace(M_alt_range[0], M_alt_range[1], M_alt_res)

    m = model.build_model(fractal, n, hole_treatment="substituted", pbc=True,
                          upscale_to_n=upscale_to_n, block_scale=block_scale,
                          pasted=pasted)

    def _worker(i, j):
        params = dict(M=float(M_vals[j]), M_alt=float(M_alt_vals[i]),
                      disorder_strength=1e-5, disorder_seed=0, t=t, B=B)
        try:
            return index_fn(m, params)
        except Exception as e:                          # noqa: BLE001
            print(f"  WARNING: failed M={M_vals[j]:.4f}, M_alt={M_alt_vals[i]:.4f}: {e}")
            return sentinel

    shape = (M_alt_res, M_res)
    L = lattice.system_length(n, upscale_to_n=upscale_to_n, block_scale=block_scale)
    with tqdm_joblib(tqdm(total=M_res * M_alt_res,
                          desc=f"{fractal} n={n} L={L}")):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_worker)(i, j) for i, j in np.ndindex(shape)
        )

    grid = np.array(results, dtype=dtype).reshape(shape)
    return grid, M_vals, M_alt_vals


def main():
    # ── compute flags ────────────────────────────────────────────────────────
    RUN_CANTOR = True
    RUN_CARPET = True

    # ── phase-diagram sweep helper ───────────────────────────────────────────
    # Size knobs per case: (n, upscale_to_n, block_scale)
    #   n            : fractal generation
    #   upscale_to_n : replicate cells to this generation before block expansion
    #   block_scale  : linear-size multiplier
    #   L = block_scale * 3**max(n, upscale_to_n); upscale_to_n=None means use n
    def run_phase_diagrams(fractal, cases, *, M_res, M_alt_res,
                           M_range, M_alt_range,
                           n_jobs=25):
        for n, upscale_to_n, block_scale in cases:
            grid, M_vals, M_alt_vals = phase_diagram(
                fractal, n, M_range, M_alt_range,
                M_res=M_res, M_alt_res=M_alt_res,
                block_scale=block_scale, upscale_to_n=upscale_to_n,
                n_jobs=n_jobs,
            )
            meta_pd = dict(
                fractal=fractal, method="substituted",
                n=n,
                L=lattice.system_length(n, upscale_to_n=upscale_to_n,
                                        block_scale=block_scale),
            )
            path_pd = io.save_result("phase_diagram", meta_pd,
                                     grid=grid, M_vals=M_vals, M_alt_vals=M_alt_vals)
            print("Saved:", path_pd)

    # ── SECTION 1 — Cantor chain polarization phase diagrams ──────────────────
    if RUN_CANTOR:
        # All 15 cantor cases: (n, upscale_to_n, block_scale), grouped by L
        cantor_cases = [
            # L=81  (block_scale=1)
            (2, 4, 1), (3, 4, 1), (4, None, 1),
            # L=162 (block_scale=2)
            (2, 4, 2), (3, 4, 2), (4, None, 2),
            # L=243 (block_scale=3)
            (2, 4, 3), (3, 4, 3), (4, None, 3),
            # L=324 (block_scale=4)
            (2, 4, 4), (3, 4, 4), (4, None, 4),
            # L=405 (block_scale=5)
            (2, 4, 5), (3, 4, 5), (4, None, 5),
        ]
        run_phase_diagrams("cantor", cantor_cases, M_res=121, M_alt_res=121,
                           M_range=(-1.0, 5.0), M_alt_range=(-1.0, 5.0))

    # ── SECTION 2 — Sierpinski carpet Bott-index phase diagrams ───────────────
    if RUN_CARPET:
        # All 10 carpet cases: (n, upscale_to_n, block_scale), grouped by L
        carpet_cases = [
            # L=27
            (1, 2, 3), (2, None, 3), (3, None, 1),
            # L=36
            (1, 2, 4), (2, None, 4),
            # L=45
            (1, 2, 5), (2, None, 5),
            # L=54
            (1, 2, 6), (2, None, 6), (3, None, 2),
        ]
        run_phase_diagrams("carpet", carpet_cases, M_res=101, M_alt_res=101,
                           M_range=(-1.0, 9.0), M_alt_range=(-1.0, 9.0))


if __name__ == "__main__":
    main()

