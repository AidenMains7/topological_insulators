#!/usr/bin/env python3
"""Run one Witten-sponge diagonalization pair and save a plot_data.py-ready result.

Solves the half-filled charge distribution once at g=0 (background) and once at
g=1 (monopole), forms the induced-charge field, integrates it to delta_Q(R) via
``project_tools.witten.dQ_of_R``, and writes the result with the same
``__meta__``/array-naming convention ``project_tools.io`` uses, so it loads
straight into ``plot_dQ_vs_R`` / ``plot_spectrum`` in ``plot_data.py`` — see
README.md, worked example C.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from project_tools import lattice, model, observables, witten

# The project's own reserved metadata key (project_tools/io.py: _META_KEY).
# Hardcoded here (rather than importing the private name) so any file this
# script writes is loadable by io.load_result / plot_data.py's _resolve().
_META_KEY = "__meta__"

CASES = {
    "sub_m010": {"method": "substituted", "M_alt": -0.10},
    "sub_m005": {"method": "substituted", "M_alt": -0.05},
    "cube": {"method": "cube"},
    "site_elim": {"method": "site_elim"},
    "renorm": {"method": "renorm"},
}


def max_rss_kib() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if sys.platform == "darwin" else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-scale", type=int, required=True)
    parser.add_argument("--case", choices=tuple(CASES), default="sub_m010")
    parser.add_argument("--resolution", type=int, default=300,
                         help="number of R samples for delta_Q(R)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def accumulate_site_charge(
    eigenvectors: np.ndarray,
    *,
    internal_dim: int,
    chunk_states: int = 128,
) -> np.ndarray:
    """Sum |psi|^2 over occupied eigenvectors with bounded temporary storage."""
    hilbert_dim, n_states = eigenvectors.shape
    if hilbert_dim % internal_dim:
        raise ValueError("Hilbert dimension is not divisible by the internal dimension")
    n_sites = hilbert_dim // internal_dim
    charge = np.zeros(n_sites, dtype=np.float64)
    for lo in range(0, n_states, chunk_states):
        hi = min(lo + chunk_states, n_states)
        block = eigenvectors[:, lo:hi].reshape(n_sites, internal_dim, hi - lo)
        charge += np.einsum("sdk,sdk->s", block.real, block.real, optimize=True)
        charge += np.einsum("sdk,sdk->s", block.imag, block.imag, optimize=True)
    return charge


def occupied_charge(built, params: dict[str, float | str], method: str):
    """Return occupied eigenvalues, half-filled charge grid, and dimensions."""
    internal_dim = int(built.internal.dim)
    solver_kwargs = {"overwrite_a": True, "check_finite": False}

    if method in ("substituted", "site_elim", "cube"):
        hilbert_dimension = int(built.active_hilbert_dim)
        occupied_count = hilbert_dimension // 2
        solver_kwargs["subset_by_index"] = [0, occupied_count - 1]
        solved = built.solve(
            hermitian=True,
            k=None,
            return_eigenvalues=True,
            return_eigenvectors=True,
            return_LDOS=False,
            params=params,
            solver_kwargs=solver_kwargs,
        )
        site_charge = accumulate_site_charge(
            solved["eigenvectors"], internal_dim=internal_dim
        )
        charge = observables.build_LDOS(site_charge[:, None], built)[0]
        active_sites = int(built.n_active_sites)

    elif method == "renorm":
        labels = np.asarray(built.sites.labels("sector"))
        active = ~np.asarray(built.vacancy_mask, dtype=bool)
        kept_full_indices = np.nonzero(active & (labels != 0))[0]
        active_sites = int(kept_full_indices.size)
        hilbert_dimension = active_sites * internal_dim
        occupied_count = hilbert_dimension // 2
        solver_kwargs["subset_by_index"] = [0, occupied_count - 1]
        solved = built.solve_schur(
            eliminate_label="sector",
            eliminate_value=0,
            energy=0.0,
            hermitian=True,
            k=None,
            return_eigenvalues=True,
            return_eigenvectors=True,
            return_LDOS=False,
            params=params,
            solver_kwargs=solver_kwargs,
        )
        site_charge = accumulate_site_charge(
            solved["eigenvectors"], internal_dim=internal_dim
        )
        charge = observables.build_LDOS_partial(site_charge[:, None], built, kept_full_indices)[0]

    else:  # Defensive: argparse already restricts this.
        raise ValueError(f"Unknown method: {method}")

    eigenvalues = np.asarray(solved["eigenvalues"], dtype=np.float64)
    del solved

    if eigenvalues.size != occupied_count:
        raise RuntimeError(
            f"Expected {occupied_count} occupied eigenvalues; received {eigenvalues.size}"
        )
    charge_sum = float(np.nansum(charge))
    tolerance = max(1e-7, 5e-10 * occupied_count)
    if not np.isclose(charge_sum, occupied_count, rtol=0.0, atol=tolerance):
        raise RuntimeError(
            f"Charge normalization failed: {charge_sum} != {occupied_count}"
        )
    if hasattr(built, "_invalidate_cache"):
        built._invalidate_cache()

    return eigenvalues, np.asarray(charge), hilbert_dimension, occupied_count, active_sites


def main() -> None:
    args = parse_args()
    if args.block_scale < 1:
        raise ValueError("--block-scale must be positive")
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Use --force to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    n = 1
    L = lattice.system_length(n, block_scale=args.block_scale, pasted=True)
    case = CASES[args.case]
    method = str(case["method"])
    M = 2.0
    M_alt = float(case.get("M_alt", M))
    base_params = {
        "M": M,
        "M_alt": M_alt,
        "M_prime": 0.01,
        "t": 1.0,
        "B": 1.0,
        "disorder_strength": 0.0,
        "disorder_seed": 0,
        "gauge": "N",
    }

    print(
        f"Starting Witten calculation: case={args.case}, method={method}, L={L}",
        flush=True,
    )
    print(f"Output: {output}", flush=True)
    started = datetime.now(timezone.utc)
    total_start = time.perf_counter()

    built = model.build_model(
        "sponge",
        n,
        hole_treatment=method,
        pbc=False,
        block_scale=args.block_scale,
        pasted=True,
        pseudo_scalar=True,
    )
    build_seconds = time.perf_counter() - total_start
    print(
        f"Model built in {build_seconds:.3f} s; "
        f"full sites={built.sites.n}; active sites={built.n_active_sites}",
        flush=True,
    )

    # Background (g=0) and monopole (g=1) solves — both are needed to form the
    # induced-charge field that delta_Q(R) integrates.
    solve_start = time.perf_counter()
    eigenvalues_bg, charge_bg, hilbert_dimension, occupied_count, active_sites = occupied_charge(
        built, dict(base_params, g=0.0), method
    )
    eigenvalues_mono, charge_mono, _, _, _ = occupied_charge(
        built, dict(base_params, g=1.0), method
    )
    solve_seconds = time.perf_counter() - solve_start

    dQ_field = charge_mono - charge_bg
    R, dQ = witten.dQ_of_R(dQ_field, resolution=args.resolution)
    total_seconds = time.perf_counter() - total_start

    # Identity of the run — matches the key set/order project_tools.io.case_tag
    # expects, and README.md's convention of dropping M_alt for single-mass
    # methods (site_elim / renorm / cube).
    meta = {
        "fractal": "sponge",
        "method": method,
        "n": n,
        "L": L,
        "pasted": True,
        "M": M,
        "M_alt": M_alt if method == "substituted" else None,
        "M_prime": 0.01,
        "t": 1.0,
        "B": 1.0,
        "gauge": "N",
    }

    run_info = {
        "case": args.case,
        "block_scale": args.block_scale,
        "hilbert_dimension": hilbert_dimension,
        "occupied_count": occupied_count,
        "active_site_count": active_sites,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": total_seconds,
        "max_rss_kib": max_rss_kib(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }

    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                R=R,
                dQ=dQ,
                charge_bg=charge_bg,
                charge_mono=charge_mono,
                dQ_field=dQ_field,
                eigenvalues_bg=eigenvalues_bg,
                eigenvalues_mono=eigenvalues_mono,
                **{_META_KEY: json.dumps(meta, sort_keys=True)},
                run_info=json.dumps(run_info, sort_keys=True),
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Finished in {total_seconds:.3f} s; solve time={solve_seconds:.3f} s", flush=True)
    print(f"Peak RSS={run_info['max_rss_kib'] / 1024**2:.3f} GiB", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()