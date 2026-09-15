#!/usr/bin/env python3
"""Combine Andras's g=0/g=1 Witten outputs into FractalsProject results.

Place this script in the FractalsProject root (next to ``project_tools/``) and run
it with the path to the ``results`` directory returned by Andras::

    python postprocess_witten_results.py /path/to/results

For each production case, the script pairs the background (g=0) and monopole
(g=1) runs, computes the induced charge field and cumulative delta_Q(R), and
saves one canonical ``data/witten/...`` result file using the project's existing
``project_tools.io`` conventions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from project_tools import io, witten
except ImportError as exc:
    raise SystemExit(
        "Could not import project_tools. Put this script in the FractalsProject "
        "root and run it from there."
    ) from exc


CASE_SPECS = {
    "sub_m010": {"fractal": "sponge", "method": "substituted", "M_alt": -0.10},
    "sub_m005": {"fractal": "sponge", "method": "substituted", "M_alt": -0.05},
    "cube": {"fractal": "cube", "method": "cube"},
    "site_elim": {"fractal": "sponge", "method": "site_elim"},
    "renorm": {"fractal": "sponge", "method": "renorm"},
}

EXPECTED_COMMON = {
    "M": 2.0,
    "M_prime": 0.01,
    "t": 1.0,
    "B": 1.0,
    "gauge": "N",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process the Witten-effect results returned by Andras."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing Andras's Witten_*.npz output files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=300,
        help="Number of radii used for delta_Q(R) (default: 300).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite canonical output files if they already exist.",
    )
    return parser.parse_args()


def read_raw_result(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        required = {"charge", "eigenvalues_occupied", "metadata"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
        arrays = {
            "charge": np.asarray(data["charge"]),
            "eigenvalues_occupied": np.asarray(data["eigenvalues_occupied"]),
        }
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata


def same_value(a, b) -> bool:
    if isinstance(b, float):
        try:
            return bool(np.isclose(float(a), b, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return a == b


def validate_metadata(path: Path, meta: dict) -> None:
    case = meta.get("case")
    if case not in CASE_SPECS:
        raise ValueError(f"{path.name}: unknown case {case!r}")
    if meta.get("g") not in (0, 1):
        raise ValueError(f"{path.name}: expected g=0 or g=1; got {meta.get('g')!r}")

    expected = dict(EXPECTED_COMMON)
    expected["method"] = CASE_SPECS[case]["method"]
    if "M_alt" in CASE_SPECS[case]:
        expected["M_alt"] = CASE_SPECS[case]["M_alt"]

    for key, wanted in expected.items():
        if not same_value(meta.get(key), wanted):
            raise ValueError(
                f"{path.name}: metadata mismatch for {key}: "
                f"expected {wanted!r}, got {meta.get(key)!r}"
            )


def collect_pairs(results_dir: Path):
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    grouped: dict[str, dict[int, tuple[Path, dict[str, np.ndarray], dict]]] = {}
    for path in sorted(results_dir.glob("*.npz")):
        arrays, meta = read_raw_result(path)
        validate_metadata(path, meta)
        case = str(meta["case"])
        g = int(meta["g"])
        if g in grouped.setdefault(case, {}):
            other = grouped[case][g][0]
            raise ValueError(
                f"Duplicate result for case={case}, g={g}: {other.name} and {path.name}"
            )
        grouped[case][g] = (path, arrays, meta)

    missing = []
    for case in CASE_SPECS:
        for g in (0, 1):
            if g not in grouped.get(case, {}):
                missing.append(f"{case} g={g}")
    if missing:
        raise ValueError("Missing production results: " + ", ".join(missing))

    extra_cases = set(grouped).difference(CASE_SPECS)
    if extra_cases:
        raise ValueError(f"Unexpected cases: {sorted(extra_cases)}")

    return grouped


def canonical_meta(case: str, source_meta: dict) -> dict:
    spec = CASE_SPECS[case]
    L = int(source_meta["L"])

    if case == "cube":
        # Match the existing FractalsProject convention for solid-cube reference data.
        return {
            "fractal": "cube",
            "method": "cube",
            "L": L,
            **EXPECTED_COMMON,
        }

    meta = {
        "fractal": "sponge",
        "method": spec["method"],
        "n": int(source_meta.get("n", 1)),
        "pasted": bool(source_meta.get("pasted", True)),
        **EXPECTED_COMMON,
        "L": L,
    }
    if "M_alt" in spec:
        meta["M_alt"] = spec["M_alt"]
    return meta


def process_case(
    case: str,
    pair: dict[int, tuple[Path, dict[str, np.ndarray], dict]],
    *,
    resolution: int,
    force: bool,
) -> Path:
    path_bg, bg, meta_bg = pair[0]
    path_mono, mono, meta_mono = pair[1]

    for key in ("L", "method", "case"):
        if meta_bg.get(key) != meta_mono.get(key):
            raise ValueError(
                f"{case}: g=0 and g=1 metadata disagree for {key}: "
                f"{meta_bg.get(key)!r} vs {meta_mono.get(key)!r}"
            )

    charge_bg = bg["charge"]
    charge_mono = mono["charge"]
    if charge_bg.shape != charge_mono.shape:
        raise ValueError(
            f"{case}: charge-grid shapes differ: {charge_bg.shape} vs {charge_mono.shape}"
        )

    dQ_field = charge_mono - charge_bg
    R, dQ = witten.dQ_of_R(dQ_field, resolution=resolution)

    meta = canonical_meta(case, meta_bg)
    output = io.result_path("witten", meta)
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Use --force only if you intend to replace the existing file."
        )

    saved = io.save_result(
        "witten",
        meta,
        R=R,
        dQ=dQ,
        charge_bg=charge_bg,
        charge_mono=charge_mono,
        dQ_field=dQ_field,
        eigenvalues_bg=bg["eigenvalues_occupied"],
        eigenvalues_mono=mono["eigenvalues_occupied"],
    )

    print(f"{case}: {path_bg.name} + {path_mono.name}")
    print(f"  -> {saved}")
    return saved


def main() -> None:
    args = parse_args()
    if args.resolution < 2:
        raise ValueError("--resolution must be at least 2")

    grouped = collect_pairs(args.results_dir.resolve())

    outputs = []
    for case in CASE_SPECS:
        outputs.append(
            process_case(
                case,
                grouped[case],
                resolution=args.resolution,
                force=args.force,
            )
        )

    print(f"\nProcessed {len(outputs)} Witten cases successfully.")


if __name__ == "__main__":
    main()
