"""Small command-line entry point for reproducible offline spectrum caches.

The CLI purposefully accepts a simple NPZ exchange format instead of binding the
core package to a particular dataset framework.  A PyG preprocessing job can
write ``edge_index`` (and optional ``edge_weight``/``num_nodes``) to NPZ, invoke
this command in worker processes, then attach the resulting cache to its Data
objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .spectral.precompute import precompute_spectrum, save_spectrum


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ast-boost-precompute",
        description="Precompute an AST-Boost spectrum from a graph NPZ file.",
    )
    parser.add_argument(
        "input", type=Path, help="NPZ with edge_index and optional edge_weight/num_nodes"
    )
    parser.add_argument("output", type=Path, help="Destination .npz spectrum cache")
    parser.add_argument("--num-nodes", type=int, help="Override num_nodes stored in the input")
    parser.add_argument(
        "--k", type=int, default=8, help="Number of smallest positive modes (default: 8)"
    )
    parser.add_argument("--laplacian", choices=("sym", "comb", "rw"), default="sym")
    parser.add_argument(
        "--keep-zero", action="store_true", help="Keep zero modes instead of skipping them"
    )
    parser.add_argument("--zero-tolerance", type=float, default=1e-8)
    parser.add_argument("--degeneracy-eps", type=float, default=1e-2)
    parser.add_argument("--degeneracy-tau", type=float, default=1e-6)
    parser.add_argument("--dense-threshold", type=int, default=256)
    parser.add_argument(
        "--sparse-sigma",
        type=float,
        default=-1e-5,
        help="Negative shift for the lowest PSD modes (default: -1e-5)",
    )
    return parser


def _num_nodes(cache: np.lib.npyio.NpzFile, edge_index: np.ndarray, override: int | None) -> int:
    if override is not None:
        return override
    if "num_nodes" in cache:
        return int(np.asarray(cache["num_nodes"]).item())
    return int(edge_index.max()) + 1 if edge_index.size else 0


def main(argv: list[str] | None = None) -> int:
    """Run the NPZ-to-NPZ precompute command and return a process exit code."""
    args = _parser().parse_args(argv)
    with np.load(args.input, allow_pickle=False) as graph:
        if "edge_index" not in graph:
            raise ValueError(f"{args.input} must contain an edge_index array")
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"] if "edge_weight" in graph else None
        n = _num_nodes(graph, edge_index, args.num_nodes)
    spectrum = precompute_spectrum(
        edge_index,
        n=n,
        k=args.k,
        edge_weight=edge_weight,
        laplacian=args.laplacian,
        skip_zero=not args.keep_zero,
        zero_tolerance=args.zero_tolerance,
        degeneracy_eps=args.degeneracy_eps,
        degeneracy_tau=args.degeneracy_tau,
        dense_threshold=args.dense_threshold,
        sparse_sigma=args.sparse_sigma,
    )
    destination = save_spectrum(spectrum, args.output)
    print(
        f"wrote {destination}: N={spectrum.num_nodes}, k={spectrum.k}, "
        f"blocks={spectrum.num_blocks}, laplacian={spectrum.laplacian}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
