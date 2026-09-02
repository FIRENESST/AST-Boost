"""Compare legacy disjoint bias with the GPU-friendly padded batch path."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import torch

from ast_boost import (
    ASTBoostPE,
    precompute_spectrum,
    prepare_spectrum,
    prepare_spectrum_batch,
)


def _path_edges(n: int) -> np.ndarray:
    source = np.arange(n - 1, dtype=np.int64)
    target = source + 1
    return np.stack((np.concatenate((source, target)), np.concatenate((target, source))))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AST-Boost CUDA batch benchmark")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--variant", choices=("lite", "kern", "full"), default="full")
    return parser.parse_args()


def _measure(
    name: str,
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> None:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end) / iterations
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"{name:18s} {milliseconds:8.3f} ms/step  peak={peak_mib:8.1f} MiB")


def main() -> int:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for this benchmark")
    if min(args.batch_size, args.nodes, args.hidden, args.heads, args.iterations) <= 0:
        raise ValueError("batch, nodes, hidden, heads, and iterations must be positive")

    device = torch.device("cuda")
    base_edges = _path_edges(args.nodes)
    edge_index = np.concatenate(
        [base_edges + graph * args.nodes for graph in range(args.batch_size)], axis=1
    )
    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    batch = torch.arange(args.batch_size, device=device).repeat_interleave(args.nodes)
    tokens = torch.randn(args.batch_size * args.nodes, args.hidden, device=device)
    spectrum = precompute_spectrum(base_edges, n=args.nodes, k=min(8, args.nodes - 1))
    prepared = prepare_spectrum(spectrum, k0_pairs=4, device=device)
    spectra = [prepared] * args.batch_size
    prepare_start = time.perf_counter()
    spectrum_batch = prepare_spectrum_batch(spectra, k0_pairs=4, device=device)
    torch.cuda.synchronize()
    prepare_milliseconds = (time.perf_counter() - prepare_start) * 1000.0
    model = (
        ASTBoostPE(
            variant=args.variant,
            heads=args.heads,
            pe_dim=32,
            sign_hidden=64,
            k0_pairs=4,
            token_dim=args.hidden,
        )
        .to(device)
        .eval()
    )

    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    print(f"spectrum batch build {prepare_milliseconds:.3f} ms (excluded from step timings)")
    _measure(
        "disjoint bias",
        lambda: model.forward_batch(tokens, edge_index_tensor, spectra, batch),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    _measure(
        "vectorized padded",
        lambda: model.forward_padded_batch(
            tokens, edge_index_tensor, spectrum_batch, batch, contiguous=True
        ),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
