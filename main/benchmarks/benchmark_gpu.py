"""Compare legacy disjoint bias with the GPU-friendly padded batch path."""

from __future__ import annotations

import argparse
import statistics
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backward", action="store_true", help="Measure forward + backward (no optimizer step)"
    )
    parser.add_argument(
        "--include-preparation",
        action="store_true",
        help="Include rebuilding padded spectra from a GPU per-graph cache",
    )
    parser.add_argument("--variant", choices=("lite", "kern", "full"), default="full")
    return parser.parse_args()


def _measure(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    backward: bool,
) -> tuple[float, float]:
    context = torch.enable_grad() if backward else torch.inference_mode()
    with context, torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
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
    return milliseconds, peak_mib


def main() -> int:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for this benchmark")
    if (
        min(args.batch_size, args.nodes, args.hidden, args.heads, args.iterations, args.repeats)
        <= 0
    ):
        raise ValueError("batch, nodes, hidden, heads, iterations, and repeats must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    torch.manual_seed(args.seed)

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
    torch.cuda.synchronize()
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
    print(
        f"mode={'forward+backward' if args.backward else 'inference'} seed={args.seed} "
        f"B={args.batch_size} N={args.nodes} repeats={args.repeats}"
    )
    print(f"spectrum batch build {prepare_milliseconds:.3f} ms (excluded from step timings)")
    if args.include_preparation:
        print("padded timing includes spectrum-batch rebuild; resident per-graph caches exclude IO")

    def padded_forward():
        cached = (
            prepare_spectrum_batch(spectra, k0_pairs=4, device=device)
            if args.include_preparation
            else spectrum_batch
        )
        return model.forward_padded_batch(tokens, edge_index_tensor, cached, batch, contiguous=True)

    paths = {
        "disjoint bias": lambda: model.forward_batch(tokens, edge_index_tensor, spectra, batch),
        "vectorized padded": padded_forward,
    }
    samples = {name: [] for name in paths}

    def step(forward):
        if not args.backward:
            return forward()
        model.zero_grad(set_to_none=True)
        encoded, bias, *_ = forward()
        # Equal loss scaling despite the reference's extra zero cross-graph entries.
        bias_count = args.batch_size * args.heads * args.nodes**2
        loss = encoded.float().square().mean() + bias.float().square().sum() / bias_count
        loss.backward()
        return None

    names = list(paths)
    for repeat in range(args.repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            samples[name].append(
                _measure(
                    lambda: step(paths[name]),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    backward=args.backward,
                )
            )
    for name, results in samples.items():
        timings = [result[0] for result in results]
        peak = max(result[1] for result in results)
        print(
            f"{name:18s} median={statistics.median(timings):8.3f} ms/step "
            f"range=[{min(timings):.3f},{max(timings):.3f}] peak={peak:.1f} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
