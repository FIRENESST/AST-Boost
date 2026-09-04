"""Compare identical AST functions with/without rebuilding an existing GPU layout."""

from __future__ import annotations

import argparse
import statistics

import torch
from benchmark_gpu import _measure, _path_edges

from ast_boost import ASTBoostPE, precompute_spectrum, prepare_spectrum_batch
from ast_boost.experiments.data import load_zinc_banks
from ast_boost.experiments.train import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["path", "zinc"], default="path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--variant", choices=["lite", "kern", "full"], default="full")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batch_size, args.nodes, args.repeats, args.iterations) < 1:
        raise ValueError("dimensions, repeats, and iterations must be positive")
    torch.manual_seed(42)
    if args.dataset == "zinc":
        banks = load_zinc_banks(
            PROJECT / "data" / "ZINC", PROJECT / ".cache" / "zinc" / "experiment_v1", device="cuda"
        )
        batch = banks["val"].batch(range(args.batch_size))
        spectra = batch.spectra
        padded_edges = batch.edge_index
    else:
        edges = _path_edges(args.nodes)
        spectrum = precompute_spectrum(edges, n=args.nodes, k=min(8, args.nodes - 1))
        spectra = prepare_spectrum_batch([spectrum] * args.batch_size, device="cuda")
        padded_edges = torch.cat(
            [torch.tensor(edges, device="cuda") + i * args.nodes for i in range(args.batch_size)],
            dim=1,
        )
    valid = spectra.valid_nodes
    tokens = torch.randn(*valid.shape, 64, device="cuda")
    flat = tokens[valid]
    membership = torch.repeat_interleave(
        torch.arange(args.batch_size, device="cuda"),
        torch.tensor(spectra.node_counts, device="cuda"),
        output_size=flat.shape[0],
    )
    mapping = torch.full((valid.numel(),), -1, device="cuda", dtype=torch.long)
    mapping[valid.reshape(-1)] = torch.arange(flat.shape[0], device="cuda")
    edges = mapping[padded_edges]
    model = (
        ASTBoostPE(variant=args.variant, heads=4, pe_dim=32, sign_hidden=64, token_dim=64)
        .cuda()
        .eval()
    )
    paths = {
        "rebuild padded layout": lambda: model.forward_padded_batch(
            flat, edges, spectra, membership, contiguous=True
        ),
        "reuse packed layout": lambda: model.forward_packed(tokens, padded_edges, spectra),
    }
    with torch.inference_mode():
        old = paths["rebuild padded layout"]()
        new = paths["reuse packed layout"]()
        torch.testing.assert_close(old[0], new[0][valid])
        torch.testing.assert_close(old[1], new[1])
    results = {name: [] for name in paths}
    for repeat in range(args.repeats):
        for name in list(paths) if repeat % 2 == 0 else list(paths)[::-1]:
            results[name].append(
                _measure(paths[name], warmup=5, iterations=args.iterations, backward=False)
            )
    print(
        f"dataset={args.dataset} variant={args.variant} B={args.batch_size} "
        f"maxN={spectra.max_nodes}; values verified"
    )
    for name, samples in results.items():
        times = [s[0] for s in samples]
        print(
            f"{name}: median={statistics.median(times):.3f}ms "
            f"range={min(times):.3f}..{max(times):.3f} "
            f"peak={max(s[1] for s in samples):.1f}MiB"
        )


if __name__ == "__main__":
    main()
