"""Benchmark exact minibatch-local removal of globally padded cache entries."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from benchmark_backbone import measure

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, seed_all


class _BatchPolicy:
    def __init__(self, bank, *, trim_padding):
        self.bank = bank
        self.trim_padding = trim_padding

    def batch(self, indices, *, static_layout=True):
        return self.bank.batch(
            indices,
            static_layout=static_layout,
            trim_padding=self.trim_padding,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    args = parser.parse_args()
    if min(args.repeats, args.iterations, args.warmup, *args.batch_sizes) < 1:
        raise ValueError("benchmark dimensions must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt",
            map_location="cpu",
            weights_only=True,
        ),
        "cuda",
    )
    policies = {
        "split_global_padding": _BatchPolicy(bank, trim_padding=False),
        "minibatch_trimmed": _BatchPolicy(bank, trim_padding=True),
    }
    seed_all(42)
    initial = GPSRegressor("full", field_scaling="size")
    state = {key: value.clone() for key, value in initial.state_dict().items()}
    parameters = initial.trainable_parameters
    del initial
    results = {}
    selection_hashes = {}
    for size in args.batch_sizes:
        rng = np.random.default_rng(20260905 + size)
        selections = [
            rng.choice(len(bank), size=size, replace=False) for _ in range(args.iterations)
        ]
        selection_hashes[str(size)] = hashlib.sha256(np.stack(selections).tobytes()).hexdigest()
        first = selections[0]
        with torch.inference_mode():
            reference_model = GPSRegressor("full", field_scaling="size").cuda().eval()
            reference_model.load_state_dict(state)
            expected = reference_model(policies["split_global_padding"].batch(first)).float()
            actual = reference_model(policies["minibatch_trimmed"].batch(first)).float()
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            fp32_max_difference = float((actual - expected).abs().max())
            del reference_model, expected, actual
        local_maxima = [max(bank.counts[int(i)] for i in ids) for ids in selections]
        for mode in ("inference", "train"):
            samples = {name: [] for name in policies}
            for repeat in range(args.repeats):
                names = list(policies)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    sample = measure(
                        lambda: GPSRegressor("full", field_scaling="size"),
                        state,
                        policies[name],
                        selections,
                        static_layout=True,
                        mode=mode,
                        precision="bf16",
                        warmup=args.warmup,
                    )
                    samples[name].append(sample)
                    print(f"B{size} {mode} repeat={repeat + 1} {name}: {sample}", flush=True)
            medians = {
                name: statistics.median(row["milliseconds_per_batch"] for row in rows)
                for name, rows in samples.items()
            }
            results[f"B{size}-{mode}"] = {
                "samples": samples,
                "median_ms": medians,
                "latency_reduction_percent": 100
                * (1 - medians["minibatch_trimmed"] / medians["split_global_padding"]),
                "fp32_max_abs_difference": fp32_max_difference,
                "batch_max_nodes": local_maxima,
            }
    report = {
        "configuration": {
            "repeats": args.repeats,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "batch_sizes": args.batch_sizes,
            "precision": "bf16",
            "field_scaling": "size",
        },
        "parameters": parameters,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "selection_sha256": selection_hashes,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "Whole Full model with identical state and selections. Only zero-padded cache "
            "extents differ. Training includes collation, forward, backward, clipping and AdamW."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
