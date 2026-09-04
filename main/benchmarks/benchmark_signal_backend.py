"""Compare scatter and dense-GEMM Full without dropping any signal fields."""

import argparse
import hashlib
import json
import statistics
from copy import copy
from pathlib import Path

import numpy as np
import torch
from benchmark_backbone import measure

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    args = parser.parse_args()
    if min(args.repeats, args.iterations, *args.batch_sizes) < 1:
        raise ValueError("repeats, iterations and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the signal-backend benchmark")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt", weights_only=True
        ),
        "cuda",
        dense_signals=True,
    )
    # Reuse exactly the same data without making sparse collation select an
    # unused dense adjacency. The dense cache remains resident in both timings.
    sparse_bank = copy(bank)
    sparse_bank.tensors = {
        key: value for key, value in bank.tensors.items() if key != "dense_adjacency"
    }
    seed_all(42)
    model = GPSRegressor("full")
    parameters = model.trainable_parameters
    state = {key: value.clone() for key, value in model.state_dict().items()}
    del model
    results, selection_hashes = {}, {}
    factories = {
        "sparse_raw": lambda: GPSRegressor("full"),
        "sparse_size": lambda: GPSRegressor("full", field_scaling="size"),
        "dense_raw": lambda: GPSRegressor("full", signal_backend="dense"),
        "dense_size": lambda: GPSRegressor("full", signal_backend="dense", field_scaling="size"),
    }
    for size in args.batch_sizes:
        rng = np.random.default_rng(20260903)
        selections = [
            rng.choice(len(bank), size=size, replace=False) for _ in range(args.iterations)
        ]
        selection_hashes[str(size)] = hashlib.sha256(np.stack(selections).tobytes()).hexdigest()
        for mode in ("inference", "train"):
            samples = {name: [] for name in factories}
            for repeat in range(args.repeats):
                names = list(factories)
                offset = repeat % len(names)
                names = names[offset:] + names[:offset]
                for name in names:
                    row = measure(
                        factories[name],
                        state,
                        sparse_bank if name.startswith("sparse") else bank,
                        selections,
                        static_layout=True,
                        mode=mode,
                        precision="bf16",
                        warmup=5,
                    )
                    samples[name].append(row)
                    print(f"B={size} {mode} {name} {repeat + 1}: {row}", flush=True)
            results[f"B{size}-{mode}"] = {
                "samples": samples,
                "median_ms": {
                    name: statistics.median(row["milliseconds_per_batch"] for row in rows)
                    for name, rows in samples.items()
                },
            }
    report = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "iterations": args.iterations,
        "precision": "bf16",
        "threads": 4,
        "parameters": parameters,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "selection_sha256": selection_hashes,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_backbone.py").read_bytes()
        ).hexdigest(),
        "scope": (
            "Full whole-model steps; shared resident data; only dense paths collate adjacency. "
            "Dense cache is resident in all paths, so memory is not standalone sparse usage. "
            "Training includes backward, clip, AdamW. B64 changes update budget."
        ),
        "source_sha256": {
            str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
