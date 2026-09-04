"""Measure cumulative exact field and input-aggregation fusions."""

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


def model_factory(*, field_fusion, input_fusion):
    model = GPSRegressor("full", field_scaling="size")
    model.pe.fuse_shared_fields = field_fusion
    model.pe.first_order_encoder.psi.fuse_input_aggregation = input_fusion
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.repeats, args.iterations, *args.batch_sizes) < 1:
        raise ValueError("repeats, iterations and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt",
            weights_only=True,
        ),
        "cuda",
    )
    seed_all(42)
    initial = model_factory(field_fusion=True, input_fusion=True)
    state = {key: value.clone() for key, value in initial.state_dict().items()}
    parameters = initial.trainable_parameters
    del initial
    factories = {
        "split_reference": lambda: model_factory(field_fusion=False, input_fusion=False),
        "field_fusion": lambda: model_factory(field_fusion=True, input_fusion=False),
        "field_and_linear_fusion": lambda: model_factory(field_fusion=True, input_fusion=True),
    }
    results, selections_sha256 = {}, {}
    for size in args.batch_sizes:
        rng = np.random.default_rng(20260904 + size)
        selections = [
            rng.choice(len(bank), size=size, replace=False) for _ in range(args.iterations)
        ]
        selections_sha256[str(size)] = hashlib.sha256(np.stack(selections).tobytes()).hexdigest()
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
                        bank,
                        selections,
                        static_layout=True,
                        mode=mode,
                        precision="bf16",
                        warmup=5,
                    )
                    samples[name].append(row)
                    print(f"B={size} {mode} repeat={repeat + 1} {name}: {row}", flush=True)
            medians = {
                name: statistics.median(row["milliseconds_per_batch"] for row in rows)
                for name, rows in samples.items()
            }
            results[f"B{size}-{mode}"] = {
                "samples": samples,
                "median_ms": medians,
                "cumulative_reduction_percent": 100
                * (1 - medians["field_and_linear_fusion"] / medians["split_reference"]),
                "linear_incremental_reduction_percent": 100
                * (1 - medians["field_and_linear_fusion"] / medians["field_fusion"]),
            }
    report = {
        "configuration": {
            "repeats": args.repeats,
            "iterations": args.iterations,
            "batch_sizes": args.batch_sizes,
            "warmup": 5,
            "precision": "bf16",
            "field_scaling": "size",
        },
        "parameters": parameters,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "selection_sha256": selections_sha256,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "Whole Full GPS-style model; same parameters/state and graph selections. Training "
            "includes collation, forward, backward, clipping and fused AdamW. No IO/validation."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                key: {
                    "cumulative": value["cumulative_reduction_percent"],
                    "linear_incremental": value["linear_incremental_reduction_percent"],
                }
                for key, value in results.items()
            }
        )
    )


if __name__ == "__main__":
    main()
