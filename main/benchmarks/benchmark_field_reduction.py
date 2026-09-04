"""Compare masked field materialization with an equivalent batched contraction."""

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


def model_factory(*, bmm_reduction):
    model = GPSRegressor("full", field_scaling="size")
    model.pe.first_order_encoder.bmm_field_reduction = bmm_reduction
    model.pe.second_order_encoder.bmm_field_reduction = bmm_reduction
    return model


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
    seed_all(42)
    initial = model_factory(bmm_reduction=False)
    state = {key: value.clone() for key, value in initial.state_dict().items()}
    parameters = initial.trainable_parameters
    del initial
    factories = {
        "masked_sum": lambda: model_factory(bmm_reduction=False),
        "bmm_reduction": lambda: model_factory(bmm_reduction=True),
    }
    results = {}
    selection_hashes = {}
    for size in args.batch_sizes:
        rng = np.random.default_rng(20260907 + size)
        selections = [
            rng.choice(len(bank), size=size, replace=False) for _ in range(args.iterations)
        ]
        selection_hashes[str(size)] = hashlib.sha256(np.stack(selections).tobytes()).hexdigest()
        model = factories["masked_sum"]().cuda().eval()
        model.load_state_dict(state)
        batch = bank.batch(selections[0])
        with torch.inference_mode():
            expected = model(batch)
            model.pe.first_order_encoder.bmm_field_reduction = True
            model.pe.second_order_encoder.bmm_field_reduction = True
            actual = model(batch)
        fp32_difference = {
            "max_abs": float((actual - expected).abs().max()),
            "mean_abs": float((actual - expected).abs().mean()),
        }
        del model, batch, expected, actual
        for mode in ("inference", "train"):
            samples = {name: [] for name in factories}
            for repeat in range(args.repeats):
                names = list(factories)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    sample = measure(
                        factories[name],
                        state,
                        bank,
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
                * (1 - medians["bmm_reduction"] / medians["masked_sum"]),
                "fp32_prediction_difference": fp32_difference,
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
            "Whole Full model with identical state and selections. Only field-mask reduction "
            "implementation differs; training includes backward, clipping and AdamW."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
