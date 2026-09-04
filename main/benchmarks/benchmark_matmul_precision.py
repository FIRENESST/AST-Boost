"""Compare conservative FP32 matmuls with TF32-enabled high precision."""

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
    initial = GPSRegressor("full", field_scaling="size")
    state = {key: value.clone() for key, value in initial.state_dict().items()}
    parameters = initial.trainable_parameters
    del initial
    results = {}
    selection_hashes = {}
    for size in args.batch_sizes:
        rng = np.random.default_rng(20260906 + size)
        selections = [
            rng.choice(len(bank), size=size, replace=False) for _ in range(args.iterations)
        ]
        selection_hashes[str(size)] = hashlib.sha256(np.stack(selections).tobytes()).hexdigest()
        predictions = {}
        model = GPSRegressor("full", field_scaling="size").cuda().eval()
        model.load_state_dict(state)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for precision in ("highest", "high"):
                torch.set_float32_matmul_precision(precision)
                predictions[precision] = model(bank.batch(selections[0])).float().cpu()
        prediction_difference = {
            "max_abs": float((predictions["highest"] - predictions["high"]).abs().max()),
            "mean_abs": float((predictions["highest"] - predictions["high"]).abs().mean()),
        }
        del model, predictions
        for mode in ("inference", "train"):
            samples = {"highest": [], "high": []}
            for repeat in range(args.repeats):
                names = list(samples)
                if repeat % 2:
                    names.reverse()
                for precision in names:
                    torch.set_float32_matmul_precision(precision)
                    sample = measure(
                        lambda: GPSRegressor("full", field_scaling="size"),
                        state,
                        bank,
                        selections,
                        static_layout=True,
                        mode=mode,
                        precision="bf16",
                        warmup=args.warmup,
                    )
                    samples[precision].append(sample)
                    print(f"B{size} {mode} repeat={repeat + 1} {precision}: {sample}", flush=True)
            medians = {
                name: statistics.median(row["milliseconds_per_batch"] for row in rows)
                for name, rows in samples.items()
            }
            results[f"B{size}-{mode}"] = {
                "samples": samples,
                "median_ms": medians,
                "high_reduction_percent": 100 * (1 - medians["high"] / medians["highest"]),
                "bf16_prediction_difference": prediction_difference,
            }
    torch.set_float32_matmul_precision("highest")
    report = {
        "configuration": {
            "repeats": args.repeats,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "batch_sizes": args.batch_sizes,
            "neural_precision": "bf16",
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
            "Whole Full model with identical state and selections. Only CUDA FP32 matmul "
            "precision changes; neural layers remain under BF16 autocast."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
