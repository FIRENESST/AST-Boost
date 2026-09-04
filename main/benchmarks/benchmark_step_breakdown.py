"""Break a warmed Full training step into GPU-timed phases."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batch_size, args.iterations, args.warmup) < 1:
        raise ValueError("batch size, iterations and warmup must be positive")
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
    rng = np.random.default_rng(20260904)
    selections = [
        rng.choice(len(bank), size=args.batch_size, replace=False)
        for _ in range(args.warmup + args.iterations)
    ]
    seed_all(42)
    model = GPSRegressor("full", field_scaling="size").cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5, fused=True)
    names = ("collate", "forward", "backward", "gradient_clip", "optimizer")
    event_rows = []
    for ids in selections:
        events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in names]
        events[0][0].record()
        batch = bank.batch(ids)
        events[0][1].record()
        optimizer.zero_grad(set_to_none=True)
        events[1][0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(batch).float() - batch.targets).abs().mean()
        events[1][1].record()
        events[2][0].record()
        loss.backward()
        events[2][1].record()
        events[3][0].record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        events[3][1].record()
        events[4][0].record()
        optimizer.step()
        events[4][1].record()
        event_rows.append(events)
    torch.cuda.synchronize()
    samples = {
        name: [row[index][0].elapsed_time(row[index][1]) for row in event_rows[args.warmup :]]
        for index, name in enumerate(names)
    }
    medians = {name: statistics.median(values) for name, values in samples.items()}
    total = sum(medians.values())
    report = {
        "configuration": {
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "precision": "bf16",
            "threads": 4,
        },
        "median_ms": medians,
        "median_phase_percent": {name: value / total * 100 for name, value in medians.items()},
        "samples_ms": samples,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "selection_sha256": hashlib.sha256(np.stack(selections).tobytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "CUDA-event phase time after complete warmup; excludes validation, IO and startup."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"median_ms": medians, "percent": report["median_phase_percent"]}))


if __name__ == "__main__":
    main()
