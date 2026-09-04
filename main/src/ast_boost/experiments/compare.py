"""Predeclared, seed-paired Full accuracy/throughput study with retained controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from . import train

ARMS = {
    "control": {"field_scaling": "none", "batch_size": 32, "lr": 0.001},
    "size32": {"field_scaling": "size", "batch_size": 32, "lr": 0.001},
    "size64": {"field_scaling": "size", "batch_size": 64, "lr": 0.002},
}


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    summary = {}
    controls = {row["seed"]: row for row in grouped.get("control", [])}
    for arm, values in grouped.items():
        scores = [row["best_val_mae"] for row in values]
        paired = [
            {
                "seed": row["seed"],
                "difference": row["best_val_mae"] - controls[row["seed"]]["best_val_mae"],
            }
            for row in values
            if row["seed"] in controls and arm != "control"
        ]
        summary[arm] = {
            "n": len(scores),
            "val_mean": statistics.mean(scores),
            "val_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "train_seconds_median": statistics.median(
                row["train_epoch_seconds_median"] for row in values
            ),
            "total_training_seconds_mean": statistics.mean(
                row["total_training_seconds"] for row in values
            ),
            "paired_minus_control": paired,
        }
    return {
        "arms": summary,
        "runs": rows,
        "scope": (
            "Same epochs/data/architecture; size64 changes batch size and LR together. "
            "Validation-only exploratory evidence; Full retained."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.arms)) != len(args.arms)
    ):
        raise ValueError("epochs must be positive and seeds/arms unique")
    args.output = args.output.resolve()
    source_hashes = {
        str(path.relative_to(train.PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((train.PROJECT / "src/ast_boost").rglob("*.py"))
    }
    plan = {
        "epochs": args.epochs,
        "seeds": args.seeds,
        "arms": {arm: ARMS[arm] for arm in args.arms},
        "source_sha256": source_hashes,
        "backend": "sparse",
        "method": "full",
        "selection": "best validation MAE",
        "test_evaluation": False,
        "full_policy": "retain all branches, arms and checkpoints",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_file = args.output / "plan.json"
    if plan_file.exists():
        if not args.resume or json.loads(plan_file.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing comparison: identical plan/source and --resume required")
    else:
        train.atomic_json(plan_file, plan)
    rows = []
    for i, seed in enumerate(args.seeds):
        shift = i % len(args.arms)
        for arm in args.arms[shift:] + args.arms[:shift]:
            train.verify_source_hashes(source_hashes)
            directory = args.output / f"{arm}-seed{seed}"
            command = [
                "--methods",
                "full",
                "--seeds",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--output",
                str(directory),
                "--signal-backend",
                "sparse",
            ]
            for key, value in ARMS[arm].items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if args.resume:
                command.append("--resume")
            print(f"ARM {arm} SEED {seed}: {ARMS[arm]}", flush=True)
            try:
                train.main(command)
            except Exception as error:
                train.record_failure(
                    directory,
                    context={"arm": arm, "seed": seed, "epochs": args.epochs},
                    error=error,
                )
                raise
            train.verify_source_hashes(source_hashes)
            result = json.loads((directory / f"full-seed{seed}" / "result.json").read_text())
            metrics = [
                json.loads(line)
                for line in (directory / f"full-seed{seed}" / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            rows.append(
                {
                    **result,
                    "arm": arm,
                    "total_training_seconds": sum(row["train_seconds"] for row in metrics),
                }
            )
            train.atomic_json(args.output / "comparison.json", summarize(rows))
    print(json.dumps(summarize(rows)["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
