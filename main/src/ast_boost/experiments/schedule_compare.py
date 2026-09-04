"""Paired Full plateau/cosine study with a fixed fast architecture and budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from . import train

ARMS = {
    "control": {"scheduler": "plateau"},
    "cosine": {"scheduler": "cosine"},
}
COMMON = {
    "field_scaling": "size",
    "batch_size": 64,
    "lr": 0.001,
    "signal_backend": "sparse",
}


def summarize(rows):
    grouped = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    controls = {row["seed"]: row for row in grouped["control"]}
    result = {}
    for arm, values in grouped.items():
        if not values:
            continue
        scores = [row["best_val_mae"] for row in values]
        result[arm] = {
            "n": len(values),
            "val_mean": statistics.mean(scores),
            "val_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "train_seconds_mean": statistics.mean(row["total_training_seconds"] for row in values),
            "train_val_seconds_mean": statistics.mean(
                row["total_train_val_seconds"] for row in values
            ),
            "paired_minus_control": [
                {
                    "seed": row["seed"],
                    "difference": row["best_val_mae"] - controls[row["seed"]]["best_val_mae"],
                }
                for row in values
                if arm != "control" and row["seed"] in controls
            ],
        }
    return {"arms": result, "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("epochs must be positive and seeds unique")
    args.output = args.output.resolve()
    source_hashes = {
        str(path.relative_to(train.PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((train.PROJECT / "src/ast_boost").rglob("*.py"))
    }
    plan = {
        "epochs": args.epochs,
        "seeds": args.seeds,
        "arms": ARMS,
        "common": COMMON,
        "source_sha256": source_hashes,
        "method": "full",
        "selection": "best validation MAE",
        "test_evaluation": False,
        "hypothesis": "cosine decay may reach lower MAE in the same measured compute budget",
        "full_policy": "retain both arms, every Full branch and all checkpoints",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_file = args.output / "plan.json"
    if plan_file.exists():
        if not args.resume or json.loads(plan_file.read_text()) != plan:
            raise ValueError("existing study requires an identical plan/source and --resume")
    else:
        train.atomic_json(plan_file, plan)
    rows = []
    arm_names = list(ARMS)
    for index, seed in enumerate(args.seeds):
        order = arm_names[index % len(arm_names) :] + arm_names[: index % len(arm_names)]
        for arm in order:
            train.verify_source_hashes(source_hashes)
            directory = args.output / f"{arm}-seed{seed}"
            configuration = {**COMMON, **ARMS[arm]}
            command = [
                "--methods",
                "full",
                "--seeds",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--output",
                str(directory),
            ]
            for key, value in configuration.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if args.resume:
                command.append("--resume")
            print(f"ARM {arm} SEED {seed}: {configuration}", flush=True)
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
            run = directory / f"full-seed{seed}"
            result = json.loads((run / "result.json").read_text())
            metrics = [
                json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
            ]
            row = {
                **result,
                "arm": arm,
                "total_training_seconds": sum(metric["train_seconds"] for metric in metrics),
                "total_train_val_seconds": sum(
                    metric["train_seconds"] + metric["val_seconds"] for metric in metrics
                ),
            }
            rows.append(row)
            train.atomic_json(args.output / "comparison.json", summarize(rows))
    print(json.dumps(summarize(rows)["arms"], indent=2), flush=True)


if __name__ == "__main__":
    main()
