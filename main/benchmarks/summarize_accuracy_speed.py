"""Paired validation accuracy, compute budgets and time-to-target (never test labels)."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t

from ast_boost.experiments.compare import summarize


def protocol_path(directory):
    plan = directory / "plan.json"
    return plan if plan.exists() else directory / "manifest.json"


def read_study(directory):
    """Require every declared run exactly once; derive scores from complete traces."""
    direct_training = not (directory / "plan.json").exists()
    if direct_training:
        manifest = json.loads((directory / "manifest.json").read_text())
        configuration = manifest["configuration"]
        if configuration["methods"] != ["full"] or configuration["evaluate_test"]:
            raise ValueError("additional training must be a validation-only Full study")
        arm = "additional_full"
        plan = {
            "epochs": configuration["epochs"],
            "seeds": configuration["seeds"],
            "source_sha256": manifest["source_sha256"],
            "arms": {arm: configuration},
        }
        report = {"runs": []}
        for seed in plan["seeds"]:
            result_file = directory / f"full-seed{seed}" / "result.json"
            if not result_file.exists():
                raise ValueError("additional Full training is incomplete")
            result = json.loads(result_file.read_text())
            if result["seed"] != seed or result["study_hash"] != manifest["study_hash"]:
                raise ValueError("additional result does not match its manifest")
            report["runs"].append({**result, "arm": arm})
    else:
        plan = json.loads((directory / "plan.json").read_text())
        report = json.loads((directory / "comparison.json").read_text())
    expected = {(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]}
    runs, traces = {}, {}
    for result in report["runs"]:
        key = (result["arm"], result["seed"])
        if key not in expected or key in runs:
            raise ValueError("unexpected or duplicate run in comparison")
        root = directory if direct_training else directory / f"{key[0]}-seed{key[1]}"
        path = root / f"full-seed{key[1]}" / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["epoch"] for row in rows] != list(range(1, plan["epochs"] + 1)):
            raise ValueError("incomplete or duplicated epoch history")
        if result["test_mae"] is not None or result["epochs"] != plan["epochs"]:
            raise ValueError("run does not match the validation-only epoch protocol")
        if any(
            not math.isfinite(row[name]) or row[name] < 0
            for row in rows
            for name in ("val_mae", "train_seconds", "val_seconds")
        ):
            raise ValueError("nonfinite or negative accuracy/timing evidence")
        total = sum(row["train_seconds"] for row in rows)
        if direct_training:
            result["total_training_seconds"] = total
        if result["best_val_mae"] != min(row["val_mae"] for row in rows) or not math.isclose(
            result["total_training_seconds"], total, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("comparison disagrees with its metric history")
        runs[key], traces[key] = result, rows
    if set(runs) != expected:
        raise ValueError("comparison is incomplete; do not summarize only finished/winning arms")
    return plan, summarize(list(runs.values())), runs, traces


def first_target(rows, target):
    elapsed = 0.0
    for row in rows:
        elapsed += row["train_seconds"] + row["val_seconds"]
        if row["val_mae"] <= target:
            return elapsed
    return None


def best_at(rows, budget):
    elapsed, best = 0.0, None
    for row in rows:
        elapsed += row["train_seconds"] + row["val_seconds"]
        if elapsed > budget + 1e-7:
            break
        best = row["val_mae"] if best is None else min(best, row["val_mae"])
    return best


def paired_interval(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": None}
    std = statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * std / len(values) ** 0.5
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "assumption": (
            "Approximate normal paired differences; small n, exploratory, "
            "no multiplicity correction."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-study", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan, report, runs, traces = read_study(args.study)
    if "control" not in plan["arms"]:
        raise ValueError("primary study requires a control arm")
    additional = None
    if args.additional_study is not None:
        extra_plan, extra_report, extra_runs, extra_traces = read_study(args.additional_study)
        if (
            extra_plan["seeds"] != plan["seeds"]
            or extra_plan["source_sha256"] != plan["source_sha256"]
        ):
            raise ValueError("additional budget study must use the same seeds and model source")
        for (arm, seed), result in extra_runs.items():
            name = f"{arm}_{extra_plan['epochs']}epochs"
            if (name, seed) in runs:
                raise ValueError("duplicate additional run")
            runs[name, seed], traces[name, seed] = result, extra_traces[arm, seed]
            report["arms"][name] = extra_report["arms"][arm]
            plan["arms"][name] = {**extra_plan["arms"][arm], "epochs": extra_plan["epochs"]}
        additional = {
            "epochs": extra_plan["epochs"],
            "path": str(args.additional_study),
            "configuration": extra_plan["arms"],
            "protocol_sha256": hashlib.sha256(
                protocol_path(args.additional_study).read_bytes()
            ).hexdigest(),
            "warning": (
                "Different epoch/sample budget; compare actual compute time, "
                "not equal-epoch accuracy."
            ),
        }
    comparisons = {}
    for arm in plan["arms"]:
        if arm == "control":
            continue
        paired = []
        for seed in plan["seeds"]:
            base, candidate = runs["control", seed], runs[arm, seed]
            raw_trace, new_trace = traces["control", seed], traces[arm, seed]
            budgets = [
                sum(r["train_seconds"] + r["val_seconds"] for r in trace)
                for trace in (raw_trace, new_trace)
            ]
            common_budget = min(budgets)
            paired.append(
                {
                    "seed": seed,
                    "val_difference": candidate["best_val_mae"] - base["best_val_mae"],
                    "training_time_ratio": candidate["total_training_seconds"]
                    / base["total_training_seconds"],
                    "train_val_time_ratio": budgets[1] / budgets[0],
                    "common_train_val_budget_seconds": common_budget,
                    "control_best_within_budget": best_at(raw_trace, common_budget),
                    "candidate_best_within_budget": best_at(new_trace, common_budget),
                }
            )
        comparisons[arm] = {
            "per_seed": paired,
            "paired_val": paired_interval([p["val_difference"] for p in paired]),
        }
    targets = {
        f"{arm}-seed{seed}": {
            str(target): first_target(traces[arm, seed], target) for target in (0.5, 0.4, 0.35)
        }
        for arm, seed in traces
    }
    result = {
        "aggregate": report["arms"],
        "total_train_val_seconds_mean": {
            arm: statistics.mean(
                sum(row["train_seconds"] + row["val_seconds"] for row in traces[arm, seed])
                for seed in plan["seeds"]
            )
            for arm in plan["arms"]
        },
        "additional_budget_study": additional,
        "paired": comparisons,
        "time_to_validation_mae": targets,
        "scope": (
            "Compute time includes training and validation, not IO/startup. "
            "Null means target not reached; do not average successful seeds alone."
        ),
        "plan_sha256": hashlib.sha256((args.study / "plan.json").read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
