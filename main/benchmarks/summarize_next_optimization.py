"""Audit and summarize B128/35 against the retained B64/20 validation study."""

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

import torch
from scipy.stats import t

from ast_boost.experiments.train import PROJECT


def snapshot_matches(root, manifest):
    path = root / "source_snapshot.zip"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["source_snapshot_sha256"]:
        raise ValueError("source snapshot archive hash mismatch")
    expected = {
        name.replace("\\", "/"): digest for name, digest in manifest["source_sha256"].items()
    }
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("source snapshot file set mismatch")
        if any(
            hashlib.sha256(archive.read(name)).hexdigest() != digest
            for name, digest in expected.items()
        ):
            raise ValueError("source snapshot content mismatch")
        if archive.testzip() is not None:
            raise ValueError("source snapshot CRC failure")
    return True


def read_candidate(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    if configuration["methods"] != ["full"] or configuration["evaluate_test"]:
        raise ValueError("candidate must be a validation-only Full study")
    snapshot_matches(root, manifest)
    rows = []
    for seed in configuration["seeds"]:
        directory = root / f"full-seed{seed}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if [row["epoch"] for row in metrics] != list(range(1, configuration["epochs"] + 1)):
            raise ValueError("candidate metric history is incomplete")
        if result["test_mae"] is not None or result["study_hash"] != manifest["study_hash"]:
            raise ValueError("candidate result does not match its validation-only manifest")
        if result["best_val_mae"] != min(row["val_mae"] for row in metrics):
            raise ValueError("candidate best result disagrees with its metrics")
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if best["epoch"] != result["best_epoch"] or last["epoch"] != configuration["epochs"]:
            raise ValueError("candidate checkpoint epochs disagree with result")
        if best["model"].keys() != last["best_model"].keys():
            raise ValueError("best checkpoint parameter set mismatch")
        for key, value in best["model"].items():
            if not torch.equal(value, last["best_model"][key]):
                raise ValueError("best checkpoint differs from atomically committed best model")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("candidate checkpoint contains a nonfinite tensor")
        rows.append(
            {
                **result,
                "train_seconds": sum(row["train_seconds"] for row in metrics),
                "train_val_seconds": sum(
                    row["train_seconds"] + row["val_seconds"] for row in metrics
                ),
                "checkpoint_sha256": {
                    name: hashlib.sha256((directory / f"{name}.pt").read_bytes()).hexdigest()
                    for name in ("best", "last")
                },
            }
        )
    return manifest, rows


def interval(values):
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * standard_deviation / len(values) ** 0.5
    return {"mean": mean, "std": standard_deviation, "ci95": [mean - half, mean + half]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-studies", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_plan = json.loads((args.baseline / "plan.json").read_text(encoding="utf-8"))
    baseline_report = json.loads((args.baseline / "comparison.json").read_text(encoding="utf-8"))
    baseline = {row["seed"]: row for row in baseline_report["runs"] if row["arm"] == "cosine"}
    manifests, candidate_rows = [], []
    for root in args.candidate_studies:
        manifest, rows = read_candidate(root)
        manifests.append(manifest)
        candidate_rows.extend(rows)
    reference_configuration = {
        key: value for key, value in manifests[0]["configuration"].items() if key != "seeds"
    }
    if any(
        {key: value for key, value in manifest["configuration"].items() if key != "seeds"}
        != reference_configuration
        or manifest["source_sha256"] != manifests[0]["source_sha256"]
        for manifest in manifests[1:]
    ):
        raise ValueError("candidate studies do not share one source and configuration")
    candidate = {row["seed"]: row for row in candidate_rows}
    seeds = baseline_plan["seeds"]
    if set(candidate) != set(seeds) or set(baseline) != set(seeds):
        raise ValueError("baseline and candidate do not contain the same unique seeds")
    paired = [
        {
            "seed": seed,
            "baseline_val_mae": baseline[seed]["best_val_mae"],
            "candidate_val_mae": candidate[seed]["best_val_mae"],
            "val_difference": candidate[seed]["best_val_mae"] - baseline[seed]["best_val_mae"],
            "baseline_train_val_seconds": baseline[seed]["total_train_val_seconds"],
            "candidate_train_val_seconds": candidate[seed]["train_val_seconds"],
        }
        for seed in seeds
    ]
    baseline_mean = statistics.mean(row["baseline_val_mae"] for row in paired)
    candidate_mean = statistics.mean(row["candidate_val_mae"] for row in paired)
    baseline_time = statistics.mean(row["baseline_train_val_seconds"] for row in paired)
    candidate_time = statistics.mean(row["candidate_train_val_seconds"] for row in paired)
    parameter_counts = {row["parameters"] for row in candidate.values()}
    if len(parameter_counts) != 1:
        raise ValueError("candidate parameter counts differ")
    current_source = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
    }
    if current_source != manifests[0]["source_sha256"]:
        raise ValueError("current runnable source differs from the candidate snapshot")
    result = {
        "baseline": {
            "configuration": {"batch_size": 64, "epochs": 20, "scheduler": "cosine"},
            "val_mae_mean": baseline_mean,
            "train_val_seconds_mean": baseline_time,
            "source_sha256": baseline_plan["source_sha256"],
        },
        "candidate": {
            "configuration": reference_configuration,
            "n": len(candidate),
            "val_mae_mean": candidate_mean,
            "val_mae_std": statistics.stdev(row["candidate_val_mae"] for row in paired),
            "train_val_seconds_mean": candidate_time,
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in candidate.values()),
            "parameters": parameter_counts.pop(),
            "source_sha256": manifests[0]["source_sha256"],
        },
        "paired": paired,
        "paired_val_difference": interval([row["val_difference"] for row in paired]),
        "val_mae_reduction_percent": 100 * (1 - candidate_mean / baseline_mean),
        "train_val_time_reduction_percent": 100 * (1 - candidate_time / baseline_time),
        "source_relation": (
            "Candidate runs share exact source snapshots. Baseline uses an earlier frozen source; "
            "new runtime options are disabled by default, but this is not a same-source rerun."
        ),
        "test_evaluations": 0,
        "full_policy": "retain all Full branches, checkpoints, controls and failed explorations",
        "integrity": {
            "candidate_runs": len(candidate),
            "candidate_source_snapshots_verified": len(manifests),
            "candidate_checkpoints_verified": 2 * len(candidate),
            "epochs_per_candidate": reference_configuration["epochs"],
            "finite": all(
                math.isfinite(row["candidate_val_mae"]) and row["candidate_val_mae"] >= 0
                for row in paired
            ),
        },
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
