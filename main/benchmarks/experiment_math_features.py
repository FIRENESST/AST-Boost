"""Preregistered paired ZINC ablations of the three spectral-information changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import scipy
import torch
from scipy.stats import t

from ast_boost.experiments.data import load_zinc_banks
from ast_boost.experiments.train import (
    PROJECT,
    arguments,
    atomic_json,
    run_one,
    source_snapshot,
    verify_source_hashes,
)

PRESETS = {
    "baseline": {"frequency_labels": "none", "kernel_spectrum": "pe", "kernel_diagonal": False},
    "frequency": {
        "frequency_labels": "eigenvalue",
        "kernel_spectrum": "pe",
        "kernel_diagonal": False,
    },
    "capacity_control": {
        "frequency_labels": "blind",
        "kernel_spectrum": "pe",
        "kernel_diagonal": False,
    },
    "full_kernel": {"frequency_labels": "none", "kernel_spectrum": "all", "kernel_diagonal": False},
    "diagonal": {"frequency_labels": "none", "kernel_spectrum": "pe", "kernel_diagonal": True},
    "combined": {
        "frequency_labels": "eigenvalue",
        "kernel_spectrum": "all",
        "kernel_diagonal": True,
    },
}


def paired_difference(rows, arm, reference):
    target = {r["seed"]: r for r in rows if r["arm"] == arm}
    base = {r["seed"]: r for r in rows if r["arm"] == reference}
    seeds = sorted(target.keys() & base.keys())
    delta = [target[s]["best_val_mae"] - base[s]["best_val_mae"] for s in seeds]
    if not delta:
        return None
    mean = statistics.mean(delta)
    half = (
        float(t.ppf(0.975, len(delta) - 1)) * statistics.stdev(delta) / len(delta) ** 0.5
        if len(delta) > 1
        else None
    )
    return {
        "reference": reference,
        "seeds": seeds,
        "differences": delta,
        "mean": mean,
        "descriptive_95pct_t_interval": [mean - half, mean + half] if half is not None else None,
        "all_seeds_better": all(d < 0 for d in delta),
        "interpretation": (
            "negative favors target; n=3 exploratory intervals, no multiplicity correction"
        ),
    }


def summarize(rows):
    arms = {}
    for arm in PRESETS:
        selected = [r for r in rows if r["arm"] == arm]
        if not selected:
            continue
        scores = [r["best_val_mae"] for r in selected]
        arms[arm] = {
            "n": len(selected),
            "seeds": [r["seed"] for r in selected],
            "val_mae": scores,
            "val_mae_mean": statistics.mean(scores),
            "val_mae_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "parameters": selected[0]["parameters"],
            "train_and_validation_seconds_mean": statistics.mean(
                r["train_and_validation_seconds"] for r in selected
            ),
            "wall_seconds_mean": statistics.mean(r["wall_seconds"] for r in selected),
            "peak_allocated_mib": max(r["peak_allocated_mib"] for r in selected),
            "versus_baseline": paired_difference(rows, arm, "baseline")
            if arm != "baseline"
            else None,
        }
    return {
        "arms": arms,
        "frequency_versus_capacity_control": paired_difference(
            rows, "frequency", "capacity_control"
        ),
        "test_evaluations": 0,
        "scope": "local GPS-style ablation, not a public benchmark or proof of H1",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--presets", choices=PRESETS, nargs="+", default=list(PRESETS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    if len(set(options.seeds)) != len(options.seeds) or len(set(options.presets)) != len(
        options.presets
    ):
        raise ValueError("seeds and presets must be unique")
    if min(options.epochs, options.batch_size) < 1:
        raise ValueError("epochs and batch-size must be positive")
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = arguments(
        [
            "--output",
            str(output),
            "--methods",
            "full",
            "--field-scaling",
            "size",
            "--scheduler",
            "cosine",
            "--batch-size",
            str(options.batch_size),
            "--epochs",
            str(options.epochs),
            "--device",
            options.device,
            "--seeds",
            *map(str, options.seeds),
        ]
    )
    args.resume = options.resume
    torch.set_num_threads(args.threads)
    source_hashes = {
        str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
    }
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runner_snapshot = output / "runner_snapshot.py"
    if runner_snapshot.exists():
        if hashlib.sha256(runner_snapshot.read_bytes()).hexdigest() != runner_hash:
            raise ValueError(
                "runner snapshot differs; replay its original code or use a new output"
            )
    else:
        runner_snapshot.write_bytes(Path(__file__).read_bytes())
    plan = {
        "configuration": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in {"output", "resume", "prepare_only"}
        },
        "arms": {name: PRESETS[name] for name in options.presets},
        "source_sha256": source_hashes,
        "runner_sha256": runner_hash,
        "source_snapshot_sha256": source_snapshot(output, source_hashes),
        "run_order": [
            {"seed": seed, "arm": arm}
            for i, seed in enumerate(options.seeds)
            for arm in options.presets[i % len(options.presets) :]
            + options.presets[: i % len(options.presets)]
        ],
        "environment": {
            "torch": str(torch.__version__),
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "selection": (
            "best checkpoint by validation MAE; compare paired seeds and capacity control; "
            "test split never loaded"
        ),
        "budget": (
            "same backbone, epochs, graph order, batch, scheduler; "
            "added parameters reported explicitly"
        ),
        "kernel": (
            "all includes zero modes, uses actual eigenvalues, and leaves k=8 PE/pairs=4 unchanged"
        ),
    }
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["plan_sha256"] = plan_hash
    path = output / "plan.json"
    if path.exists():
        if not options.resume or json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("existing study requires identical protocol/source and --resume")
    else:
        atomic_json(path, plan)
    banks = load_zinc_banks(
        PROJECT / "data/ZINC",
        PROJECT / ".cache/zinc/math_features_v2",
        device=args.device,
        k=args.k,
        pairs=args.pairs,
        rw_steps=args.rw_steps,
        kernel_spectrum="all",
        splits=("train", "val"),
    )
    dataset = {name: bank.metadata for name, bank in banks.items()}
    dataset_path = output / "dataset.json"
    if dataset_path.exists() and json.loads(dataset_path.read_text(encoding="utf-8")) != dataset:
        raise ValueError("dataset cache changed since the study was started")
    atomic_json(dataset_path, dataset)
    previous = output / "results.json"
    rows = json.loads(previous.read_text()) if options.resume and previous.exists() else []
    for item in plan["run_order"]:
        verify_source_hashes(source_hashes)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_hash:
            raise RuntimeError("experiment runner changed during the study")
        arm, seed = item["arm"], item["seed"]
        args.output = output / arm
        for key, value in PRESETS[arm].items():
            setattr(args, key, value)
        study_hash = hashlib.sha256(f"{plan_hash}:{arm}".encode()).hexdigest()
        print(f"ARM {arm} SEED {seed}", flush=True)
        start = time.perf_counter()
        try:
            result = run_one("full", seed, banks, args, study_hash)
        except Exception as error:
            atomic_json(
                output / f"failure-{arm}-seed{seed}.json",
                {
                    **item,
                    "type": type(error).__name__,
                    "error": str(error),
                    "plan_sha256": plan_hash,
                },
            )
            raise
        verify_source_hashes(source_hashes)
        log = args.output / f"full-seed{seed}" / "metrics.jsonl"
        history = [json.loads(line) for line in log.read_text().splitlines()]
        if len(history) != args.epochs or result["test_mae"] is not None:
            raise RuntimeError("incomplete or test-evaluated run is excluded from this study")
        prior = next((r for r in rows if r["arm"] == arm and r["seed"] == seed), None)
        rows = [r for r in rows if not (r["arm"] == arm and r["seed"] == seed)]
        rows.append(
            {
                **result,
                "arm": arm,
                "wall_seconds": prior["wall_seconds"] if prior else time.perf_counter() - start,
                "train_and_validation_seconds": sum(
                    r["train_seconds"] + r["val_seconds"] for r in history
                ),
            }
        )
        atomic_json(output / "results.json", rows)
        atomic_json(output / "comparison.json", summarize(rows))
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
