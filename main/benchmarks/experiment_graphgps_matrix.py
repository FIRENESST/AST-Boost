"""Run the frozen GraphGPS-compatible ZINC baseline and H1/H2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch

from ast_boost.experiments.data import load_zinc_banks
from ast_boost.experiments.graphgps import GRAPHGPS_REFERENCE_COMMIT, GRAPHGPS_REFERENCE_URL
from ast_boost.experiments.model import REFERENCE_METHODS
from ast_boost.experiments.train import (
    PROJECT,
    arguments,
    atomic_json,
    run_one,
    source_snapshot,
    verify_source_hashes,
)

MATRIX_METHODS = (
    *REFERENCE_METHODS,
    "rwse_kernel_graphgps",
    "signnet_local",
    "kern",
    "full",
)


def paired_difference(rows, target: str, reference: str):
    targets = {row["seed"]: row for row in rows if row["method"] == target}
    references = {row["seed"]: row for row in rows if row["method"] == reference}
    seeds = sorted(targets.keys() & references.keys())
    differences = [
        targets[seed]["best_val_mae"] - references[seed]["best_val_mae"]
        for seed in seeds
    ]
    return {
        "target": target,
        "reference": reference,
        "seeds": seeds,
        "differences": differences,
        "mean": statistics.mean(differences) if differences else None,
        "interpretation": "negative favors target",
    }


def summarize(rows):
    methods = {}
    for method in MATRIX_METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["seed"]
        )
        if not selected:
            continue
        scores = [row["best_val_mae"] for row in selected]
        methods[method] = {
            "seeds": [row["seed"] for row in selected],
            "val_mae": scores,
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else None,
            "parameters": selected[0]["parameters"],
        }
    return {
        "methods": methods,
        "h2_kern_minus_signnet": paired_difference(rows, "kern", "signnet_graphgps"),
        "h2_kern_minus_matched_first_order": paired_difference(
            rows, "kern", "signnet_local"
        ),
        "h2_kern_minus_rwse": paired_difference(rows, "kern", "rwse_graphgps"),
        "h2_kernel_on_rwse": paired_difference(
            rows, "rwse_kernel_graphgps", "rwse_graphgps"
        ),
        "h1_full_minus_kern": paired_difference(rows, "full", "kern"),
        "test_evaluations": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", choices=MATRIX_METHODS, nargs="+", default=MATRIX_METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sign-hidden", type=int, default=32)
    parser.add_argument("--sign-layers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    if len(set(options.methods)) != len(options.methods) or len(set(options.seeds)) != len(
        options.seeds
    ):
        raise ValueError("methods and seeds must be unique")

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args = arguments(
        [
            "--output",
            str(output),
            "--methods",
            *options.methods,
            "--seeds",
            *map(str, options.seeds),
            "--epochs",
            str(options.epochs),
            "--warmup-epochs",
            str(options.warmup_epochs),
            "--batch-size",
            str(options.batch_size),
            "--sign-hidden",
            str(options.sign_hidden),
            "--sign-layers",
            str(options.sign_layers),
            "--train-limit",
            str(options.train_limit),
            "--device",
            options.device,
            "--precision",
            options.precision,
            "--backbone",
            "graphgps",
            "--scheduler",
            "cosine_warmup",
            "--field-scaling",
            "size",
            "--frequency-labels",
            "eigenvalue",
            "--kernel-spectrum",
            "pe",
        ]
    )
    args.resume = options.resume
    source_hashes = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
    }
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output", "resume", "prepare_only", "evaluate_test"}
        },
        "graphgps_reference": {
            "repository": GRAPHGPS_REFERENCE_URL,
            "commit": GRAPHGPS_REFERENCE_COMMIT,
            "compatibility_scope": (
                "ported GPSLayer, ZINC PE encoders, pooling/head and optimizer schedule; "
                "standalone data/training loop, not GraphGym runtime"
            ),
        },
        "hypotheses": {
            "H2": "kern versus public SignNet and RWSE",
            "H1": "full versus kern",
        },
        "ast_controls": {
            "frequency_labels": "eigenvalue",
            "kernel_spectrum": "pe",
            "kernel_diagonal": False,
            "field_scaling": "size",
        },
        "selection": "best validation MAE; test split is never loaded",
        "source_sha256": source_hashes,
        "runner_sha256": runner_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
    }
    protocol["source_snapshot_sha256"] = source_snapshot(output, source_hashes)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["protocol_sha256"] = protocol_hash
    manifest = output / "manifest.json"
    if manifest.exists():
        if not options.resume or json.loads(manifest.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing matrix requires identical protocol/source and --resume")
    else:
        atomic_json(manifest, protocol)

    banks = load_zinc_banks(
        PROJECT / "data/ZINC",
        PROJECT / ".cache/zinc/graphgps_matrix_v1",
        device=args.device,
        k=args.k,
        pairs=args.pairs,
        rw_steps=args.rw_steps,
        kernel_spectrum=args.kernel_spectrum,
        splits=("train", "val"),
    )
    atomic_json(output / "dataset.json", {name: bank.metadata for name, bank in banks.items()})
    rows = []
    results_path = output / "results.json"
    if options.resume and results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))

    order = []
    for index, seed in enumerate(options.seeds):
        offset = index % len(options.methods)
        rotated = options.methods[offset:] + options.methods[:offset]
        order.extend((method, seed) for method in rotated)
    for method, seed in order:
        verify_source_hashes(source_hashes)
        if any(row["method"] == method and row["seed"] == seed for row in rows):
            continue
        result = run_one(method, seed, banks, args, protocol_hash)
        rows.append(result)
        atomic_json(results_path, rows)
        atomic_json(output / "comparison.json", summarize(rows))
    print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
