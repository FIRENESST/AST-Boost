"""Run the frozen GraphGPS-compatible Peptides validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ast_boost.experiments.graphgps import GRAPHGPS_REFERENCE_COMMIT, GRAPHGPS_REFERENCE_URL
from ast_boost.experiments.lrgb import (
    PEPTIDES_ATOM_FEATURE_DIMS,
    PEPTIDES_BOND_FEATURE_DIMS,
    PEPTIDES_TARGET_DIMS,
    load_lrgb_banks,
    validation_metric,
)
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.sampling import batches
from ast_boost.experiments.train import (
    PROJECT,
    atomic_checkpoint,
    atomic_json,
    autocast,
    cosine_with_warmup,
    seed_all,
    source_snapshot,
    synchronize,
    verify_source_hashes,
)

DEFAULT_METHODS = (
    "lappe_graphgps",
    "signnet_graphgps",
    "signnet_local",
    "kern",
    "full",
)
MATRIX_METHODS = (*DEFAULT_METHODS, "rwse_graphgps", "rwse_gated_full_graphgps")


def metric_name(dataset):
    return "ap" if dataset == "Peptides-func" else "mae"


def better(candidate, incumbent, dataset):
    return candidate > incumbent if dataset == "Peptides-func" else candidate < incumbent


def objective(logits, targets, dataset):
    if dataset == "Peptides-func":
        return F.binary_cross_entropy_with_logits(logits, targets)
    return F.l1_loss(logits, targets)


def publish_history(directory, history):
    path = directory / "metrics.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in history),
        encoding="utf-8",
    )
    temporary.replace(path)


def publish_best(directory, saved, dataset):
    atomic_checkpoint(
        directory / "best.pt",
        {
            "model": saved["best_model"],
            "epoch": saved["best_epoch"],
            "metric": saved["best"],
            "metric_name": metric_name(dataset),
            "protocol_sha256": saved["protocol_sha256"],
        },
    )


def evaluate(model, bank, args):
    model.eval()
    predictions, targets = [], []
    synchronize(bank.device)
    start = time.perf_counter()
    with torch.inference_mode(), autocast(args):
        for indices in batches(range(len(bank)), bank.counts, args.batch_size):
            batch = bank.batch(indices)
            predictions.append(model(batch).float().cpu())
            targets.append(batch.targets.float().cpu())
    synchronize(bank.device)
    value = validation_metric(torch.cat(predictions), torch.cat(targets), args.dataset)
    return value, time.perf_counter() - start


def make_model(method, args):
    return GPSRegressor(
        method,
        width=96,
        layers=4,
        heads=4,
        pe_dim=16,
        sign_hidden=args.sign_hidden,
        sign_layers=args.sign_layers,
        k=10,
        pairs=4,
        rw_steps=args.rw_steps,
        attention_dropout=0.5,
        field_scaling="size",
        frequency_labels="eigenvalue",
        kernel_spectrum=args.kernel_spectrum,
        kernel_diagonal=False,
        backbone="graphgps",
        node_feature_dims=PEPTIDES_ATOM_FEATURE_DIMS,
        edge_feature_dims=PEPTIDES_BOND_FEATURE_DIMS,
        output_dim=PEPTIDES_TARGET_DIMS[args.dataset],
        pooling="mean",
        head_type="linear",
        local_gnn="gatedgcn",
        reference_pe_dim=16 if method in {
            "lappe_graphgps", "signnet_graphgps", "rwse_graphgps", "rwse_gated_full_graphgps"
        } else 0,
    ).to(args.device)


def run_one(method, seed, banks, args, protocol_hash):
    directory = args.output / f"{method}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["protocol_sha256"] != protocol_hash:
            raise ValueError("completed result protocol does not match this study")
        validate_completed_run(directory, result, method, seed, args, protocol_hash)
        return result
    seed_all(seed)
    model = make_model(method, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0,
        fused=torch.device(args.device).type == "cuda",
    )
    scheduler = cosine_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, epochs=args.epochs
    )
    scaler = torch.amp.GradScaler(torch.device(args.device).type, enabled=args.precision == "fp16")
    maximize = args.dataset == "Peptides-func"
    best = -float("inf") if maximize else float("inf")
    best_epoch = 0
    best_model = None
    history = []
    start_epoch = 1
    peak_allocated_mib = 0.0
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["protocol_sha256"] != protocol_hash:
            raise ValueError("checkpoint protocol does not match this study")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        if [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
            raise ValueError("checkpoint history is incomplete or duplicated")
        peak_allocated_mib = saved.get("peak_allocated_mib", 0.0)
        best, best_epoch, best_model = (
            saved["best"],
            saved["best_epoch"],
            saved["best_model"],
        )
        start_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        numpy_rng = saved["numpy_rng"]
        np.random.set_state(
            (numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:])
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        publish_history(directory, history)
        publish_best(directory, saved, args.dataset)
    train_ids = np.arange(len(banks["train"]))
    durations = [row["train_seconds"] for row in history]
    print(
        f"RUN {args.dataset} {method} seed={seed} params={model.trainable_parameters} "
        f"train={len(train_ids)}",
        flush=True,
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=args.device)
        synchronize(args.device)
        started = time.perf_counter()
        for indices in batches(
            train_ids, banks["train"].counts, args.batch_size,
            seed=seed * 100000 + epoch, sampler=args.sampler,
        ):
            batch = banks["train"].batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                logits = model(batch).float()
                loss = objective(logits, batch.targets, args.dataset)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * len(indices)
        synchronize(args.device)
        duration = time.perf_counter() - started
        durations.append(duration)
        val_metric, val_seconds = evaluate(model, banks["val"], args)
        if better(val_metric, best, args.dataset):
            best, best_epoch = val_metric, epoch
            best_model = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_loss": float(loss_sum / len(train_ids)),
            f"val_{metric_name(args.dataset)}": val_metric,
            f"best_val_{metric_name(args.dataset)}": best,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": duration,
            "val_seconds": val_seconds,
        }
        history.append(row)
        scheduler.step()
        if torch.device(args.device).type == "cuda":
            peak_allocated_mib = max(
                peak_allocated_mib, torch.cuda.max_memory_allocated(args.device) / 2**20
            )
        numpy_rng = np.random.get_state()
        state = {
            "format_version": 2,
            "peak_allocated_mib": peak_allocated_mib,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best": best,
            "best_epoch": best_epoch,
            "best_model": best_model,
            "protocol_sha256": protocol_hash,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.device(args.device).type == "cuda"
            else [],
        }
        atomic_checkpoint(checkpoint, state)
        publish_history(directory, history)
        if best_epoch == epoch:
            publish_best(directory, state, args.dataset)
        print(json.dumps({"method": method, "seed": seed, **row}), flush=True)
    result = {
        "dataset": args.dataset,
        "method": method,
        "seed": seed,
        "metric": metric_name(args.dataset),
        "best_val_metric": best,
        "best_epoch": best_epoch,
        "parameters": model.trainable_parameters,
        "train_epoch_seconds_median": statistics.median(durations),
        "peak_allocated_mib": peak_allocated_mib,
        "protocol_sha256": protocol_hash,
        "test_evaluations": 0,
    }
    atomic_json(result_path, result)
    return result


def validate_completed_run(directory, result, method, seed, args, protocol_hash):
    """Reject incomplete or inconsistent evidence without rewriting completed runs."""
    required = ("last.pt", "best.pt", "metrics.jsonl")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("completed run is missing checkpoint or history evidence")
    saved = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    published = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    history = saved["history"]
    if (saved["protocol_sha256"] != protocol_hash or saved["epoch"] != args.epochs
            or [row["epoch"] for row in history] != list(range(1, args.epochs + 1))):
        raise ValueError("completed checkpoint does not cover the frozen protocol")
    metric = metric_name(args.dataset)
    selected = (max if args.dataset == "Peptides-func" else min)(
        history, key=lambda row: row[f"val_{metric}"]
    )
    expected = {
        "dataset": args.dataset, "method": method, "seed": seed, "metric": metric,
        "best_epoch": selected["epoch"], "best_val_metric": selected[f"val_{metric}"],
        "test_evaluations": 0,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("completed result disagrees with checkpoint history or run identity")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    if rows != history:
        raise ValueError("completed published history disagrees with checkpoint")
    if (saved["best"] != expected["best_val_metric"]
            or saved["best_epoch"] != expected["best_epoch"]
            or published["protocol_sha256"] != protocol_hash
            or published["epoch"] != saved["best_epoch"]
            or published["metric"] != saved["best"]
            or published["model"].keys() != saved["best_model"].keys()
            or any(not torch.equal(value, published["model"][key])
                   for key, value in saved["best_model"].items())):
        raise ValueError("completed best checkpoint disagrees with committed best state")


def summarize(rows, dataset):
    methods = {}
    for method in MATRIX_METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method), key=lambda row: row["seed"]
        )
        if selected:
            scores = [row["best_val_metric"] for row in selected]
            methods[method] = {
                "seeds": [row["seed"] for row in selected],
                "values": scores,
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else None,
                "parameters": selected[0]["parameters"],
            }
    comparisons = {}
    for target, reference, name in (
        ("kern", "signnet_graphgps", "h2_kern_minus_signnet"),
        ("kern", "signnet_local", "h2_kern_minus_matched_first_order"),
        ("kern", "lappe_graphgps", "h2_kern_minus_lappe"),
        ("full", "kern", "h1_full_minus_kern"),
        ("rwse_gated_full_graphgps", "rwse_graphgps", "h2_gated_full_minus_rwse"),
    ):
        left = {row["seed"]: row["best_val_metric"] for row in rows if row["method"] == target}
        right = {
            row["seed"]: row["best_val_metric"] for row in rows if row["method"] == reference
        }
        seeds = sorted(left.keys() & right.keys())
        differences = [left[seed] - right[seed] for seed in seeds]
        comparisons[name] = {
            "seeds": seeds,
            "differences": differences,
            "mean": statistics.mean(differences) if differences else None,
            "interpretation": (
                "positive favors target"
                if dataset == "Peptides-func"
                else "negative favors target"
            ),
        }
    return {
        "dataset": dataset,
        "metric": metric_name(dataset),
        "methods": methods,
        **comparisons,
        "test_evaluations": 0,
    }


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PEPTIDES_TARGET_DIMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", choices=MATRIX_METHODS, nargs="+", default=list(DEFAULT_METHODS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampler", choices=["random", "sortish"], default="random")
    parser.add_argument("--sign-hidden", type=int, default=32)
    parser.add_argument("--sign-layers", type=int, default=2)
    parser.add_argument("--rw-steps", type=int, default=0)
    parser.add_argument("--kernel-spectrum", choices=["pe", "all"], default="pe")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("require epochs > warmup_epochs >= 0")
    if args.batch_size < 1 or min(args.train_limit, args.val_limit, args.rw_steps) < 0:
        raise ValueError("invalid batch size or dataset/feature limit")
    if any(method.startswith("rwse_") for method in args.methods) and args.rw_steps < 1:
        raise ValueError("RWSE methods require --rw-steps > 0")
    if "rwse_gated_full_graphgps" in args.methods and args.kernel_spectrum != "all":
        raise ValueError("gated full residual requires --kernel-spectrum all")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("methods and seeds must be unique")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
    }
    protocol = {
        "dataset": args.dataset,
        "methods": args.methods,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "sign_hidden": args.sign_hidden,
        "sign_layers": args.sign_layers,
        "rw_steps": args.rw_steps,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "device": args.device,
        "precision": args.precision,
        "batch_sampler": args.sampler,
        "sortish_window_batches": 8 if args.sampler == "sortish" else None,
        "model": {
            "layer": "CustomGatedGCN+Transformer",
            "layers": 4,
            "width": 96,
            "heads": 4,
            "attention_dropout": 0.5,
            "pooling": "mean",
            "lap_pe_dim": 16,
            "lap_pe_frequencies": 10,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": 3e-4,
            "weight_decay": 0.0,
            "warmup_epochs": args.warmup_epochs,
            "schedule": "cosine_with_warmup",
            "clip_grad_norm": 1.0,
        },
        "ast_controls": {
            "frequency_labels": "eigenvalue",
            "kernel_spectrum": args.kernel_spectrum,
            "kernel_diagonal": False,
            "field_scaling": "size",
            "pairs": 4,
        },
        "selection": (
            f"best validation {metric_name(args.dataset)}; test split is never instantiated"
        ),
        "rwse_residual": {
            "gate": "tanh(gamma), gamma initialized exactly zero; shared across layers",
            "response": "Bernstein degree 8; coefficients exp(-2*j/8) at initialization",
            "standardize": False, "diagonal": False, "size_scaling": "none",
            "reference_pe_dim": 16, "backbone_training": "joint; no frozen baseline parameters",
        } if "rwse_gated_full_graphgps" in args.methods else None,
        "graphgps_reference": {
            "repository": GRAPHGPS_REFERENCE_URL,
            "commit": GRAPHGPS_REFERENCE_COMMIT,
            "scope": "standalone source port; not the GraphGym training runtime",
        },
        "source_sha256": source_hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "test_evaluations": 0,
    }
    protocol["source_snapshot_sha256"] = source_snapshot(args.output, source_hashes)
    runner_snapshot = args.output / "runner_snapshot.py"
    runner_bytes = Path(__file__).read_bytes()
    if runner_snapshot.exists():
        if runner_snapshot.read_bytes() != runner_bytes:
            raise ValueError("runner snapshot differs from current runner")
    else:
        runner_snapshot.write_bytes(runner_bytes)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["protocol_sha256"] = protocol_hash
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing matrix requires identical protocol/source and --resume")
    else:
        atomic_json(manifest_path, protocol)
    verify_source_hashes(source_hashes)
    banks = load_lrgb_banks(
        PROJECT / "data/LRGB",
        PROJECT / ".cache/lrgb",
        dataset=args.dataset,
        device=args.device,
        k=10,
        pairs=4,
        rw_steps=args.rw_steps,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        splits=("train", "val"),
        kernel_spectrum=args.kernel_spectrum,
    )
    identity = {
        name: {key: bank.metadata[key] for key in (
            "dataset_sha256", "target_sha256", "spectral_cache_sha256"
        )}
        for name, bank in banks.items()
    }
    identity_path = args.output / "dataset_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("dataset targets, topology or cached spectra changed")
    else:
        atomic_json(identity_path, identity)
    dataset_path = args.output / "dataset.json"
    if not dataset_path.exists():
        atomic_json(dataset_path, {name: bank.metadata for name, bank in banks.items()})
    if args.prepare_only:
        return
    rows = []
    results_path = args.output / "results.json"
    if args.resume and results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    order = []
    for index, seed in enumerate(args.seeds):
        offset = index % len(args.methods)
        rotated = args.methods[offset:] + args.methods[:offset]
        order.extend((method, seed) for method in rotated)
    for method, seed in order:
        verify_source_hashes(source_hashes)
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol["runner_sha256"]:
            raise RuntimeError("runner changed after the experiment plan was frozen")
        if any(row["method"] == method and row["seed"] == seed for row in rows):
            continue
        rows.append(run_one(method, seed, banks, args, protocol_hash))
        atomic_json(results_path, rows)
        atomic_json(args.output / "comparison.json", summarize(rows, args.dataset))
    print(json.dumps(summarize(rows, args.dataset), indent=2), flush=True)


if __name__ == "__main__":
    main()
