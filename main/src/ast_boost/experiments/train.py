"""Run controlled ZINC pilots; keep Full and every run's evidence on disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch

from .data import load_zinc_banks
from .model import METHODS, GPSRegressor

PROJECT = Path(__file__).resolve().parents[3]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["rwse", "lappe", "signnet_local", "kern", "full"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pe-dim", type=int, default=16)
    parser.add_argument("--sign-hidden", type=int, default=32)
    parser.add_argument("--node-layout", choices=["compact", "padded"], default="compact")
    parser.add_argument("--field-scaling", choices=["none", "size"], default="none")
    parser.add_argument("--signal-backend", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--k", type=int, default=8, help="minimum retained nonzero frequencies")
    parser.add_argument("--pairs", type=int, default=4, help="second-order low-frequency cutoff k0")
    parser.add_argument("--rw-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--kernel-eps", type=float, default=1e-6)
    parser.add_argument("--attention-dropout", type=float, default=0.5)
    parser.add_argument(
        "--train-limit", type=int, default=0, help="0 uses all 10,000 training graphs"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluate-test", action="store_true", help="Opt in only after protocol is frozen"
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def autocast(args):
    return torch.autocast(
        torch.device(args.device).type,
        dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16,
        enabled=args.precision != "fp32",
    )


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def verify_source_hashes(source_hashes):
    """Reject a study if its runnable package changes after the parent plan freezes."""
    current = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
    }
    if current != source_hashes:
        raise RuntimeError("runnable source changed after the experiment plan was frozen")


def record_failure(directory, *, context, error):
    """Persist a bounded failure record without modifying the run checkpoint."""
    checkpoint = directory / "full-seed{}".format(context["seed"]) / "last.pt"
    last_epoch = None
    checkpoint_sha256 = None
    if checkpoint.exists():
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        last_epoch = saved.get("epoch")
    failure = {
        **context,
        "exception_type": type(error).__name__,
        "error": str(error),
        "last_committed_epoch": last_epoch,
        "last_checkpoint_sha256": checkpoint_sha256,
        "policy": "retain committed metrics/checkpoints; never summarize incomplete repetitions",
    }
    path = directory.parent / "failures.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(failure)
    atomic_json(path, rows)


def source_snapshot(directory, source_hashes):
    """Freeze exact runnable source automatically, without replacing old evidence."""
    path = directory / "source_snapshot.zip"
    if not path.exists():
        temporary = directory / "source_snapshot.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, expected in source_hashes.items():
                content = (PROJECT / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError("source changed while creating the experiment snapshot")
                archive.writestr(name.replace("\\", "/"), content)
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        expected_names = {name.replace("\\", "/") for name in source_hashes}
        if set(archive.namelist()) != expected_names:
            raise ValueError("source snapshot contains a different file set")
        for name, expected in source_hashes.items():
            if hashlib.sha256(archive.read(name.replace("\\", "/"))).hexdigest() != expected:
                raise ValueError("source snapshot does not match the experiment protocol")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_epoch_artifacts(directory, saved, *, recovery=False):
    """Derive logs and best.pt from the single atomic epoch commit, last.pt.

    An interruption before last.pt commits replays the previous epoch. An
    interruption after it commits repairs these derived files without replaying
    the optimizer. A conflicting log is backed up instead of silently discarded.
    """
    metrics = directory / "metrics.jsonl"
    content = "".join(json.dumps(row, allow_nan=False) + "\n" for row in saved["history"])
    previous = metrics.read_text(encoding="utf-8") if metrics.exists() else ""
    if content != previous:
        if not content.startswith(previous):
            backup = directory / f"metrics-uncommitted-{uuid.uuid4().hex}.jsonl"
            shutil.copy2(metrics, backup)
        temporary = metrics.with_suffix(".jsonl.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(metrics)
    if recovery or saved["best_epoch"] == saved["epoch"]:
        atomic_checkpoint(
            directory / "best.pt",
            {
                "model": saved["best_model"],
                "epoch": saved["best_epoch"],
                "val_mae": saved["best"],
                "study_hash": saved["study_hash"],
            },
        )


def evaluate(model, bank, args):
    model.eval()
    total = torch.zeros((), device=bank.device)
    synchronize(bank.device)
    start = time.perf_counter()
    with torch.inference_mode(), autocast(args):
        for offset in range(0, len(bank), args.batch_size):
            batch = bank.batch(range(offset, min(offset + args.batch_size, len(bank))))
            prediction = model(batch).float()
            total += (prediction - batch.targets).abs().sum()
    synchronize(bank.device)
    value = float(total / len(bank))
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite evaluation MAE")
    return value, time.perf_counter() - start


def paired_summary(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)
    summary = {}
    for method, rows in grouped.items():
        scores = [r["best_val_mae"] for r in rows]
        summary[method] = {
            "n": len(rows),
            "val_mae_mean": statistics.mean(scores),
            "val_mae_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "train_epoch_seconds_median": statistics.median(
                r["train_epoch_seconds_median"] for r in rows
            ),
            "parameters": rows[0]["parameters"],
            "peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
        }
    paired = []
    if "kern" in grouped and "full" in grouped:
        kern = {r["seed"]: r for r in grouped["kern"]}
        for full in grouped["full"]:
            if full["seed"] in kern:
                paired.append(full["best_val_mae"] - kern[full["seed"]]["best_val_mae"])
    return {
        "methods": summary,
        "full_minus_kern_val": {
            "paired_differences": paired,
            "mean": statistics.mean(paired) if paired else None,
            "std": statistics.stdev(paired) if len(paired) > 1 else None,
            "interpretation": (
                "negative favors Full; pilot differences are not grounds to remove Full"
            ),
        },
    }


def run_one(method, seed, banks, args, study_hash):
    directory = args.output / f"{method}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "result.json"
    if result_file.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {result_file}")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if result["study_hash"] != study_hash:
            raise ValueError("completed result protocol does not match this study")
        return result
    seed_all(seed)
    model = GPSRegressor(
        method,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        pe_dim=args.pe_dim,
        sign_hidden=args.sign_hidden,
        attention_dropout=args.attention_dropout,
        kernel_eps=args.kernel_eps,
        node_layout=args.node_layout,
        field_scaling=args.field_scaling,
        signal_backend=args.signal_backend,
        k=args.k,
        pairs=args.pairs,
        rw_steps=args.rw_steps,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=torch.device(args.device).type == "cuda",
    )
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=args.plateau_patience, min_lr=args.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    scaler = torch.amp.GradScaler(torch.device(args.device).type, enabled=args.precision == "fp16")
    start_epoch, best, best_epoch = 1, float("inf"), 0
    durations = []
    history, best_model = [], None
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["study_hash"] != study_hash:
            raise ValueError("checkpoint protocol does not match this study")
        if saved.get("format_version") != 2:
            raise ValueError("legacy checkpoint: resume using its original source snapshot")
        history = saved["history"]
        if [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
            raise ValueError("checkpoint epoch history is incomplete or duplicated")
        best_model = saved["best_model"]
        publish_epoch_artifacts(directory, saved, recovery=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["rng"])
        random.setstate(saved["python_rng"])
        np_rng = saved["numpy_rng"]
        np.random.set_state((np_rng[0], np.asarray(np_rng[1], dtype=np.uint32), *np_rng[2:]))
        if torch.device(args.device).type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch, best, best_epoch = saved["epoch"] + 1, saved["best"], saved["best_epoch"]
        durations = saved["durations"]
    train_ids = np.arange(len(banks["train"]))
    if args.train_limit:
        train_ids = np.random.default_rng(12345).permutation(train_ids)[: args.train_limit]
    print(
        f"RUN {method} seed={seed} params={model.trainable_parameters} train={len(train_ids)}",
        flush=True,
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_lr = optimizer.param_groups[0]["lr"]
        # Sampling uses an independent generator: LapPE sign augmentation and
        # model-specific dropout cannot change training graph order.
        order = np.random.default_rng(seed * 100000 + epoch).permutation(train_ids)
        total = torch.zeros((), device=args.device)
        synchronize(args.device)
        start = time.perf_counter()
        for offset in range(0, len(order), args.batch_size):
            batch = banks["train"].batch(order[offset : offset + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                prediction = model(batch).float()
                loss = (prediction - batch.targets).abs().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach() * len(batch.targets)
        synchronize(args.device)
        duration = time.perf_counter() - start
        durations.append(duration)
        train_mae = float(total / len(order))
        val_mae, val_seconds = evaluate(model, banks["val"], args)
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            # Clone: state_dict itself aliases live model tensors. Keeping the
            # best weights in the epoch commit makes best.pt repairable too.
            best_model = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if args.scheduler == "plateau":
            scheduler.step(val_mae)
        else:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "best_val_mae": best,
            "train_seconds": duration,
            "val_seconds": val_seconds,
            "lr": epoch_lr,
            "next_lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        np_rng = np.random.get_state()
        committed = {
            "format_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
            "durations": durations,
            "history": history,
            "best_model": best_model,
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (np_rng[0], np_rng[1].tolist(), *np_rng[2:]),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "study_hash": study_hash,
        }
        atomic_checkpoint(checkpoint, committed)
        publish_epoch_artifacts(directory, committed)
        print(
            f"{method} seed={seed} epoch={epoch}/{args.epochs} "
            f"train={train_mae:.4f} val={val_mae:.4f} best={best:.4f} time={duration:.2f}s",
            flush=True,
        )
    saved = torch.load(directory / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(saved["model"])
    _, inference_seconds = evaluate(model, banks["val"], args)
    result = {
        "method": method,
        "seed": seed,
        "epochs": args.epochs,
        "study_hash": study_hash,
        "parameters": model.trainable_parameters,
        "best_epoch": best_epoch,
        "best_val_mae": best,
        "train_epoch_seconds_median": statistics.median(durations),
        "validation_inference_seconds": inference_seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 1024**2
        if torch.device(args.device).type == "cuda"
        else 0,
        "test_mae": evaluate(model, banks["test"], args)[0] if args.evaluate_test else None,
        "full_policy": "retain; no automatic model deletion or negative conclusion",
    }
    atomic_json(result_file, result)
    return result


def main(argv=None):
    args = arguments(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.train_limit < 0:
        raise ValueError("epochs, batch and threads must be positive; limit must be nonnegative")
    if args.plateau_patience < 0 or not 0 <= args.min_lr < args.lr:
        raise ValueError("patience must be nonnegative and 0 <= min-lr < lr")
    if min(args.k, args.rw_steps) < 1 or args.pairs < 0:
        raise ValueError("k and rw-steps must be positive; pairs must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        raise ValueError("methods and seeds must not repeat")
    torch.set_num_threads(args.threads)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"resume", "prepare_only", "output"}
    }
    source_hashes = {
        str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((PROJECT / "src" / "ast_boost").rglob("*.py"))
    }
    protocol = {"configuration": configuration, "source_sha256": source_hashes}
    study_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_file = args.output / "manifest.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not args.resume or previous["study_hash"] != study_hash:
            raise ValueError(
                "existing study: use --resume with identical code and protocol, or a new output"
            )
        snapshot_hash = source_snapshot(args.output, source_hashes)
        if previous.get("source_snapshot_sha256") != snapshot_hash:
            raise ValueError("source snapshot archive changed since this study started")
    else:
        snapshot_hash = source_snapshot(args.output, source_hashes)
        atomic_json(
            manifest_file,
            {
                **protocol,
                "study_hash": study_hash,
                "source_snapshot_sha256": snapshot_hash,
                "environment": {
                    "torch": str(torch.__version__),
                    "python": platform.python_version(),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                },
                "scope": "local GPS-style pilot, NOT official GraphGPS/SignNet reproduction",
                "cuda_scatter_bitwise_deterministic": False,
                "full_policy": "Full must be retained; no automatic elimination",
            },
        )
    banks = load_zinc_banks(
        PROJECT / "data" / "ZINC",
        PROJECT / ".cache" / "zinc" / "experiment_v1",
        device=args.device,
        k=args.k,
        pairs=args.pairs,
        rw_steps=args.rw_steps,
        dense_signals=args.signal_backend == "dense",
    )
    atomic_json(
        args.output / "dataset.json", {split: bank.metadata for split, bank in banks.items()}
    )
    if args.prepare_only:
        return
    results = []
    # Rotate run order by seed to reduce consistent thermal/order confounding.
    for index, seed in enumerate(args.seeds):
        offset = index % len(args.methods)
        for method in args.methods[offset:] + args.methods[:offset]:
            results.append(run_one(method, seed, banks, args, study_hash))
            atomic_json(args.output / "summary.json", paired_summary(results))
    print(json.dumps(paired_summary(results), indent=2), flush=True)


if __name__ == "__main__":
    main()
