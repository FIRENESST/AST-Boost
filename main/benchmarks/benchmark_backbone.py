"""Paired whole-model timing against the retained padded implementation.

Optional --reference-archive loads model.py from a trusted local source backup;
never pass an untrusted archive (the selected Python source is executed).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, seed_all


def load_reference(archive):
    if archive is None:
        return lambda method: GPSRegressor(method, node_layout="padded"), None
    source_name = "main/src/ast_boost/experiments/model.py"
    with zipfile.ZipFile(archive) as handle:
        source = handle.read(source_name)
    module = types.ModuleType("ast_boost.experiments._trusted_backup_model")
    module.__package__ = "ast_boost.experiments"
    exec(compile(source, str(archive) + "/" + source_name, "exec"), module.__dict__)
    return module.GPSRegressor, {
        "archive": str(archive.resolve()),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(source).hexdigest(),
    }


def measure(factory, state, bank, selections, *, static_layout, mode, precision, warmup):
    model = factory().cuda()
    model.load_state_dict(state)
    model.train(mode == "train")
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5, fused=True)
        if mode == "train"
        else None
    )
    seed_all(314159)

    def step(ids, model=model, optimizer=optimizer):
        batch = bank.batch(ids, static_layout=static_layout)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss = (model(batch).float() - batch.targets).abs().mean()
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        return loss.detach()

    context = torch.enable_grad if mode == "train" else torch.inference_mode
    with context():
        for i in range(warmup):
            step(selections[i % len(selections)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for ids in selections:
            loss = step(ids)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / len(selections)
    assert torch.isfinite(loss), "nonfinite benchmark loss"
    result = {
        "milliseconds_per_batch": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    del model, optimizer, loss
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["rwse", "kern", "full"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--reference-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.iterations, args.warmup, args.batch_size) < 1:
        raise ValueError("all dimensions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.set_num_threads(4)
    reference_class, reference_info = load_reference(args.reference_archive)
    payload = torch.load(
        PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt",
        map_location="cpu",
        weights_only=True,
    )
    bank = GraphBank(payload, "cuda")
    del payload
    rng = np.random.default_rng(20260903)
    selections = [
        rng.choice(len(bank), size=args.batch_size, replace=False) for _ in range(args.iterations)
    ]
    results = {}
    for method in args.methods:
        seed_all(42)
        initial = GPSRegressor(method)
        state = {key: value.clone() for key, value in initial.state_dict().items()}
        parameters = initial.trainable_parameters
        del initial
        factories = {
            "padded_reference": lambda method=method: reference_class(method),
            "compact": lambda method=method: GPSRegressor(method),
        }
        # Independent source/state compatibility + real-data FP32 output check.
        outputs = []
        for name, factory in factories.items():
            model = factory().cuda().eval()
            model.load_state_dict(state, strict=True)
            with torch.inference_mode():
                outputs.append(model(bank.batch(selections[0])).cpu())
            del model
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-4, rtol=2e-4)
        difference = float((outputs[0] - outputs[1]).abs().max())
        results[method] = {"parameters": parameters, "fp32_max_abs_difference": difference}
        for mode in ("inference", "train"):
            samples = {name: [] for name in factories}
            for repeat in range(args.repeats):
                order = list(factories)
                if repeat % 2:
                    order.reverse()
                for name in order:
                    sample = measure(
                        factories[name],
                        state,
                        bank,
                        selections,
                        static_layout=name == "compact",
                        mode=mode,
                        precision=args.precision,
                        warmup=args.warmup,
                    )
                    samples[name].append(sample)
                    print(f"{method} {mode} repeat={repeat + 1} {name}: {sample}", flush=True)
            timings = {
                name: statistics.median(row["milliseconds_per_batch"] for row in rows)
                for name, rows in samples.items()
            }
            results[method][mode] = {
                "samples": samples,
                "median_ms": timings,
                "latency_reduction_percent": 100
                * (1 - timings["compact"] / timings["padded_reference"]),
            }
    report = {
        "configuration": {
            key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()
        },
        "environment": {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(),
            "threads": torch.get_num_threads(),
        },
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "reference": reference_info,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "selection_sha256": hashlib.sha256(np.stack(selections).tobytes()).hexdigest(),
        "scope": (
            "Whole local GPS-style model; includes bank collation, forward, and for train "
            "backward/gradient clipping/fused AdamW. No validation, IO or preprocessing. "
            "Identical initial weights each round; not an accuracy/convergence experiment."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects any previous evidence even if another run finished meanwhile.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                m: {k: v for k, v in r.items() if k in {"parameters", "fp32_max_abs_difference"}}
                for m, r in results.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
