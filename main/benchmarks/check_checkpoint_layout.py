"""Check saved, trained models on the full validation split without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchmark_backbone import load_reference

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT


def predictions(model, bank, precision):
    result = []
    model.cuda().eval()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"),
    ):
        for offset in range(0, len(bank), 32):
            result.append(model(bank.batch(range(offset, min(offset + 32, len(bank))))).float())
    return torch.cat(result).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["rwse", "kern", "full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The output is a new evidence directory, never a previous training run.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reference_class, reference_info = load_reference(args.reference_archive)
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-val-k8-p4-rw20-v1.pt", weights_only=True
        ),
        "cuda",
    )
    targets = bank.tensors["targets"].cpu()
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            checkpoint = args.study / f"{method}-seed{seed}" / "best.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            arrays = {"targets": targets.numpy()}
            for precision in ("fp32", "bf16"):
                values = {}
                for name, model_class in (
                    ("reference", reference_class),
                    ("compact", GPSRegressor),
                ):
                    model = model_class(method)
                    model.load_state_dict(saved["model"], strict=True)
                    values[name] = predictions(model, bank, precision)
                    arrays[f"{name}_{precision}"] = values[name].numpy()
                    del model
                old, new = values["reference"], values["compact"]
                assert torch.isfinite(old).all() and torch.isfinite(new).all()
                if precision == "fp32":
                    torch.testing.assert_close(new, old, atol=1e-4, rtol=1e-4)
                delta = (new - old).abs()
                row = {
                    "method": method,
                    "seed": seed,
                    "precision": precision,
                    "graphs": len(bank),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "reference_val_mae": float((old - targets).abs().mean()),
                    "compact_val_mae": float((new - targets).abs().mean()),
                    "prediction_difference_max": float(delta.max()),
                    "prediction_difference_mean": float(delta.mean()),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            np.savez_compressed(args.output / f"{method}-seed{seed}.npz", **arrays)
    report = {
        "reference": reference_info,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "Same saved weights, full validation split. FP32 numerical equivalence check; "
            "BF16 rounding sensitivity recorded separately. No test split or retraining."
        ),
        "rows": rows,
    }
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
