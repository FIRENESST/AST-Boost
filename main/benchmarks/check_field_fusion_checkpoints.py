"""Compare fused/split BF16 validation predictions on retained Full weights."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, autocast


def predict(model, bank, args):
    values = []
    model.eval()
    with torch.inference_mode(), autocast(args):
        for offset in range(0, len(bank), args.batch_size):
            batch = bank.batch(range(offset, min(offset + args.batch_size, len(bank))))
            values.append(model(batch).float().cpu())
    return torch.cat(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.study / "manifest.json").read_text())
    configuration = manifest["configuration"]
    if configuration["methods"] != ["full"] or configuration["evaluate_test"]:
        raise ValueError("expected a validation-only Full study")
    torch.set_num_threads(configuration["threads"])
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-val-k8-p4-rw20-v1.pt",
            weights_only=True,
        ),
        "cuda",
    )
    evaluation = SimpleNamespace(
        device="cuda", precision=configuration["precision"], batch_size=configuration["batch_size"]
    )
    keys = (
        "width",
        "layers",
        "heads",
        "pe_dim",
        "sign_hidden",
        "k",
        "pairs",
        "rw_steps",
        "attention_dropout",
        "kernel_eps",
        "node_layout",
        "field_scaling",
        "signal_backend",
    )
    rows = []
    for seed in configuration["seeds"]:
        checkpoint = args.study / f"full-seed{seed}" / "best.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = GPSRegressor("full", **{key: configuration[key] for key in keys}).cuda()
        model.load_state_dict(saved["model"])
        model.pe.fuse_shared_fields = False
        split = predict(model, bank, evaluation)
        model.pe.fuse_shared_fields = True
        fused = predict(model, bank, evaluation)
        targets = bank.tensors["targets"].cpu()
        row = {
            "seed": seed,
            "split_val_mae": float((split - targets).abs().mean()),
            "fused_val_mae": float((fused - targets).abs().mean()),
            "prediction_max_abs_difference": float((split - fused).abs().max()),
            "prediction_mean_abs_difference": float((split - fused).abs().mean()),
            "checkpoint_sha256": digest,
        }
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source checkpoint changed during read-only comparison")
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "rows": rows,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "test_evaluation": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT / "src/ast_boost").rglob("*.py"))
        },
        "scope": "Read-only BF16 validation prediction equivalence; no checkpoint replacement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
