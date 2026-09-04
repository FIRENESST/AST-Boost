"""Evaluate validation-only interpolation between retained best and last weights."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, autocast


def interpolate(best, last, alpha):
    result = {}
    for key, best_value in best.items():
        last_value = last[key]
        if best_value.is_floating_point():
            result[key] = torch.lerp(best_value, last_value, alpha)
        else:
            result[key] = best_value.clone()
    return result


def predict(model, bank, evaluation):
    values = []
    model.eval()
    with torch.inference_mode(), autocast(evaluation):
        for offset in range(0, len(bank), evaluation.batch_size):
            values.append(
                model(bank.batch(range(offset, min(offset + evaluation.batch_size, len(bank)))))
            )
    return torch.cat(values).float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.study_roots) != len(args.seeds):
        raise ValueError("provide one study root per seed")
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < alpha <= 1 for alpha in args.alphas):
        raise ValueError("alphas must be in (0, 1]")
    manifests = [
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for root in args.study_roots
    ]
    configurations = [manifest["configuration"] for manifest in manifests]
    if any(config["methods"] != ["full"] or config["evaluate_test"] for config in configurations):
        raise ValueError("expected validation-only Full studies")
    torch.set_num_threads(configurations[0]["threads"])
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-val-k8-p4-rw20-v1.pt",
            weights_only=True,
        ),
        "cuda",
    )
    targets = bank.tensors["targets"].cpu()
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
    for root, seed, configuration in zip(args.study_roots, args.seeds, configurations, strict=True):
        directory = root / f"full-seed{seed}"
        best_path, last_path = directory / "best.pt", directory / "last.pt"
        hashes = {
            "best": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            "last": hashlib.sha256(last_path.read_bytes()).hexdigest(),
        }
        best = torch.load(best_path, map_location="cpu", weights_only=True)["model"]
        last = torch.load(last_path, map_location="cpu", weights_only=True)["model"]
        model = GPSRegressor("full", **{key: configuration[key] for key in keys}).cuda()
        evaluation = SimpleNamespace(
            device="cuda",
            precision=configuration["precision"],
            batch_size=configuration["batch_size"],
        )
        candidates = {"best": best}
        candidates.update(
            {f"best_to_last_{alpha:g}": interpolate(best, last, alpha) for alpha in args.alphas}
        )
        scores = {}
        for name, state in candidates.items():
            model.load_state_dict(state)
            predictions = predict(model, bank, evaluation)
            scores[name] = float((predictions - targets).abs().mean())
        if hashes != {
            "best": hashlib.sha256(best_path.read_bytes()).hexdigest(),
            "last": hashlib.sha256(last_path.read_bytes()).hexdigest(),
        }:
            raise RuntimeError("checkpoint changed during read-only interpolation")
        row = {"seed": seed, "checkpoint_sha256": hashes, "val_mae": scores}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "rows": rows,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "test_evaluation": False,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "Exploratory read-only validation interpolation; no test evaluation and no "
            "checkpoint replacement. Alpha zero is the retained best, alpha one is last."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
