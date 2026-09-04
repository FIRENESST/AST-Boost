"""Read-only validation checks for optional runtime implementations."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, autocast


def predict(model, bank, args, *, trim_padding=False, matmul_precision="highest"):
    values = []
    torch.set_float32_matmul_precision(matmul_precision)
    model.eval()
    with torch.inference_mode(), autocast(args):
        for offset in range(0, len(bank), args.batch_size):
            ids = range(offset, min(offset + args.batch_size, len(bank)))
            values.append(model(bank.batch(ids, trim_padding=trim_padding)).float().cpu())
    return torch.cat(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.study / "plan.json").read_text(encoding="utf-8"))
    if plan["method"] != "full" or plan["test_evaluation"]:
        raise ValueError("expected a validation-only Full study")
    first_manifest = json.loads(
        (args.study / f"cosine-seed{plan['seeds'][0]}" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    configuration = first_manifest["configuration"]
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
    targets = bank.tensors["targets"].cpu()
    rows = []
    for seed in plan["seeds"]:
        manifest = json.loads(
            (args.study / f"cosine-seed{seed}" / "manifest.json").read_text(encoding="utf-8")
        )
        seed_configuration = manifest["configuration"]
        checkpoint = args.study / f"cosine-seed{seed}" / f"full-seed{seed}" / "best.pt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = GPSRegressor("full", **{key: seed_configuration[key] for key in keys}).cuda()
        model.load_state_dict(saved["model"])
        reference = predict(model, bank, evaluation)
        model.pe.first_order_encoder.bmm_field_reduction = True
        model.pe.second_order_encoder.bmm_field_reduction = True
        bmm = predict(model, bank, evaluation)
        model.pe.first_order_encoder.bmm_field_reduction = False
        model.pe.second_order_encoder.bmm_field_reduction = False
        trimmed = predict(model, bank, evaluation, trim_padding=True)
        tf32 = predict(model, bank, evaluation, matmul_precision="high")
        methods = {"reference": reference, "bmm_reduction": bmm, "trimmed": trimmed, "tf32": tf32}
        row = {
            "seed": seed,
            "checkpoint_sha256": digest,
            "implementations": {
                name: {
                    "val_mae": float((values - targets).abs().mean()),
                    "prediction_mean_abs_difference": float((values - reference).abs().mean()),
                    "prediction_max_abs_difference": float((values - reference).abs().max()),
                }
                for name, values in methods.items()
            },
        }
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
            raise RuntimeError("checkpoint changed during read-only evaluation")
        rows.append(row)
        print(json.dumps(row), flush=True)
    torch.set_float32_matmul_precision("highest")
    report = {
        "rows": rows,
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "test_evaluation": False,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": (
            "Read-only BF16 validation predictions on retained Cosine checkpoints; "
            "no checkpoint or prior result is modified."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
