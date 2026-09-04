"""Train-only, offline masked-BN recalibration; never replace original checkpoints.

This is an explicit exploratory postprocessing step, not official SWA. Dropout
is disabled. BN layers normalize each calibration batch while their input
population moments are accumulated, so deeper-layer statistics are approximate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor, MaskedBatchNorm
from ast_boost.experiments.train import PROJECT, atomic_checkpoint, autocast, evaluate, synchronize


def recalibrate(model, bank, indices, args):
    """Pool valid-node moments with between-batch correction, without labels."""
    norms = [m for m in model.modules() if isinstance(m, MaskedBatchNorm)]
    if any("forward_compact" in m.__dict__ for m in norms):
        raise ValueError("refusing to replace an existing instance-specific forward method")
    states = {m: {"count": 0} for m in norms}
    modes = {m: m.training for m in model.modules()}
    originals = {m: (m.running_mean.clone(), m.running_var.clone()) for m in norms}
    success = False

    def wrap(original, state):
        def forward(module, real):
            variance, mean = torch.var_mean(real.float(), dim=0, correction=0)
            count = len(real)
            if not state["count"]:
                state.update(count=count, mean=mean, m2=variance * count)
            else:
                previous = state["count"]
                total = previous + count
                delta = mean - state["mean"]
                state["m2"] = (
                    state["m2"] + variance * count + delta.square() * (previous * count / total)
                )
                state["mean"] = state["mean"] + delta * (count / total)
                state["count"] = total
            return original(real)

        return forward

    synchronize(bank.device)
    start = time.perf_counter()
    try:
        model.eval()
        for module in norms:
            module.training = True
            module.forward_compact = MethodType(
                wrap(module.forward_compact, states[module]), module
            )
        with torch.inference_mode(), autocast(args):
            for offset in range(0, len(indices), args.batch_size):
                model(bank.batch(indices[offset : offset + args.batch_size]))
        if any(not state["count"] for state in states.values()):
            raise ValueError("empty calibration data or an unvisited BatchNorm layer")
        for module, state in states.items():
            variance = state["m2"] / max(state["count"] - 1, 1)
            if not torch.isfinite(state["mean"]).all() or not torch.isfinite(variance).all():
                raise FloatingPointError("nonfinite calibration moments")
            module.running_mean.copy_(state["mean"])
            module.running_var.copy_(variance)
        success = True
    finally:
        for module in norms:
            if "forward_compact" in module.__dict__:
                del module.forward_compact
            if not success:
                module.running_mean.copy_(originals[module][0])
                module.running_var.copy_(originals[module][1])
        for module, mode in modes.items():
            module.training = mode
    synchronize(bank.device)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", type=int, default=2048)
    args = parser.parse_args()
    if args.graphs < 1:
        raise ValueError("calibration graphs must be positive")
    plan = json.loads((args.study / "plan.json").read_text())
    if any(
        hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest
        for name, digest in plan["source_sha256"].items()
    ):
        raise ValueError("model source changed since the parent experiment")
    expected = [(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]]
    if any(
        not (args.study / f"{arm}-seed{seed}" / f"full-seed{seed}" / "result.json").exists()
        for arm, seed in expected
    ):
        raise ValueError("finish the entire paired study before postprocessing")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    banks = {
        split: GraphBank(
            torch.load(
                PROJECT / f".cache/zinc/experiment_v1/zinc-{split}-k8-p4-rw20-v1.pt",
                weights_only=True,
            ),
            "cuda",
        )
        for split in ("train", "val")
    }
    indices = np.random.default_rng(9173).permutation(len(banks["train"]))[: args.graphs]
    with (args.output / "plan.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "parent_plan_sha256": hashlib.sha256(
                    (args.study / "plan.json").read_bytes()
                ).hexdigest(),
                "calibration_graphs": len(indices),
                "selection_seed": 9173,
                "dataset_sha256": {
                    split: bank.metadata["dataset_sha256"] for split, bank in banks.items()
                },
                "test_evaluation": False,
                "selection": "all declared arms/seeds; no checkpoint replacement",
            },
            handle,
            indent=2,
        )
    rows = []
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
    for arm, seed in expected:
        directory = args.study / f"{arm}-seed{seed}"
        configuration = json.loads((directory / "manifest.json").read_text())["configuration"]
        if (
            configuration["k"] != 8
            or configuration["pairs"] != 4
            or configuration["signal_backend"] != "sparse"
        ):
            raise ValueError("this diagnostic expects the declared k8/p4 sparse study")
        checkpoint = directory / f"full-seed{seed}" / "best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = GPSRegressor("full", **{key: configuration[key] for key in keys}).cuda()
        model.load_state_dict(saved["model"])
        evaluation = SimpleNamespace(
            device="cuda",
            precision=configuration["precision"],
            batch_size=configuration["batch_size"],
        )
        before, _ = evaluate(model, banks["val"], evaluation)
        seconds = recalibrate(model, banks["train"], indices, evaluation)
        after, _ = evaluate(model, banks["val"], evaluation)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.cpu(), saved["model"][name], atol=0, rtol=0)
        row = {
            "arm": arm,
            "seed": seed,
            "original_val_mae": before,
            "calibrated_val_mae": after,
            "calibration_seconds": seconds,
            "calibration_graphs": len(indices),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        rows.append(row)
        atomic_checkpoint(
            args.output / f"{arm}-seed{seed}-calibrated.pt",
            {"model": model.state_dict(), "source_epoch": saved["epoch"], "calibration": row},
        )
        print(json.dumps(row), flush=True)
        del model
    with (args.output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": rows,
                "selection_seed": 9173,
                "calibration_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": (
                    "Train-only statistics; unchanged parameters; approximate deeper BN moments; "
                    "validation only, no test evaluation."
                ),
            },
            handle,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
