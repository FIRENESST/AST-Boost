"""Replay the next epoch in memory, without changing source training artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, autocast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    manifest = json.loads((options.run.parent / "manifest.json").read_text())
    args = SimpleNamespace(**manifest["configuration"])
    torch.set_num_threads(args.threads)
    checkpoint = options.run / "last.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
    model = GPSRegressor("full", **{key: getattr(args, key) for key in keys}).to(args.device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    optimizer.load_state_dict(saved["optimizer"])
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt",
            weights_only=True,
        ),
        args.device,
    )
    epoch = saved["epoch"] + 1
    seed = args.seeds[0]
    order = np.random.default_rng(seed * 100000 + epoch).permutation(len(bank))
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    model.train()
    result = {"checkpoint_sha256": digest, "replay_epoch": epoch, "batches": []}
    for offset in range(0, len(order), args.batch_size):
        batch = bank.batch(order[offset : offset + args.batch_size])
        optimizer.zero_grad(set_to_none=True)
        with autocast(args):
            prediction = model(batch).float()
            loss = (prediction - batch.targets).abs().mean()
        loss.backward()
        bad = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        row = {"batch": offset // args.batch_size, "loss": float(loss.detach()), "bad_grads": bad}
        if bad:
            row["failure"] = "nonfinite gradient elements"
            result["batches"].append(row)
            break
        try:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        except RuntimeError as error:
            row["failure"] = str(error)
            # FP64 norm distinguishes overflowing FP32 reduction from bad elements.
            row["fp64_norm"] = float(
                torch.stack(
                    [
                        parameter.grad.double().norm()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                ).norm()
            )
            result["batches"].append(row)
            break
        row["grad_norm"] = float(norm)
        result["batches"].append(row)
        optimizer.step()
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("source checkpoint changed during read-only replay")
    result["scope"] = "In-memory replay; CUDA scatter may diverge numerically; no artifact updates."
    with options.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["batches"][-3:], indent=2))


if __name__ == "__main__":
    main()
