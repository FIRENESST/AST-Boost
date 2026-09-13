"""Alternating CUDA timing of trained spectral ablation models on one fixed batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ast_boost.experiments.data import load_zinc_banks
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, atomic_json, verify_source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if min(args.repeats, args.iterations) < 1 or args.warmup < 0:
        raise ValueError("invalid timing budget")
    study = args.study.resolve()
    path = study / "step_timing.json"
    if path.exists():
        raise FileExistsError(path)
    plan = json.loads((study / "plan.json").read_text())
    verify_source_hashes(plan["source_sha256"])
    config = plan["configuration"]
    torch.set_num_threads(config["threads"])
    banks = load_zinc_banks(
        PROJECT / "data/ZINC",
        PROJECT / ".cache/zinc/math_features_v2",
        device="cuda",
        k=config["k"],
        pairs=config["pairs"],
        rw_steps=config["rw_steps"],
        kernel_spectrum="all",
        splits=("train",),
    )
    ids = np.random.default_rng(12345).permutation(len(banks["train"]))[: config["batch_size"]]
    batch = banks["train"].batch(ids)
    models, checkpoints = {}, {}
    for arm, parameters in plan["arms"].items():
        model = (
            GPSRegressor(
                "full",
                **{
                    key: config[key]
                    for key in (
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
                },
                **parameters,
            )
            .cuda()
            .train()
        )
        checkpoint = study / arm / "full-seed42/best.pt"
        saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
        model.load_state_dict(saved["model"])
        models[arm] = model
        checkpoints[arm] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    samples = {arm: [] for arm in models}

    def step(model):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(batch).float()
            loss = (prediction - batch.targets).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)

    names = list(models)
    for repetition in range(args.repeats):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        if repetition % 2:
            order.reverse()
        for name in order:
            model = models[name]
            for _ in range(args.warmup):
                step(model)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                step(model)
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end) / args.iterations)
    verify_source_hashes(plan["source_sha256"])
    atomic_json(
        path,
        {
            "scope": (
                f"fixed B{config['batch_size']} batch, BF16 forward/backward/clip; "
                "excludes batching, AdamW and IO"
            ),
            "checkpoint_seed": 42,
            "checkpoint_sha256": checkpoints,
            "graph_ids": ids.tolist(),
            "repeats": args.repeats,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "source_sha256": plan["source_sha256"],
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(),
            "milliseconds_per_step": samples,
            "median_milliseconds_per_step": {
                name: statistics.median(values) for name, values in samples.items()
            },
            "note": (
                "no optimizer updates or checkpoint writes; "
                "BatchNorm running statistics update in memory"
            ),
        },
    )
    print(json.dumps(json.loads(path.read_text())["median_milliseconds_per_step"], indent=2))


if __name__ == "__main__":
    main()
