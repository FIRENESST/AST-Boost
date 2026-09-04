"""Record the dominant steady-state CUDA operators in a Full training step."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ast_boost.experiments.data import GraphBank
from ast_boost.experiments.model import GPSRegressor
from ast_boost.experiments.train import PROJECT, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.warmup, args.top) < 1:
        raise ValueError("profile dimensions must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    bank = GraphBank(
        torch.load(
            PROJECT / ".cache/zinc/experiment_v1/zinc-train-k8-p4-rw20-v1.pt",
            map_location="cpu",
            weights_only=True,
        ),
        "cuda",
    )
    batches = [bank.batch(range(offset, offset + args.batch_size)) for offset in range(args.steps)]
    seed_all(42)
    model = GPSRegressor("full", field_scaling="size").cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5, fused=True)

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(batch).float() - batch.targets).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()

    for index in range(args.warmup):
        step(batches[index % len(batches)])
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        for batch in batches:
            step(batch)
    torch.cuda.synchronize()
    events = sorted(
        profile.key_averages(), key=lambda event: event.self_device_time_total, reverse=True
    )
    rows = [
        {
            "operator": event.key,
            "count": event.count,
            "self_cuda_ms": event.self_device_time_total / 1000,
            "total_cuda_ms": event.device_time_total / 1000,
            "self_cpu_ms": event.self_cpu_time_total / 1000,
        }
        for event in events[: args.top]
    ]
    total_cuda = sum(event.self_device_time_total for event in events) / 1000
    report = {
        "configuration": vars(args) | {"output": str(args.output), "precision": "bf16"},
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "dataset_sha256": bank.metadata["dataset_sha256"],
        "profiled_self_cuda_ms": total_cuda,
        "top_operators": rows,
        "source_sha256": {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sorted((PROJECT / "src/ast_boost").rglob("*.py")), Path(__file__)]
        },
        "scope": "Profiler-instrumented steady Full training steps; timings include profiler cost.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"total_cuda_ms": total_cuda, "top": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
