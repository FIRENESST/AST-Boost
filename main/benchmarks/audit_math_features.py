"""Verify complete paired study records, frozen code, and committed best weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch

from ast_boost.experiments.train import atomic_json, verify_source_hashes


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    plan = json.loads((study / "plan.json").read_text())
    verify_source_hashes(plan["source_sha256"])
    digest = plan.pop("plan_sha256")
    assert hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() == digest
    assert sha256(study / "source_snapshot.zip") == plan["source_snapshot_sha256"]
    assert sha256(study / "runner_snapshot.py") == plan["runner_sha256"]
    with zipfile.ZipFile(study / "source_snapshot.zip") as archive:
        assert set(archive.namelist()) == {
            name.replace("\\", "/") for name in plan["source_sha256"]
        }
        for name, expected in plan["source_sha256"].items():
            assert hashlib.sha256(archive.read(name.replace("\\", "/"))).hexdigest() == expected
    records = json.loads((study / "results.json").read_text())
    expected_runs = {(item["arm"], item["seed"]) for item in plan["run_order"]}
    assert len(records) == len(expected_runs)
    assert {(row["arm"], row["seed"]) for row in records} == expected_runs
    audited = []
    for row in records:
        directory = study / row["arm"] / f"full-seed{row['seed']}"
        expected_hash = hashlib.sha256(f"{digest}:{row['arm']}".encode()).hexdigest()
        assert row["study_hash"] == expected_hash and row["test_mae"] is None
        result = json.loads((directory / "result.json").read_text())
        assert all(row[key] == value for key, value in result.items())
        history = [
            json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        assert [h["epoch"] for h in history] == list(range(1, plan["configuration"]["epochs"] + 1))
        assert all(math.isfinite(value) for h in history for value in h.values())
        assert min(h["val_mae"] for h in history) == row["best_val_mae"]
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        assert last["history"] == history
        assert last["study_hash"] == best["study_hash"] == expected_hash
        assert last["epoch"] == plan["configuration"]["epochs"]
        assert last["best_epoch"] == best["epoch"] == row["best_epoch"]
        assert last["best"] == best["val_mae"] == row["best_val_mae"]
        assert last["best_model"].keys() == best["model"].keys()
        for key, tensor in best["model"].items():
            assert torch.equal(tensor, last["best_model"][key])
            assert torch.isfinite(tensor).all()
        assert all(torch.isfinite(tensor).all() for tensor in last["model"].values())
        audited.append(
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "epochs": last["epoch"],
                "best_sha256": sha256(directory / "best.pt"),
                "last_sha256": sha256(directory / "last.pt"),
                "best_committed_weights_identical": True,
            }
        )
    dataset = json.loads((study / "dataset.json").read_text())
    assert set(dataset) == {"train", "val"}
    assert dataset["train"]["graphs"] == 10000 and dataset["val"]["graphs"] == 1000
    output = {
        "plan_sha256": digest,
        "complete_runs": len(audited),
        "runs": audited,
        "current_source_matches_snapshot": True,
        "test_evaluations": 0,
        "comparison_sha256": sha256(study / "comparison.json"),
        "dataset_sha256": {split: info["dataset_sha256"] for split, info in dataset.items()},
    }
    atomic_json(study / "audit.json", output)
    print(json.dumps({key: value for key, value in output.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
