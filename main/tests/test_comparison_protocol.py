"""Experimental choices are frozen and paired before metrics can select a winner."""

import json
import sys

import pytest

from ast_boost.experiments import compare, train


def test_comparison_freezes_plan_rotates_arms_and_keeps_full(tmp_path, monkeypatch):
    calls = []

    def fake_train(argv):
        args = train.arguments(argv)
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["test_evaluation"] is False
        assert plan["source_sha256"]
        assert args.methods == ["full"] and not args.evaluate_test
        assert args.k == 8 and args.pairs == 4 and args.sign_hidden == 32
        assert args.signal_backend == "sparse"
        directory = args.output / f"full-seed{args.seeds[0]}"
        directory.mkdir(parents=True)
        arm = args.output.name.rsplit("-seed", 1)[0]
        calls.append((arm, args.seeds[0]))
        result = {"seed": args.seeds[0], "best_val_mae": 0.3, "train_epoch_seconds_median": 1.0}
        train.atomic_json(directory / "result.json", result)
        (directory / "metrics.jsonl").write_text('{"train_seconds": 1.0}\n')

    monkeypatch.setattr(train, "main", fake_train)
    command = ["compare", "--output", str(tmp_path), "--epochs", "2", "--seeds", "4", "5", "6"]
    monkeypatch.setattr(sys, "argv", command)
    compare.main()
    assert calls == [
        ("control", 4),
        ("size32", 4),
        ("size64", 4),
        ("size32", 5),
        ("size64", 5),
        ("control", 5),
        ("size64", 6),
        ("control", 6),
        ("size32", 6),
    ]
    with pytest.raises(ValueError, match="existing comparison"):
        compare.main()
    monkeypatch.setattr(sys, "argv", [*command, "--resume", "--epochs", "3"])
    with pytest.raises(ValueError, match="existing comparison"):
        compare.main()
    assert len(calls) == 9
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert all(row["n"] == 3 for row in report["arms"].values())


def test_comparison_pairs_by_seed_and_reports_incomplete_pairs():
    base = {"train_epoch_seconds_median": 1, "total_training_seconds": 2}
    rows = [
        dict(base, arm="control", seed=1, best_val_mae=0.5),
        dict(base, arm="size64", seed=2, best_val_mae=0.3),
        dict(base, arm="control", seed=2, best_val_mae=0.4),
        dict(base, arm="size64", seed=3, best_val_mae=0.1),
    ]
    report = compare.summarize(rows)
    paired = report["arms"]["size64"]["paired_minus_control"]
    assert len(paired) == 1 and paired[0]["seed"] == 2
    assert paired[0]["difference"] == pytest.approx(-0.1)
