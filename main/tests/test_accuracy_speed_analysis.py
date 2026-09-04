"""Complete paired evidence, honest time budgets, and no hidden failed runs."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def analysis():
    path = Path(__file__).resolve().parents[1] / "benchmarks/summarize_accuracy_speed.py"
    spec = importlib.util.spec_from_file_location("accuracy_speed_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_study(path):
    plan = {"epochs": 2, "seeds": [42, 43], "arms": {"control": {}}, "source_sha256": {}}
    write_json(path / "plan.json", plan)
    runs = []
    for seed in plan["seeds"]:
        root = path / f"control-seed{seed}" / f"full-seed{seed}"
        root.mkdir(parents=True)
        rows = [
            {"epoch": 1, "val_mae": 0.5, "train_seconds": 2, "val_seconds": 1},
            {"epoch": 2, "val_mae": 0.4, "train_seconds": 2, "val_seconds": 1},
        ]
        (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        runs.append(
            {
                "arm": "control",
                "seed": seed,
                "epochs": 2,
                "test_mae": None,
                "best_val_mae": 0.4,
                "total_training_seconds": 4,
                "train_epoch_seconds_median": 2,
            }
        )
    write_json(path / "comparison.json", {"runs": runs})
    return runs


def test_time_budget_includes_validation_and_never_extrapolates(analysis):
    rows = [
        {"val_mae": 0.5, "train_seconds": 2, "val_seconds": 1},
        {"val_mae": 0.4, "train_seconds": 2, "val_seconds": 1},
    ]
    assert analysis.first_target(rows, 0.4) == 6
    assert analysis.first_target(rows, 0.3) is None
    assert analysis.best_at(rows, 2.9) is None
    assert analysis.best_at(rows, 3) == 0.5
    assert analysis.best_at(rows, 5) == 0.5
    assert analysis.best_at(rows, 100) == 0.4
    assert analysis.paired_interval([0.1])["ci95"] is None


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "score", "epochs", "test", "nan"])
def test_rejects_incomplete_or_inconsistent_evidence(analysis, tmp_path, corruption):
    runs = fixture_study(tmp_path)
    if corruption == "missing":
        runs.pop()
    elif corruption == "duplicate":
        runs[1] = runs[0]
    elif corruption == "score":
        runs[0]["best_val_mae"] = 0.1
    elif corruption == "epochs":
        runs[0]["epochs"] = 3
    elif corruption == "test":
        runs[0]["test_mae"] = 0.3
    else:
        trace = tmp_path / "control-seed42/full-seed42/metrics.jsonl"
        trace.write_text(trace.read_text().replace('"val_mae": 0.4', '"val_mae": NaN'))
    write_json(tmp_path / "comparison.json", {"runs": runs})
    with pytest.raises(ValueError):
        analysis.read_study(tmp_path)


def test_complete_evidence_is_reaggregated(analysis, tmp_path):
    fixture_study(tmp_path)
    plan, report, runs, traces = analysis.read_study(tmp_path)
    assert plan["epochs"] == 2 and len(runs) == len(traces) == 2
    assert report["arms"]["control"]["val_mean"] == 0.4


def test_direct_training_refuses_missing_seed(analysis, tmp_path):
    write_json(
        tmp_path / "manifest.json",
        {
            "configuration": {
                "methods": ["full"],
                "evaluate_test": False,
                "seeds": [42],
                "epochs": 2,
            },
            "source_sha256": {},
            "study_hash": "example",
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        analysis.read_study(tmp_path)


def test_direct_training_complete_and_protocol_mismatch(analysis, tmp_path):
    write_json(
        tmp_path / "manifest.json",
        {
            "configuration": {
                "methods": ["full"],
                "evaluate_test": False,
                "seeds": [42],
                "epochs": 1,
            },
            "source_sha256": {},
            "study_hash": "example",
        },
    )
    root = tmp_path / "full-seed42"
    root.mkdir()
    result = {
        "seed": 42,
        "study_hash": "example",
        "epochs": 1,
        "test_mae": None,
        "best_val_mae": 0.4,
        "train_epoch_seconds_median": 2,
    }
    write_json(root / "result.json", result)
    (root / "metrics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 1,
                "val_mae": 0.4,
                "train_seconds": 2,
                "val_seconds": 1,
            }
        )
    )
    _, report, runs, _ = analysis.read_study(tmp_path)
    assert report["arms"]["additional_full"]["total_training_seconds_mean"] == 2
    assert runs["additional_full", 42]["best_val_mae"] == 0.4
    result["study_hash"] = "changed"
    write_json(root / "result.json", result)
    with pytest.raises(ValueError, match="manifest"):
        analysis.read_study(tmp_path)
