"""Scientific comparisons must not silently mix seeds or experimental controls."""

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def reporting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("report_lrgb_replication")


def fixture_studies(reporting):
    plan = {
        "reference_study": "reference", "dataset": "Peptides-struct", "epochs": 35,
        "warmup_epochs": 10, "batch_size": 128, "precision": "bf16", "sampler": "random",
        "seeds": [42, 43, 44], "methods": ["signnet_local", "kern"],
        "reuse": {"seed": 42, "methods": ["signnet_local", "kern"]},
        "new_studies": [{"directory": "new", "methods": ["signnet_local", "kern"],
                         "seeds": [43, 44]}],
        "comparisons": [{"target": "kern", "reference": "signnet_local"}],
    }
    protocol = dict.fromkeys(reporting.CONTROL_KEYS)
    protocol.update({key: plan[key] for key in (
        "dataset", "epochs", "warmup_epochs", "batch_size", "precision"
    )})
    protocol.update(batch_sampler="random", source_sha256={"model.py": "sha"},
                    runner_sha256="runner", methods=plan["methods"], seeds=[42])

    def row(method, seed, value):
        return {"method": method, "seed": seed, "best_val_metric": value,
                "parameters": 100 if method == "signnet_local" else 104, "test_evaluations": 0}

    reference = {
        "manifest": protocol,
        "dataset_identity": {"train": {"target_sha256": "targets"}},
        "results": [row("signnet_local", 42, 0.3), row("kern", 42, 0.31)],
    }
    new = deepcopy(reference)
    new["manifest"]["seeds"] = [43, 44]
    new["results"] = [row("kern", 44, 0.47), row("signnet_local", 43, 0.4),
                      row("kern", 43, 0.38), row("signnet_local", 44, 0.5)]
    return plan, {"reference": reference, "new": new}


def test_replication_pairs_by_seed_and_uses_sample_standard_deviation(reporting):
    plan, studies = fixture_studies(reporting)
    report = reporting.aggregate(plan, studies)
    assert report["methods"]["signnet_local"]["mean"] == pytest.approx(0.4)
    assert report["methods"]["signnet_local"]["sample_std"] == pytest.approx(0.1)
    comparison = report["contrasts"][0]
    assert comparison["differences"] == pytest.approx([0.01, -0.02, -0.03])
    assert comparison["mean"] == pytest.approx(-0.04 / 3)
    assert comparison["target_wins"] == 2 and comparison["reference_wins"] == 1


@pytest.mark.parametrize("key", ["source_sha256", "runner_sha256", "epochs", "optimizer"])
def test_replication_rejects_incompatible_controls(reporting, key):
    plan, studies = fixture_studies(reporting)
    studies["new"]["manifest"][key] = "changed"
    with pytest.raises(ValueError, match="incompatible study control"):
        reporting.aggregate(plan, studies)


def test_replication_rejects_changed_labels(reporting):
    plan, studies = fixture_studies(reporting)
    studies["new"]["dataset_identity"]["train"]["target_sha256"] = "different"
    with pytest.raises(ValueError, match="dataset identity"):
        reporting.aggregate(plan, studies)


@pytest.mark.parametrize("duplicate", [False, True])
def test_replication_rejects_missing_or_duplicate_seed(reporting, duplicate):
    plan, studies = fixture_studies(reporting)
    if duplicate:
        studies["new"]["results"].append(studies["new"]["results"][0])
    else:
        studies["new"]["results"].pop()
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        reporting.aggregate(plan, studies)


def test_evidence_reader_rejects_a_modified_manifest_before_using_results(reporting, tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"epochs": 35, "protocol_sha256": "not-the-content-hash"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest protocol hash is invalid"):
        reporting.read_study(tmp_path)
