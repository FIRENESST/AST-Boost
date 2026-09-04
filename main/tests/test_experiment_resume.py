"""Checkpoint recovery and protocol-protection tests on CPU toy graphs."""

import json
from types import SimpleNamespace

import pytest
import torch

from ast_boost.experiments.data import GraphBank, build_payload
from ast_boost.experiments.train import run_one


@pytest.mark.parametrize("phase", ["before_last", "after_last", "before_best", "after_best"])
def test_interrupted_training_restores_optimizer_and_rng_exactly(tmp_path, monkeypatch, phase):
    import ast_boost.experiments.train as training

    torch.set_num_threads(1)
    graph = SimpleNamespace(
        num_nodes=3,
        x=torch.tensor([[0], [1], [2]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.ones(4, dtype=torch.long),
        y=torch.tensor([0.3]),
    )
    bank = GraphBank(build_payload([graph, graph]))
    banks = {"train": bank, "val": bank}
    args = SimpleNamespace(
        output=tmp_path / "continuous",
        resume=False,
        device="cpu",
        precision="fp32",
        width=8,
        layers=1,
        heads=2,
        pe_dim=2,
        sign_hidden=4,
        attention_dropout=0.1,
        kernel_eps=1e-6,
        node_layout="compact",
        field_scaling="none",
        signal_backend="sparse",
        k=8,
        pairs=4,
        rw_steps=20,
        lr=0.001,
        scheduler="plateau",
        plateau_patience=10,
        min_lr=1e-6,
        weight_decay=1e-5,
        train_limit=0,
        epochs=2,
        batch_size=2,
        evaluate_test=False,
    )
    expected = run_one("full", 42, banks, args, "protocol")
    reference = torch.load(args.output / "full-seed42" / "last.pt", weights_only=True)
    save = training.atomic_checkpoint

    def interrupted_save(path, state):
        if phase == "before_last" and path.name == "last.pt" and state["epoch"] == 2:
            torch.save(state, path.with_suffix(".tmp"))
            raise RuntimeError("simulated interruption before atomic commit")
        if phase == "before_best" and path.name == "best.pt":
            raise RuntimeError("simulated interruption before best publication")
        save(path, state)
        if phase == "after_last" and path.name == "last.pt" and state["epoch"] == 2:
            raise RuntimeError("simulated interruption after checkpoint")
        if phase == "after_best" and path.name == "best.pt":
            raise RuntimeError("simulated interruption after best publication")

    args.output = tmp_path / "interrupted"
    monkeypatch.setattr(training, "atomic_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_one("full", 42, banks, args, "protocol")
    monkeypatch.setattr(training, "atomic_checkpoint", save)
    args.resume = True
    actual = run_one("full", 42, banks, args, "protocol")
    assert actual["best_val_mae"] == expected["best_val_mae"]
    recovered = torch.load(args.output / "full-seed42" / "last.pt", weights_only=True)
    for key, tensor in reference["model"].items():
        torch.testing.assert_close(tensor, recovered["model"][key], rtol=0, atol=0)
    for key, tensor in reference["best_model"].items():
        torch.testing.assert_close(tensor, recovered["best_model"][key], rtol=0, atol=0)
    torch.testing.assert_close(reference["rng"], recovered["rng"], rtol=0, atol=0)
    assert reference["python_rng"] == recovered["python_rng"]
    assert reference["numpy_rng"] == recovered["numpy_rng"]
    rows = [
        json.loads(line)
        for line in (args.output / "full-seed42" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2]
    for actual_row, expected_row in zip(rows, reference["history"], strict=True):
        for key in ("train_mae", "val_mae", "best_val_mae", "lr"):
            assert actual_row[key] == expected_row[key]


def test_completed_run_resume_is_read_only_and_full_is_retained(tmp_path):
    torch.set_num_threads(1)
    graph = SimpleNamespace(
        num_nodes=3,
        x=torch.tensor([[0], [1], [2]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.ones(4, dtype=torch.long),
        y=torch.tensor([0.3]),
    )
    bank = GraphBank(build_payload([graph, graph]))
    args = SimpleNamespace(
        output=tmp_path,
        resume=False,
        device="cpu",
        precision="fp32",
        width=8,
        layers=1,
        heads=2,
        pe_dim=2,
        sign_hidden=4,
        attention_dropout=0.0,
        kernel_eps=1e-6,
        node_layout="compact",
        field_scaling="none",
        signal_backend="sparse",
        k=8,
        pairs=4,
        rw_steps=20,
        lr=0.001,
        scheduler="plateau",
        plateau_patience=10,
        min_lr=1e-6,
        weight_decay=1e-5,
        train_limit=0,
        epochs=1,
        batch_size=2,
        evaluate_test=False,
    )
    banks = {"train": bank, "val": bank}  # No test bank: pilot must not access it.
    result = run_one("full", 42, banks, args, "test-protocol")
    directory = tmp_path / "full-seed42"
    assert result["test_mae"] is None
    assert (directory / "best.pt").is_file()
    assert (directory / "last.pt").is_file()
    before = (directory / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        run_one("full", 42, banks, args, "test-protocol")
    args.resume = True
    actual = run_one("full", 42, banks, args, "test-protocol")
    assert actual == json.loads((directory / "result.json").read_text())
    assert (directory / "metrics.jsonl").read_bytes() == before
    saved = torch.load(directory / "last.pt", weights_only=True)
    assert saved["study_hash"] == "test-protocol"
    assert "rng" in saved and "optimizer" in saved
    with pytest.raises(ValueError, match="completed result protocol"):
        run_one("full", 42, banks, args, "wrong-protocol")


def test_conflicting_epoch_log_is_backed_up_and_best_is_repaired(tmp_path):
    from ast_boost.experiments.train import publish_epoch_artifacts

    saved = {
        "history": [{"epoch": 1, "val_mae": 0.5}],
        "epoch": 1,
        "best_epoch": 1,
        "best": 0.5,
        "best_model": {"weight": torch.tensor([1.0])},
        "study_hash": "protocol",
    }
    original = '{"epoch": 99, "val_mae": 0.1}\n'
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(original, encoding="utf-8")
    publish_epoch_artifacts(tmp_path, saved, recovery=True)
    backups = list(tmp_path.glob("metrics-uncommitted-*.jsonl"))
    assert len(backups) == 1 and backups[0].read_text() == original
    assert json.loads(metrics.read_text()) == saved["history"][0]
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["epoch"] == 1 and best["val_mae"] == 0.5
    torch.testing.assert_close(best["model"]["weight"], saved["best_model"]["weight"])


def test_source_snapshot_is_verified_and_never_replaced(tmp_path, monkeypatch):
    import hashlib

    import ast_boost.experiments.train as training

    monkeypatch.setattr(training, "PROJECT", tmp_path)
    source = tmp_path / "module.py"
    source.write_text("value = 42\n")
    hashes = {"module.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    expected = training.source_snapshot(tmp_path, hashes)
    archive = tmp_path / "source_snapshot.zip"
    content = archive.read_bytes()
    assert training.source_snapshot(tmp_path, hashes) == expected
    source.write_text("value = 43\n")
    hashes["module.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not match"):
        training.source_snapshot(tmp_path, hashes)
    assert archive.read_bytes() == content


def test_parent_source_freeze_and_failure_record_preserve_checkpoint(tmp_path, monkeypatch):
    import hashlib

    import ast_boost.experiments.train as training

    monkeypatch.setattr(training, "PROJECT", tmp_path)
    package = tmp_path / "src/ast_boost"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("value = 1\n")
    hashes = {str(source.relative_to(tmp_path)): hashlib.sha256(source.read_bytes()).hexdigest()}
    training.verify_source_hashes(hashes)
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        training.verify_source_hashes(hashes)

    run = tmp_path / "study/control-seed7"
    checkpoint = run / "full-seed7/last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 3}, checkpoint)
    before = checkpoint.read_bytes()
    training.record_failure(
        run,
        context={"arm": "control", "seed": 7, "epochs": 5},
        error=FloatingPointError("nonfinite gradient"),
    )
    assert checkpoint.read_bytes() == before
    failure = json.loads((run.parent / "failures.json").read_text())[0]
    assert failure["last_committed_epoch"] == 3
    assert failure["last_checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert failure["exception_type"] == "FloatingPointError"
