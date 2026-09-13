"""Sampling and crash recovery checks for the Peptides runner."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ast_boost.experiments.sampling import batches


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "lrgb_runner", Path(__file__).resolve().parents[1] / "benchmarks/experiment_lrgb_matrix.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


@pytest.mark.parametrize("sampler", ["random", "sortish"])
def test_sampling_visits_every_graph_once_and_changes_between_epochs(sampler):
    counts = list(range(101))
    first = batches(range(101), counts, 7, seed=42, sampler=sampler)
    assert first == batches(range(101), counts, 7, seed=42, sampler=sampler)
    assert sorted(index for batch in first for index in batch) == list(range(101))
    assert sorted(map(len, first)) == [3] + [7] * 14
    second = batches(range(101), counts, 7, seed=43, sampler=sampler)
    assert {frozenset(batch) for batch in first} != {frozenset(batch) for batch in second}


def test_random_sampling_preserves_uniform_permutation_without_size_sorting():
    actual = batches(range(101), list(range(101)), 7, seed=42)
    assert [index for batch in actual for index in batch] == (
        np.random.default_rng(42).permutation(101).tolist()
    )
    assert batches([0, 1, 2], [9, 3, 5], 2) == [[1, 2], [0]]


class ToyBank:
    counts = [2, 3, 5, 7, 11]
    device = "cpu"

    def __len__(self):
        return len(self.counts)

    def batch(self, indices):
        x = torch.tensor(indices, dtype=torch.float32)[:, None] / 5
        return SimpleNamespace(x=x, targets=x * 0.25 + 0.1)


class ToyModel(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(1, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))

    def forward(self, batch):
        return super().forward(batch.x)

    @property
    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())


@pytest.mark.parametrize("damage", ["none", "result", "missing", "history", "weights"])
def test_completed_resume_validates_evidence(tmp_path, monkeypatch, damage):
    runner = load_runner()
    monkeypatch.setattr(runner, "make_model", lambda method, args: ToyModel())
    args = SimpleNamespace(
        output=tmp_path, dataset="Peptides-struct", resume=False,
        device="cpu", precision="fp32", epochs=3, warmup_epochs=1,
        batch_size=2, sampler="random",
    )
    banks = {"train": ToyBank(), "val": ToyBank()}
    expected = runner.run_one("kern", 42, banks, args, "protocol")
    directory = tmp_path / "kern-seed42"
    if damage == "result":
        altered = dict(expected, best_val_metric=99.0)
        (directory / "result.json").write_text(json.dumps(altered))
    elif damage == "missing":
        (directory / "best.pt").unlink()
    elif damage == "history":
        (directory / "metrics.jsonl").write_text("{}\n")
    elif damage == "weights":
        state = torch.load(directory / "best.pt", weights_only=True)
        next(iter(state["model"].values())).add_(1)
        torch.save(state, directory / "best.pt")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    args.resume = True
    if damage == "none":
        assert runner.run_one("kern", 42, banks, args, "protocol") == expected
    else:
        with pytest.raises(ValueError, match="completed"):
            runner.run_one("kern", 42, banks, args, "protocol")
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("phase", ["before_last", "after_last", "before_best", "after_best"])
def test_lrgb_resume_restores_training_and_repairs_publications(tmp_path, monkeypatch, phase):
    runner = load_runner()
    monkeypatch.setattr(runner, "make_model", lambda method, args: ToyModel())
    args = SimpleNamespace(
        output=tmp_path / "continuous", dataset="Peptides-struct", resume=False,
        device="cpu", precision="fp32", epochs=3, warmup_epochs=1, batch_size=2,
        sampler="random",
    )
    banks = {"train": ToyBank(), "val": ToyBank()}
    expected = runner.run_one("kern", 42, banks, args, "protocol")
    reference = torch.load(args.output / "kern-seed42/last.pt", weights_only=True)
    original_save = runner.atomic_checkpoint

    def interrupted_save(path, state):
        selected = path.name == "best.pt" if "best" in phase else (
            path.name == "last.pt" and state["epoch"] == 2
        )
        if selected and phase.startswith("before"):
            raise RuntimeError("simulated interruption")
        original_save(path, state)
        if selected and phase.startswith("after"):
            raise RuntimeError("simulated interruption")

    args.output = tmp_path / "interrupted"
    monkeypatch.setattr(runner, "atomic_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run_one("kern", 42, banks, args, "protocol")
    monkeypatch.setattr(runner, "atomic_checkpoint", original_save)
    args.resume = True
    actual = runner.run_one("kern", 42, banks, args, "protocol")
    directory = args.output / "kern-seed42"
    recovered = torch.load(directory / "last.pt", weights_only=True)
    assert actual["best_val_metric"] == expected["best_val_metric"]
    for section in ("model", "best_model"):
        for key, tensor in reference[section].items():
            torch.testing.assert_close(tensor, recovered[section][key], rtol=0, atol=0)
    torch.testing.assert_close(reference["torch_rng"], recovered["torch_rng"], rtol=0, atol=0)
    assert reference["scheduler"] == recovered["scheduler"]
    assert reference["numpy_rng"] == recovered["numpy_rng"]
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, recovered["optimizer"]["state"][key][name])
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    for left, right in zip(rows, reference["history"], strict=True):
        for key in ("train_loss", "val_mae", "best_val_mae", "lr"):
            assert left[key] == right[key]
    published = torch.load(directory / "best.pt", weights_only=True)
    assert published["epoch"] == recovered["best_epoch"]
    for key, tensor in recovered["best_model"].items():
        torch.testing.assert_close(tensor, published["model"][key], rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["dataset_sha256", "target_sha256", "spectral_cache_sha256"])
def test_lrgb_default_matrix_resumes_and_rejects_changed_data(tmp_path, monkeypatch, changed):
    runner = load_runner()
    metadata = {
        "dataset_sha256": "topology", "target_sha256": "labels",
        "spectral_cache_sha256": "spectrum",
    }
    banks = {name: SimpleNamespace(metadata=metadata) for name in ("train", "val")}
    monkeypatch.setattr(runner, "load_lrgb_banks", lambda *args, **kwargs: banks)
    argv = [
        "--dataset", "Peptides-struct", "--output", str(tmp_path),
        "--device", "cpu", "--prepare-only",
    ]
    runner.main(argv)
    original = (tmp_path / "dataset.json").read_bytes()
    runner.main([*argv, "--resume"])
    assert (tmp_path / "dataset.json").read_bytes() == original
    metadata[changed] = "changed"
    with pytest.raises(ValueError, match="dataset targets, topology or cached spectra changed"):
        runner.main([*argv, "--resume"])
    assert (tmp_path / "dataset.json").read_bytes() == original
