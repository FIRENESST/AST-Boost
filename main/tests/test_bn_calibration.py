"""The optional offline calibration may change buffers, never weights or labels."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ast_boost.experiments.model import MaskedBatchNorm


def load_calibrator():
    path = Path(__file__).resolve().parents[1] / "benchmarks/calibrate_batchnorm.py"
    spec = importlib.util.spec_from_file_location("bn_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recalibrate


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = MaskedBatchNorm(2)
        self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        self.dropout = torch.nn.Dropout(0.9)

    def forward(self, batch):
        return self.bn.forward_compact(self.dropout(batch.x) * self.weight)


class Bank:
    device = "cpu"

    def __init__(self, x, fail=False):
        self.x, self.fail, self.calls = x, fail, 0

    def batch(self, indices):
        self.calls += 1
        if self.fail and self.calls == 2:
            raise RuntimeError("injected data error")
        return SimpleNamespace(x=self.x[list(indices)])  # No target/label exists.


def test_calibration_pools_between_batch_variance_and_restores_modes():
    model = ToyModel().train()
    model.bn.eval()  # Deliberately mixed initial modes must be restored exactly.
    weights = model.weight.detach().clone()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0], [8.0, 9.0]])
    args = SimpleNamespace(device="cpu", precision="fp32", batch_size=2)
    elapsed = load_calibrator()(model, Bank(x), list(range(len(x))), args)
    expected = x * weights
    torch.testing.assert_close(model.bn.running_mean, expected.mean(0))
    torch.testing.assert_close(model.bn.running_var, expected.var(0, correction=1))
    torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)
    assert model.training and model.dropout.training and not model.bn.training
    assert "forward_compact" not in model.bn.__dict__
    assert model.weight.grad is None and elapsed >= 0


def test_failed_calibration_rolls_back_all_buffers_and_methods():
    model = ToyModel().eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    args = SimpleNamespace(device="cpu", precision="fp32", batch_size=2)
    with pytest.raises(RuntimeError, match="injected"):
        load_calibrator()(model, Bank(torch.randn(4, 2), fail=True), list(range(4)), args)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not model.training and not model.bn.training
    assert "forward_compact" not in model.bn.__dict__
