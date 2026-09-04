"""Compact layout must change execution, not Full's learned graph function."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_experiments import graphs

from ast_boost.experiments.data import GraphBank, build_payload
from ast_boost.experiments.model import METHODS, GPSRegressor
from ast_boost.experiments.train import seed_all


@pytest.mark.parametrize("ids", [[2, 0, 2], [0], [1, 2, 0]])
def test_static_bank_layout_matches_boolean_reference(ids):
    bank = GraphBank(build_payload(graphs()))
    actual = bank.batch(ids)
    reference = bank.batch(ids, static_layout=False)
    for key in ("node_types", "edge_index", "edge_types", "targets", "rwse", "lappe"):
        torch.testing.assert_close(getattr(actual, key), getattr(reference, key), rtol=0, atol=0)
    torch.testing.assert_close(
        actual.node_index, actual.spectra.valid_nodes.reshape(-1).nonzero().reshape(-1)
    )
    assert reference.node_index is None


def test_static_layout_supports_edgeless_graphs_and_checks_counts():
    graph = graphs()[0]
    graph.edge_index = torch.empty(2, 0, dtype=torch.long)
    graph.edge_attr = torch.empty(0, dtype=torch.long)
    payload = build_payload([graph])
    batch = GraphBank(payload).batch([0, 0])
    assert batch.edge_index.shape == (2, 0)
    assert batch.edge_types.shape == (0,)
    assert batch.node_index.numel() == 6
    payload["node_counts"][0] += 1
    with pytest.raises(ValueError, match="counts disagree"):
        GraphBank(payload)


@pytest.mark.parametrize("method", METHODS)
def test_batch_local_padding_trim_preserves_predictions(method):
    seed_all(27)
    payload = build_payload(graphs())
    payload["tensors"] = {
        key: value.double() if value.is_floating_point() else value
        for key, value in payload["tensors"].items()
    }
    bank = GraphBank(payload)
    trimmed = bank.batch([0, 1], trim_padding=True)
    reference = bank.batch([0, 1], trim_padding=False)
    assert trimmed.spectra.max_nodes == 5
    assert reference.spectra.max_nodes == 7
    model = (
        GPSRegressor(
            method,
            width=16,
            heads=2,
            layers=2,
            pe_dim=4,
            sign_hidden=8,
            attention_dropout=0,
        )
        .double()
        .eval()
    )
    torch.testing.assert_close(model(trimmed), model(reference), atol=1e-10, rtol=1e-9)


def test_batch_local_padding_trim_preserves_full_gradients():
    seed_all(28)
    payload = build_payload(graphs())
    payload["tensors"] = {
        key: value.double() if value.is_floating_point() else value
        for key, value in payload["tensors"].items()
    }
    bank = GraphBank(payload)
    trimmed = bank.batch([0, 1], trim_padding=True)
    reference = bank.batch([0, 1], trim_padding=False)
    actual = (
        GPSRegressor(
            "full", width=16, heads=2, layers=2, pe_dim=4, sign_hidden=8, attention_dropout=0
        )
        .double()
        .eval()
    )
    expected = deepcopy(actual)
    model_output = actual(trimmed)
    reference_output = expected(reference)
    model_output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(model_output, reference_output, atol=1e-10, rtol=1e-9)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        actual.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == reference_name
        if reference_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, atol=1e-9, rtol=1e-8, msg=name
            )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("training", [False, True])
def test_compact_matches_padded_values_parameter_gradients_and_bn_state(method, training):
    seed_all(91)
    payload = build_payload(graphs())
    payload["tensors"] = {
        key: value.double() if value.is_floating_point() else value
        for key, value in payload["tensors"].items()
    }
    batch = GraphBank(payload).batch([2, 0, 1, 0])
    compact = GPSRegressor(
        method, width=16, heads=2, layers=2, pe_dim=4, sign_hidden=8, attention_dropout=0
    ).double()
    padded = deepcopy(compact)
    padded.node_layout = "padded"
    compact.train(training)
    padded.train(training)
    # LapPE signs are paired independently of each implementation's execution.
    seed_all(71)
    actual = compact(batch)
    seed_all(71)
    expected = padded(batch)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    (actual - batch.targets).square().mean().backward()
    (expected - batch.targets).square().mean().backward()
    for (key, p), (other_key, reference) in zip(
        compact.named_parameters(), padded.named_parameters(), strict=True
    ):
        assert key == other_key
        if reference.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, reference.grad, atol=1e-9, rtol=1e-8, msg=key)
    for key, value in compact.state_dict().items():
        torch.testing.assert_close(value, padded.state_dict()[key], atol=1e-10, rtol=1e-9, msg=key)


def test_compact_fallback_accepts_batches_without_cached_node_indices():
    bank = GraphBank(build_payload(graphs()))
    batch = bank.batch([2, 0])
    model = GPSRegressor("full", width=16, layers=1, pe_dim=4, sign_hidden=8).eval()
    torch.testing.assert_close(model(batch), model(replace(batch, node_index=None)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_compact_cuda_values_and_second_order_gradients(precision):
    seed_all(42)
    batch = GraphBank(build_payload(graphs()), "cuda").batch([2, 0, 1])
    compact = GPSRegressor("full", width=16, layers=2, pe_dim=4, sign_hidden=8).cuda().eval()
    padded = deepcopy(compact)
    padded.node_layout = "padded"
    with torch.autocast("cuda", dtype=precision, enabled=precision != torch.float32):
        actual, expected = compact(batch), padded(batch)
    tolerance = 0.02 if precision == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    actual.float().square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in compact.parameters())
    assert compact.pe.second_order_encoder.rho[0].weight.grad.abs().sum() > 0


def test_nondefault_frequency_and_pair_budgets_reach_model():
    batch = GraphBank(build_payload(graphs(), k=3, pairs=2, rw_steps=5)).batch([0, 1, 2])
    for method in ("rwse", "lappe", "full"):
        model = GPSRegressor(
            method, width=16, layers=1, pe_dim=4, sign_hidden=8, k=3, pairs=2, rw_steps=5
        ).eval()
        assert torch.isfinite(model(batch)).all()
