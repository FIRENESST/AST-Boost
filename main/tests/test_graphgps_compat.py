"""Checks for the auditable GraphGPS-compatible experiment path."""

from copy import deepcopy

import pytest
import torch
from test_experiments import graphs

from ast_boost import build_laplacian
from ast_boost.experiments.data import GraphBank, build_payload
from ast_boost.experiments.graphgps import (
    GRAPHGPS_REFERENCE_COMMIT,
    GraphGPSLayer,
    GraphGPSSignNet,
)
from ast_boost.experiments.model import GPSRegressor


def test_graphgps_statistics_use_combinatorial_laplacian_and_include_zero_mode():
    graph = graphs()[1]
    payload = build_payload([graph], k=4, rw_steps=5)
    values = payload["tensors"]["graphgps_eigenvalues"][0]
    vectors = payload["tensors"]["graphgps_eigenvectors"][0, : graph.num_nodes]
    laplacian = torch.from_numpy(
        build_laplacian(graph.edge_index, n=graph.num_nodes, kind="comb")
    ).float()
    torch.testing.assert_close(laplacian @ vectors, vectors * values, atol=2e-6, rtol=2e-6)
    assert values[0].item() < 1e-7
    assert payload["metadata"]["graphgps_reference_laplacian"].startswith("combinatorial")


@pytest.mark.parametrize(
    "method",
    ["rwse_graphgps", "rwse_kernel_graphgps", "lappe_graphgps", "signnet_graphgps"],
)
def test_graphgps_reference_baselines_are_batch_independent_and_differentiable(method):
    bank = GraphBank(build_payload(graphs(), k=3, rw_steps=5))
    model = GPSRegressor(
        method,
        backbone="graphgps",
        width=32,
        heads=4,
        layers=1,
        k=3,
        rw_steps=5,
        attention_dropout=0,
    ).eval()
    together = model(bank.batch([0, 1, 2]))
    separate = torch.cat([model(bank.batch([index])) for index in range(3)])
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)
    model.train()
    loss = model(bank.batch([0, 1, 2])).square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_public_signnet_is_invariant_to_independent_graph_frequency_signs():
    bank = GraphBank(build_payload(graphs(), k=3, rw_steps=5))
    batch = bank.batch([0, 1, 2])
    model = GraphGPSSignNet(frequencies=3).eval()
    mapping = batch.node_index.new_full((batch.spectra.valid_nodes.numel(),), -1)
    mapping[batch.node_index] = torch.arange(len(batch.node_index))
    compact_edges = mapping[batch.edge_index]
    original = model(batch.graphgps_eigenvectors, compact_edges, batch.node_index)
    signs = torch.tensor(
        [[[1.0, -1.0, 1.0]], [[-1.0, 1.0, 1.0]], [[-1.0, -1.0, 1.0]]]
    )
    flipped = model(
        batch.graphgps_eigenvectors * signs,
        compact_edges,
        batch.node_index,
    )
    torch.testing.assert_close(original, flipped, atol=2e-6, rtol=2e-6)


def test_graphgps_ast_bias_reaches_native_multihead_attention():
    torch.manual_seed(18)
    bank = GraphBank(build_payload(graphs(), k=3, rw_steps=5))
    batch = bank.batch([0, 1, 2])
    model = GPSRegressor(
        "kern",
        backbone="graphgps",
        width=16,
        heads=2,
        layers=1,
        pe_dim=4,
        sign_hidden=8,
        k=3,
        rw_steps=5,
        attention_dropout=0,
    ).eval()
    without_bias = deepcopy(model)
    without_bias.pe.kernel.alpha.data.zero_()
    actual = model(batch)
    reference = without_bias(batch)
    assert not torch.allclose(actual, reference)
    actual.sum().backward()
    assert model.pe.kernel.alpha.grad is not None
    assert model.pe.kernel.alpha.grad.abs().sum() > 0
    assert isinstance(model.layers[0], GraphGPSLayer)
    assert GRAPHGPS_REFERENCE_COMMIT == "28015707cbab7f8ad72bed0ee872d068ea59c94b"


def test_kernel_only_control_preserves_rwse_initialization_and_trains_kernel():
    torch.manual_seed(51)
    rwse = GPSRegressor("rwse_graphgps", backbone="graphgps", layers=1)
    torch.manual_seed(51)
    kernel = GPSRegressor("rwse_kernel_graphgps", backbone="graphgps", layers=1)
    for name, value in rwse.state_dict().items():
        torch.testing.assert_close(value, kernel.state_dict()[name], rtol=0, atol=0)
    batch = GraphBank(build_payload(graphs())).batch([0, 1, 2])
    kernel(batch).sum().backward()
    assert kernel.kernel_only.alpha.grad.abs().sum() > 0


def test_reference_encoder_names_cannot_be_mixed_with_legacy_backbone():
    with pytest.raises(ValueError, match="require backbone"):
        GPSRegressor("signnet_graphgps")
    with pytest.raises(ValueError, match="native compact-node"):
        GPSRegressor("kern", backbone="graphgps", node_layout="padded")


def test_ast_signnet_capacity_is_an_explicit_experiment_dimension():
    model = GPSRegressor(
        "kern", backbone="graphgps", width=32, sign_hidden=24, sign_layers=5, layers=1
    )
    assert model.pe.first_order_encoder.hidden_dim == 24
    assert len(model.pe.first_order_encoder.psi.layers) == 5


def test_graphgps_signnet_keeps_fixed_frequency_width_when_batch_padding_is_trimmed():
    bank = GraphBank(build_payload(graphs(), k=8, rw_steps=5))
    batch = bank.batch([0, 1], trim_padding=True)
    assert batch.graphgps_eigenvectors.shape == (2, 5, 8)
    model = GPSRegressor(
        "signnet_graphgps",
        backbone="graphgps",
        width=32,
        layers=1,
        k=8,
        rw_steps=5,
        attention_dropout=0,
    ).eval()
    assert torch.isfinite(model(batch)).all()
