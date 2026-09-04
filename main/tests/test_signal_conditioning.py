"""Small-graph GEMM and field conditioning preserve spectral invariances."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_experiments import graphs

from ast_boost import ASTBoostPE, precompute_spectrum
from ast_boost.experiments.data import GraphBank, build_payload
from ast_boost.experiments.model import GPSRegressor
from ast_boost.spectral.signnet import SignInvariantFieldEncoder


def test_batch_padding_trim_crops_dense_adjacency_without_changing_edges():
    bank = GraphBank(build_payload(graphs()), dense_signals=True)
    reference = bank.batch([0, 1], trim_padding=False)
    trimmed = bank.batch([0, 1], trim_padding=True)
    assert trimmed.dense_adjacency.shape == (2, 5, 5)
    torch.testing.assert_close(trimmed.dense_adjacency, reference.dense_adjacency[:, :5, :5])
    reference_edges = torch.stack(
        (
            reference.edge_index[0] // 7,
            reference.edge_index[0] % 7,
            reference.edge_index[1] % 7,
        )
    )
    trimmed_edges = torch.stack(
        (
            trimmed.edge_index[0] // 5,
            trimmed.edge_index[0] % 5,
            trimmed.edge_index[1] % 5,
        )
    )
    torch.testing.assert_close(trimmed_edges, reference_edges)
    torch.testing.assert_close(trimmed.edge_types, reference.edge_types)


@pytest.mark.parametrize("field_count", [0, 1, 4])
def test_dense_signal_matches_sparse_values_and_gradients_with_multiedges(field_count):
    torch.manual_seed(11)
    items = graphs()
    # Directed, duplicated edges expose orientation and multiplicity mistakes.
    items[0].edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]])
    items[0].edge_attr = torch.ones(3, dtype=torch.long)
    batch = GraphBank(build_payload(items), dense_signals=True).batch([2, 0, 1])
    assert batch.dense_adjacency[1, 1, 0] == 2
    assert batch.dense_adjacency[1, 0, 1] == 0
    fields = torch.randn(3, field_count, 7, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(3, field_count, dtype=torch.bool)
    mask[1] = False
    sparse = SignInvariantFieldEncoder(hidden_dim=8, out_dim=4, layers=2).double()
    dense = deepcopy(sparse)
    expected = sparse.forward_padded(fields, batch.edge_index, mask)
    actual = dense.forward_padded(fields, batch.edge_index, mask, adjacency=batch.dense_adjacency)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    if field_count:
        expected.square().sum().backward(retain_graph=True)
        expected_grad = fields.grad.clone()
        fields.grad = None
        actual.square().sum().backward()
        torch.testing.assert_close(fields.grad, expected_grad, atol=1e-10, rtol=1e-9)
        for p, q in zip(sparse.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-8)
        assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("dense", [False, True])
def test_size_conditioned_full_matches_single_graph_api_and_keeps_kernel(dense):
    torch.manual_seed(12)
    items = graphs()
    batch = GraphBank(build_payload(items), dense_signals=True).batch([0, 1, 2])
    model = ASTBoostPE(
        variant="full", field_scaling="size", heads=2, pe_dim=4, sign_hidden=8, token_dim=3
    )
    x = torch.randn(3, 7, 3)
    actual, bias = model.forward_packed(
        x, batch.edge_index, batch.spectra, dense_adjacency=batch.dense_adjacency if dense else None
    )
    for i, g in enumerate(items):
        expected, kernel = model(
            x[i, : g.num_nodes], precompute_spectrum(g.edge_index, n=g.num_nodes, k=8), g.edge_index
        )
        torch.testing.assert_close(actual[i, : g.num_nodes], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            bias[i, :, : g.num_nodes, : g.num_nodes], kernel, atol=1e-5, rtol=1e-5
        )
    raw = deepcopy(model)
    raw.field_scaling = "none"
    _, raw_bias = raw.forward_packed(x, batch.edge_index, batch.spectra)
    torch.testing.assert_close(bias, raw_bias, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["sparse", "dense"])
def test_conditioned_dense_full_sign_and_node_permutation_invariance(backend):
    torch.manual_seed(93)
    batch = GraphBank(build_payload(graphs()), dense_signals=True).batch([0, 1, 2])
    model = GPSRegressor(
        "full",
        width=16,
        layers=2,
        pe_dim=4,
        sign_hidden=8,
        field_scaling="size",
        signal_backend=backend,
    ).eval()
    expected = model(batch)
    spectra = replace(
        batch.spectra,
        first_order=-batch.spectra.first_order,
        second_order=-batch.spectra.second_order,
    )
    torch.testing.assert_close(model(replace(batch, spectra=spectra)), expected)
    permutations = torch.stack(
        [torch.cat((torch.randperm(n), torch.arange(n, 7))) for n in batch.spectra.node_counts]
    )
    inverse = torch.empty(21, dtype=torch.long)
    inverse[(permutations + torch.arange(3)[:, None] * 7).flatten()] = torch.arange(21)
    permuted_spectrum = replace(
        batch.spectra,
        eigenvectors=batch.spectra.eigenvectors.gather(
            1, permutations[..., None].expand_as(batch.spectra.eigenvectors)
        ),
        first_order=batch.spectra.first_order.gather(
            2, permutations[:, None].expand_as(batch.spectra.first_order)
        ),
        second_order=batch.spectra.second_order.gather(
            2, permutations[:, None].expand_as(batch.spectra.second_order)
        ),
    )
    adjacency = batch.dense_adjacency.gather(1, permutations[..., None].expand(3, 7, 7))
    adjacency = adjacency.gather(2, permutations[:, None].expand(3, 7, 7))
    permuted = replace(
        batch,
        node_types=batch.node_types.gather(1, permutations),
        edge_index=inverse[batch.edge_index],
        spectra=permuted_spectrum,
        dense_adjacency=adjacency,
    )
    torch.testing.assert_close(model(permuted), expected, rtol=1e-5, atol=1e-5)
    second = batch.spectra.second_order.clone().requires_grad_(True)
    derivative = torch.autograd.grad(
        model(replace(batch, spectra=replace(batch.spectra, second_order=second))).sum(), second
    )[0]
    assert derivative.abs().sum() > 0 and torch.isfinite(derivative).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_size_full_cuda_bf16_optimizer_updates_second_order():
    torch.manual_seed(44)
    batch = GraphBank(build_payload(graphs()), "cuda", dense_signals=True).batch([0, 1, 2])
    model = GPSRegressor(
        "full",
        width=16,
        layers=2,
        pe_dim=4,
        sign_hidden=8,
        field_scaling="size",
        signal_backend="dense",
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.pe.second_order_encoder.rho[0].weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = (model(batch).float() - batch.targets).abs().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before, model.pe.second_order_encoder.rho[0].weight)


@pytest.mark.parametrize("backend", ["sparse", "dense"])
def test_fused_shared_fields_matches_split_values_and_parameter_gradients(backend):
    torch.manual_seed(2026)
    batch = GraphBank(build_payload(graphs()), dense_signals=True).batch([2, 0, 1])
    memory_saving = GPSRegressor(
        "full",
        width=16,
        layers=2,
        heads=2,
        pe_dim=4,
        sign_hidden=8,
        field_scaling="size",
        signal_backend=backend,
    ).double()
    memory_saving.pe.first_order_encoder.psi.fuse_input_aggregation = True
    fused = deepcopy(memory_saving)
    fused.pe.first_order_encoder.psi.fuse_input_aggregation = False
    bmm_reduction = deepcopy(fused)
    bmm_reduction.pe.first_order_encoder.bmm_field_reduction = True
    bmm_reduction.pe.second_order_encoder.bmm_field_reduction = True
    split = deepcopy(fused)
    split.pe.fuse_shared_fields = False
    memory_saving.eval()
    fused.eval()
    bmm_reduction.eval()
    split.eval()
    expected = split(batch)
    actual = fused(batch)
    bmm_actual = bmm_reduction(batch)
    memory_saving_actual = memory_saving(batch)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(bmm_actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(memory_saving_actual, expected, rtol=1e-10, atol=1e-10)
    expected.square().sum().backward()
    actual.square().sum().backward()
    bmm_actual.square().sum().backward()
    memory_saving_actual.square().sum().backward()
    for (fused_name, fused_parameter), (split_name, split_parameter) in zip(
        fused.named_parameters(), split.named_parameters(), strict=True
    ):
        assert fused_name == split_name
        torch.testing.assert_close(fused_parameter.grad, split_parameter.grad, rtol=1e-9, atol=1e-9)
    for memory_parameter, split_parameter in zip(
        memory_saving.parameters(), split.parameters(), strict=True
    ):
        torch.testing.assert_close(
            memory_parameter.grad, split_parameter.grad, rtol=1e-9, atol=1e-9
        )
    for bmm_parameter, split_parameter in zip(
        bmm_reduction.parameters(), split.parameters(), strict=True
    ):
        torch.testing.assert_close(bmm_parameter.grad, split_parameter.grad, rtol=1e-9, atol=1e-9)
