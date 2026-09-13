"""Frequency identity, complete functional calculus, and diagonal-path regressions."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ast_boost import ASTBoostPE, precompute_spectrum, prepare_spectrum, prepare_spectrum_batch
from ast_boost.experiments.data import GraphBank, build_payload
from ast_boost.spectral.kernel import SpectralKernelBias, bernstein_basis
from ast_boost.spectral.signnet import SignInvariantFieldEncoder


def graph(n, cycle=False):
    source = torch.arange(n if cycle else n - 1)
    target = (source + 1) % n
    edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    return SimpleNamespace(
        num_nodes=n,
        edge_index=edges,
        x=torch.arange(n)[:, None] % 8,
        edge_attr=torch.ones(edges.shape[1], dtype=torch.long),
        y=torch.tensor([0.2]),
    )


def spectra(g, k=3):
    low = precompute_spectrum(g.edge_index, n=g.num_nodes, k=k)
    full = precompute_spectrum(g.edge_index, n=g.num_nodes, k=g.num_nodes, skip_zero=False)
    return low, full


def test_complete_kernel_matches_matrix_function_values_and_parameter_gradients():
    g = graph(7)
    low, full = spectra(g)
    prepared = prepare_spectrum(low, kernel_spectrum=full, dtype=torch.float64)
    batch = prepare_spectrum_batch([prepared], dtype=torch.float64)
    kernel = SpectralKernelBias(heads=2, degree=3).double()
    with torch.no_grad():
        kernel.coefficients.copy_(torch.tensor([[0.8, -0.2, 0.3, 0.7], [-0.1, 0.4, 0.9, 0.2]]))
    actual = kernel.raw_kernel_padded(
        batch.kernel_eigenvalues,
        batch.kernel_eigenvectors,
        batch.kernel_frequency_mask,
        complete=True,
    )
    expected = kernel.raw_kernel(full).unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * probe).sum(), kernel.coefficients)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), kernel.coefficients)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
    # Independent NumPy functional calculus, including the zero mode.
    response = bernstein_basis(full.eigenvalues, degree=3) @ kernel.coefficients[0].detach().numpy()
    reference = (full.eigenvectors * response) @ full.eigenvectors.T
    np.testing.assert_allclose(actual[0, 0].detach(), reference, atol=1e-12)
    assert not torch.allclose(actual[0], kernel.raw_kernel(low))


def test_constant_full_kernel_has_exact_identity_and_zero_attention_bias():
    items = [graph(5), graph(6, cycle=True)]
    bank = GraphBank(build_payload(items, k=2, kernel_spectrum="all"))
    batch = bank.batch([0, 1])
    model = ASTBoostPE(variant="full", heads=2, pe_dim=4, sign_hidden=8, kernel_spectrum="all")
    raw = model._raw_kernel_padded(batch.spectra)
    expected = torch.diag_embed(batch.spectra.valid_nodes.float())[:, None].expand_as(raw)
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    bias = model.kernel.bias_from_raw_padded(raw, batch.spectra.valid_nodes)
    assert torch.count_nonzero(bias) == 0


def test_labels_distinguish_frequency_assignments_without_breaking_set_invariance():
    torch.manual_seed(7)
    encoder = SignInvariantFieldEncoder(hidden_dim=8, out_dim=4).double()
    encoder.enable_labels("eigenvalue")
    encoder.double()
    hidden = torch.randn(3, 5, 8, dtype=torch.float64)
    labels = torch.tensor([[0.1, 0.1], [0.5, 0.5], [1.7, 1.7]], dtype=torch.float64)
    order = torch.tensor([2, 0, 1])
    expected = encoder.readout(hidden, labels)
    torch.testing.assert_close(encoder.readout(hidden[order], labels[order]), expected)
    assert not torch.allclose(encoder.readout(hidden, labels[order]), expected)
    blind = deepcopy(encoder)
    blind.label_mode = "blind"
    torch.testing.assert_close(blind.readout(hidden, labels[order]), blind.readout(hidden, labels))
    assert sum(p.numel() for p in blind.parameters()) == sum(
        p.numel() for p in encoder.parameters()
    )


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_new_paths_match_single_graph_and_trimmed_batch_with_gradients(dense, fused):
    torch.manual_seed(13)
    items = [graph(5), graph(6, cycle=True), graph(1)]
    bank = GraphBank(build_payload(items, k=3, kernel_spectrum="all"), dense_signals=True)
    batch = bank.batch([0, 1, 2])
    model = ASTBoostPE(
        variant="full",
        heads=2,
        pe_dim=4,
        sign_hidden=8,
        token_dim=3,
        frequency_labels="eigenvalue",
        kernel_spectrum="all",
        kernel_diagonal=True,
        field_scaling="size",
        fuse_shared_fields=fused,
    )
    with torch.no_grad():
        model.kernel.coefficients.normal_()
        model.diagonal_projection.weight.normal_()
    x = torch.randn(3, 6, 3)
    actual, bias = model.forward_packed(
        x,
        batch.edge_index,
        batch.spectra,
        dense_adjacency=batch.dense_adjacency if dense else None,
    )
    for i, g in enumerate(items):
        low, full = spectra(g)
        single = prepare_spectrum(low, kernel_spectrum=full)
        expected, expected_bias = model(x[i, : g.num_nodes], single, g.edge_index)
        torch.testing.assert_close(actual[i, : g.num_nodes], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            bias[i, :, : g.num_nodes, : g.num_nodes], expected_bias, atol=2e-5, rtol=2e-5
        )
    assert torch.count_nonzero(actual[~batch.spectra.valid_nodes]) == 0
    trimmed = bank.batch([0], trim_padding=True)
    trimmed_output, _ = model.forward_packed(x[:1, :5], trimmed.edge_index, trimmed.spectra)
    torch.testing.assert_close(trimmed_output[0], actual[0, :5], atol=2e-5, rtol=2e-5)
    actual.square().sum().backward()
    for parameter in (
        model.diagonal_projection.weight,
        model.kernel.coefficients,
        model.second_order_encoder.field_mixer[0].weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["lite", "kern", "full"])
def test_new_features_keep_sign_basis_and_node_permutation_symmetries(variant):
    torch.manual_seed(23)
    g = graph(6, cycle=True)
    low, full = spectra(g)
    model = ASTBoostPE(
        variant=variant,
        heads=2,
        pe_dim=4,
        sign_hidden=8,
        frequency_labels="none" if variant == "lite" else "eigenvalue",
        kernel_spectrum="all",
        kernel_diagonal=True,
    ).double()
    with torch.no_grad():
        model.kernel.coefficients.normal_()
        model.diagonal_projection.weight.normal_()
    x = torch.randn(6, 3, dtype=torch.float64)
    prepared = prepare_spectrum(low, kernel_spectrum=full, dtype=torch.float64)
    expected, bias = model(x, prepared, g.edge_index)
    rng = np.random.default_rng(23)

    def rotated(s):
        vectors = s.eigenvectors.copy()
        # C6 has exact repeated eigenspaces, so rotations preserve the operator.
        for block in range(s.num_blocks):
            ids = s.block_indices(block)
            q, _ = np.linalg.qr(rng.normal(size=(len(ids), len(ids))))
            vectors[:, ids] = vectors[:, ids] @ q
        return replace(s, eigenvectors=vectors)

    rotated_low, rotated_full = rotated(low), rotated(full)
    changed = prepare_spectrum(rotated_low, kernel_spectrum=rotated_full, dtype=torch.float64)
    actual, changed_bias = model(x, changed, g.edge_index)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(changed_bias, bias, atol=1e-10, rtol=1e-10)
    order = torch.randperm(6)
    inverse = torch.argsort(order)
    changed = prepare_spectrum(
        replace(rotated_low, eigenvectors=rotated_low.eigenvectors[order]),
        kernel_spectrum=replace(rotated_full, eigenvectors=rotated_full.eigenvectors[order]),
        dtype=torch.float64,
    )
    actual, changed_bias = model(x[order], changed, inverse[g.edge_index])
    torch.testing.assert_close(actual, expected[order], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(changed_bias, bias[:, order][:, :, order], atol=1e-10, rtol=1e-10)


def test_full_kernel_budget_leaves_absolute_fields_and_pairs_unchanged():
    items = [graph(9), graph(6, cycle=True)]
    old = build_payload(items, k=3)
    new = build_payload(items, k=3, kernel_spectrum="all")
    for key, value in old["tensors"].items():
        torch.testing.assert_close(new["tensors"][key], value, rtol=0, atol=0)
    assert new["tensors"]["kernel_frequency_mask"].sum(1).tolist() == [9, 6]
    assert new["metadata"]["dataset_sha256"] == old["metadata"]["dataset_sha256"]


def test_label_readout_empty_graphs_remain_zero_and_padding_does_not_backpropagate():
    encoder = SignInvariantFieldEncoder(hidden_dim=8, out_dim=4)
    encoder.enable_labels("eigenvalue")
    hidden = torch.randn(3, 2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    labels = torch.randn(2, 3, 2)
    actual = encoder.readout_padded(hidden, mask, labels)
    assert torch.count_nonzero(actual[1]) == 0
    actual.sum().backward()
    assert torch.count_nonzero(hidden.grad[:, 1]) == 0
    assert torch.count_nonzero(hidden.grad[1:, 0]) == 0
    assert encoder.readout_padded(hidden[:0], mask[:, :0], labels[:, :0]).count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_complete_labeled_diagonal_full_trains_in_cuda_bf16():
    from ast_boost.experiments.model import GPSRegressor

    torch.manual_seed(44)
    bank = GraphBank(build_payload([graph(7), graph(6, cycle=True)], kernel_spectrum="all"), "cuda")
    batch = bank.batch([0, 1])
    model = GPSRegressor(
        "full",
        width=16,
        layers=2,
        pe_dim=4,
        sign_hidden=8,
        field_scaling="size",
        frequency_labels="eigenvalue",
        kernel_spectrum="all",
        kernel_diagonal=True,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, fused=True)
    before = model.pe.diagonal_projection.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(batch).float()
            loss = (prediction - batch.targets).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(loss)
    assert not torch.equal(before, model.pe.diagonal_projection.weight)
    assert model.pe.second_order_encoder.field_mixer[0].weight.grad.abs().sum() > 0


def test_optional_modules_preserve_existing_parameter_initialization_and_default_function():
    options = {"variant": "full", "heads": 2, "pe_dim": 4, "sign_hidden": 8, "token_dim": 3}
    torch.manual_seed(71)
    baseline = ASTBoostPE(**options)
    torch.manual_seed(71)
    changed = ASTBoostPE(**options, frequency_labels="eigenvalue", kernel_diagonal=True)
    for key, tensor in baseline.state_dict().items():
        torch.testing.assert_close(changed.state_dict()[key], tensor, rtol=0, atol=0)
    torch.manual_seed(71)
    diagonal = ASTBoostPE(**options, kernel_diagonal=True)
    g = graph(7)
    low, _ = spectra(g)
    x = torch.randn(7, 3)
    actual, bias = diagonal(x, low, g.edge_index)
    expected, original_bias = baseline(x, low, g.edge_index)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(bias, original_bias, rtol=0, atol=0)


def test_complete_kernel_requires_an_explicit_untruncated_spectrum():
    low, full = spectra(graph(7))
    with pytest.raises(ValueError, match="complete spectrum"):
        prepare_spectrum(low, kernel_spectrum=low)
    model = ASTBoostPE(variant="kern", kernel_spectrum="all")
    with pytest.raises(ValueError, match="complete kernel cache"):
        model(torch.randn(7, 3), low, graph(7).edge_index)
    prepared = prepare_spectrum(low, kernel_spectrum=full, dtype=torch.float64)
    single = prepared.to("cpu", dtype=torch.float32)
    assert single.first_order_labels.dtype == torch.float32
    assert single.kernel_eigenvectors.dtype == torch.float32
    assert single.kernel_frequency_mask.dtype == torch.bool
    with pytest.raises(ValueError, match="all graphs"):
        prepare_spectrum_batch([single, low])
