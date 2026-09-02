"""Runtime integration coverage for caching, sparse solving, and GPS batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ast_boost.cli import main as precompute_main
from ast_boost.spectral.gps_adapter import ASTBoostPE, add_attention_bias
from ast_boost.spectral.kernel import SpectralKernelBias
from ast_boost.spectral.precompute import (
    build_laplacian,
    build_sparse_laplacian,
    load_spectrum,
    precompute_spectrum,
)
from ast_boost.spectral.torch_spectrum import (
    prepare_spectra,
    prepare_spectrum,
    prepare_spectrum_batch,
)


def _path_edge_index(n: int, *, offset: int = 0) -> np.ndarray:
    source = np.arange(n - 1, dtype=np.int64) + offset
    target = source + 1
    return np.stack((np.concatenate((source, target)), np.concatenate((target, source))))


def test_sparse_precompute_matches_dense_solver() -> None:
    edge_index = _path_edge_index(40)
    dense_laplacian = build_laplacian(edge_index, n=40, kind="sym")
    sparse_laplacian = build_sparse_laplacian(edge_index, n=40, kind="sym")
    np.testing.assert_allclose(sparse_laplacian.toarray(), dense_laplacian, atol=1e-12)

    dense = precompute_spectrum(edge_index, n=40, k=4, dense_threshold=256)
    sparse = precompute_spectrum(edge_index, n=40, k=4, dense_threshold=8)
    np.testing.assert_allclose(sparse.eigenvalues, dense.eigenvalues, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        sparse.eigenvectors @ sparse.eigenvectors.T,
        dense.eigenvectors @ dense.eigenvectors.T,
        rtol=1e-6,
        atol=1e-7,
    )


def test_npz_cli_roundtrip(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.npz"
    spectrum_path = tmp_path / "spectrum.npz"
    np.savez(graph_path, edge_index=_path_edge_index(6), num_nodes=np.asarray(6))

    exit_code = precompute_main([str(graph_path), str(spectrum_path), "--k", "3"])

    assert exit_code == 0
    loaded = load_spectrum(spectrum_path)
    assert loaded.num_nodes == 6
    assert loaded.k == 3
    assert loaded.laplacian == "sym"
    np.testing.assert_allclose(loaded.eigenvectors.T @ loaded.eigenvectors, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("variant", ["lite", "kern", "full"])
def test_ast_boost_batch_adapter_shapes_mask_and_dtype(variant: str) -> None:
    first_edges = _path_edge_index(3)
    second_edges = _path_edge_index(4, offset=3)
    edge_index = torch.as_tensor(
        np.concatenate((first_edges, second_edges), axis=1), dtype=torch.long
    )
    spectra = [
        precompute_spectrum(first_edges, n=3, k=2),
        precompute_spectrum(_path_edge_index(4), n=4, k=3),
    ]
    prepared_spectra = prepare_spectra(spectra, k0_pairs=3)
    prepared_batch = prepare_spectrum_batch(prepared_spectra, k0_pairs=3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    tokens = torch.arange(28, dtype=torch.float64).reshape(7, 4)
    model = ASTBoostPE(
        variant=variant,
        heads=2,
        pe_dim=3,
        sign_hidden=8,
        sign_layers=1,
        k0_pairs=3,
        kernel_degree=2,
        token_dim=4,
    ).eval()

    with torch.no_grad():
        encoded, bias, same_graph = model.forward_batch(tokens, edge_index, spectra, batch)
        padded_encoded, padded_bias, padded_mask, valid_nodes = model.forward_padded_batch(
            tokens, edge_index, prepared_batch, batch, contiguous=True
        )

    assert encoded.shape == (7, 4)
    assert encoded.dtype == next(model.parameters()).dtype
    assert bias.shape == (2, 7, 7)
    assert same_graph.shape == (7, 7)
    assert same_graph[:3, :3].all() and same_graph[3:, 3:].all()
    assert not same_graph[:3, 3:].any() and not same_graph[3:, :3].any()
    torch.testing.assert_close(bias[:, :3, 3:], torch.zeros_like(bias[:, :3, 3:]))

    logits = torch.zeros_like(bias)
    masked_logits = add_attention_bias(logits, bias, attention_mask=same_graph)
    assert torch.isneginf(masked_logits[:, :3, 3:]).all()
    assert torch.isfinite(masked_logits[:, :3, :3]).all()
    torch.testing.assert_close(padded_encoded, encoded)
    assert padded_bias.shape == (2, 2, 4, 4)
    torch.testing.assert_close(padded_bias[0, :, :3, :3], bias[:, :3, :3])
    torch.testing.assert_close(padded_bias[1], bias[:, 3:, 3:])
    assert padded_mask.shape == (2, 4, 4)
    assert valid_nodes.tolist() == [[True, True, True, False], [True, True, True, True]]
    padded_logits = torch.zeros_like(padded_bias)
    padded_masked = add_attention_bias(padded_logits, padded_bias, attention_mask=padded_mask)
    assert torch.isneginf(padded_masked[0, :, 3, :]).all()
    assert torch.isfinite(padded_masked[1]).all()


def test_vectorized_batch_preserves_interleaved_node_order() -> None:
    first_edges = _path_edge_index(3)
    second_edges = _path_edge_index(4)
    first_nodes = np.asarray([0, 2, 4])
    second_nodes = np.asarray([1, 3, 5, 6])
    edge_index = torch.as_tensor(
        np.concatenate((first_nodes[first_edges], second_nodes[second_edges]), axis=1),
        dtype=torch.long,
    )
    spectra = [
        precompute_spectrum(first_edges, n=3, k=2),
        precompute_spectrum(second_edges, n=4, k=3),
    ]
    prepared_batch = prepare_spectrum_batch(spectra, k0_pairs=3)
    assert prepared_batch.to("cpu", dtype=torch.float32) is prepared_batch
    batch = torch.tensor([10, 20, 10, 20, 10, 20, 20])
    tokens = torch.randn(7, 5)
    model = ASTBoostPE(
        variant="full",
        heads=2,
        pe_dim=4,
        sign_hidden=8,
        sign_layers=1,
        k0_pairs=3,
        kernel_degree=2,
        token_dim=5,
    ).eval()
    mismatched_model = ASTBoostPE(variant="full", heads=2, pe_dim=4, k0_pairs=2, token_dim=5).eval()

    with pytest.raises(ValueError, match="must match model limit"):
        mismatched_model.forward_padded_batch(tokens, edge_index, prepared_batch, batch)

    with torch.no_grad():
        reference, reference_bias, _ = model.forward_batch(tokens, edge_index, spectra, batch)
        encoded, padded_bias, _, valid_nodes = model.forward_padded_batch(
            tokens, edge_index, prepared_batch, batch
        )
        with pytest.raises(ValueError, match="contiguous=True"):
            model.forward_padded_batch(tokens, edge_index, prepared_batch, batch, contiguous=True)

    torch.testing.assert_close(encoded, reference)
    torch.testing.assert_close(
        padded_bias[0, :, :3, :3], reference_bias[:, first_nodes][:, :, first_nodes]
    )
    torch.testing.assert_close(
        padded_bias[1, :, :4, :4], reference_bias[:, second_nodes][:, :, second_nodes]
    )
    assert valid_nodes.tolist() == [[True, True, True, False], [True, True, True, True]]

    train_tokens = tokens.clone().requires_grad_(True)
    train_encoded, train_bias, _, _ = model.forward_padded_batch(
        train_tokens, edge_index, prepared_batch, batch
    )
    (train_encoded.square().mean() + train_bias.square().mean()).backward()
    assert train_tokens.grad is not None and torch.isfinite(train_tokens.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_learned_kernel_response_has_finite_gradients() -> None:
    edge_index = _path_edge_index(8)
    spectrum = precompute_spectrum(edge_index, n=8, k=4)
    module = SpectralKernelBias(heads=3, degree=3, standardize=True)

    prepared = prepare_spectrum(spectrum, k0_pairs=4)
    assert prepared.to("cpu", dtype=torch.float32) is prepared
    bias = module(prepared)
    loss = bias.square().sum()
    loss.backward()

    assert module.coefficients.grad is not None
    assert module.alpha.grad is not None
    assert torch.isfinite(module.coefficients.grad).all()
    assert torch.isfinite(module.alpha.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_prepared_spectrum_amp_forward_backward() -> None:
    device = torch.device("cuda")
    edge_index = torch.as_tensor(_path_edge_index(12), dtype=torch.long, device=device)
    spectrum = precompute_spectrum(_path_edge_index(12), n=12, k=8)
    prepared = prepare_spectrum(spectrum, k0_pairs=4, device=device)
    model = ASTBoostPE(
        variant="full",
        heads=4,
        pe_dim=8,
        sign_hidden=16,
        sign_layers=2,
        k0_pairs=4,
        kernel_degree=4,
        token_dim=16,
    ).to(device)
    tokens = torch.randn(12, 16, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        encoded, bias = model(tokens, prepared, edge_index)
        loss = encoded.float().square().mean() + bias.float().square().mean()
    loss.backward()

    assert encoded.is_cuda and bias.is_cuda
    assert torch.isfinite(encoded).all() and torch.isfinite(bias).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_batch_paths_follow_computed_dtype() -> None:
    device = torch.device("cuda")
    local_edges = _path_edge_index(6)
    spectrum = precompute_spectrum(local_edges, n=6, k=4)
    prepared = prepare_spectrum(spectrum, k0_pairs=2, device=device)
    spectra = [prepared, prepared]
    prepared_batch = prepare_spectrum_batch(spectra, k0_pairs=2, device=device)
    edge_index = torch.as_tensor(
        np.concatenate((local_edges, local_edges + 6), axis=1),
        dtype=torch.long,
        device=device,
    )
    batch = torch.arange(2, device=device).repeat_interleave(6)
    tokens = torch.randn(12, 8, device=device)
    model = ASTBoostPE(
        variant="full",
        heads=2,
        pe_dim=8,
        sign_hidden=16,
        k0_pairs=2,
        kernel_degree=3,
        token_dim=8,
    ).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        disjoint_tokens, disjoint_bias, _ = model.forward_batch(tokens, edge_index, spectra, batch)
        padded_tokens, padded_bias, _, _ = model.forward_padded_batch(
            tokens, edge_index, prepared_batch, batch, contiguous=True
        )

    assert disjoint_tokens.dtype == padded_tokens.dtype == torch.float16
    assert disjoint_bias.dtype == padded_bias.dtype == torch.float32
    # Fused batched GEMMs can accumulate fp16 products in a different order
    # from the per-graph reference while remaining equivalent at AMP precision.
    torch.testing.assert_close(disjoint_tokens, padded_tokens, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(padded_bias[0], disjoint_bias[:, :6, :6])
    torch.testing.assert_close(padded_bias[1], disjoint_bias[:, 6:, 6:])
