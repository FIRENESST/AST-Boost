"""Theory-to-runtime regressions, including cases ordinary path graphs miss."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy import sparse

from ast_boost import ASTBoostPE, prepare_spectrum, prepare_spectrum_batch
from ast_boost.spectral.kernel import SpectralKernelBias, standardize_offdiagonal
from ast_boost.spectral.precompute import (
    _low_spectrum,
    build_laplacian,
    build_sparse_laplacian,
    precompute_spectrum,
)
from ast_boost.spectral.signnet import SignInvariantFieldEncoder
from ast_boost.spectral.types import Spectrum


def path_edges(n: int) -> np.ndarray:
    source = np.arange(n - 1)
    return np.stack((np.r_[source, source + 1], np.r_[source + 1, source]))


def test_masked_field_readout_matches_an_empty_single_graph() -> None:
    torch.manual_seed(713)
    encoder = SignInvariantFieldEncoder(hidden_dim=8, out_dim=3, layers=1)
    states = torch.randn(2, 2, 4, 8, requires_grad=True)
    mask = torch.tensor([[False, False], [True, True]])
    actual = encoder.readout_padded(states, mask)
    expected = encoder.readout(states[:0, 0])
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    actual[0].sum().backward()
    assert states.grad is not None
    assert torch.count_nonzero(states.grad) == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in encoder.parameters())


@pytest.mark.parametrize("kind", ["sym", "comb", "rw"])
def test_duplicate_edges_have_identical_dense_and_sparse_semantics(kind: str) -> None:
    edges = np.asarray([[0, 0, 1, 1, 2, 2, 3], [1, 1, 0, 2, 1, 3, 2]])
    weights = np.asarray([1.0, 3.0, 2.0, 0.5, 0.5, 2.0, 2.0])
    dense = build_laplacian(edges, n=4, edge_weight=weights, kind=kind)
    csr = build_sparse_laplacian(edges, n=4, edge_weight=weights, kind=kind)
    np.testing.assert_allclose(csr.toarray(), dense, atol=1e-14)


def test_default_shift_invert_returns_lowest_not_nearest_positive_modes() -> None:
    values = np.r_[0.0, 1e-7, 2e-7, np.linspace(8e-6, 12e-6, 10), 0.3, 0.8, 1.4]
    default_sigma = inspect.signature(precompute_spectrum).parameters["sparse_sigma"].default
    actual, _ = _low_spectrum(
        sparse.diags(values), requested=3, dense_threshold=1, sparse_sigma=default_sigma
    )
    np.testing.assert_allclose(actual, values[:3], atol=1e-13, rtol=1e-6)


@pytest.mark.parametrize("padded", [False, True])
def test_offdiagonal_variance_avoids_cancellation(monkeypatch, padded: bool) -> None:
    values = 1000.0 + torch.arange(16, dtype=torch.float32).reshape(4, 4) * 0.01
    values = (values + values.T) / 2
    values.fill_diagonal_(9e6)
    values.requires_grad_(True)
    kernel = SpectralKernelBias(heads=1, alpha_init=1.0)
    expected = torch.from_numpy(standardize_offdiagonal(values.detach().double().numpy()))
    if padded:
        monkeypatch.setattr(kernel, "raw_kernel_padded", lambda *args: values[None, None])
        output = kernel.forward_padded(None, None, None, torch.ones(1, 4, dtype=torch.bool))[0, 0]
    else:
        monkeypatch.setattr(kernel, "raw_kernel", lambda *args, **kwargs: values[None])
        output = kernel(None)[0]
    torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-5)
    output.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_autocast_preserves_float32_spectral_kernel() -> None:
    spectrum = precompute_spectrum(path_edges(7), n=7, k=4)
    kernel = SpectralKernelBias(heads=2, degree=4)
    prepared = prepare_spectrum(spectrum)
    with torch.no_grad():
        kernel.coefficients.copy_(torch.linspace(-0.7, 1.3, 10).reshape(2, 5))
    expected = kernel.raw_kernel(prepared)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = kernel.raw_kernel(prepared)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("variant", ["lite", "kern", "full"])
@pytest.mark.parametrize("field_scaling", ["none", "size"])
@pytest.mark.sign_invariance
@pytest.mark.degeneracy
@pytest.mark.permutation
def test_learned_batch_is_invariant_and_matches_mixed_graph_singletons(
    variant: str, field_scaling: str
) -> None:
    """Includes no spectrum, only a degenerate block, and legal singleton pairs."""
    torch.manual_seed(419)
    rng = np.random.default_rng(91)
    empty = precompute_spectrum(path_edges(2), n=2, k=0)
    basis = np.linalg.qr(rng.normal(size=(4, 2)))[0]
    degenerate = Spectrum(np.array([0.4, 0.403]), basis, np.array([0, 0]))
    simple = precompute_spectrum(path_edges(6), n=6, k=4)
    spectra = [empty, degenerate, simple]
    # Independent signs, block rotation, and a within-graph node permutation.
    q = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    permutation = np.asarray([3, 0, 5, 1, 4, 2])
    changed = [
        empty,
        Spectrum(degenerate.eigenvalues, basis @ q, degenerate.block_ids),
        Spectrum(
            simple.eigenvalues,
            simple.eigenvectors[permutation] * np.array([-1, 1, -1, 1]),
            simple.block_ids,
        ),
    ]
    inverse = np.argsort(permutation)
    edges = torch.tensor(
        np.concatenate([path_edges(2), path_edges(4) + 2, path_edges(6) + 6], axis=1)
    )
    changed_edges = torch.tensor(
        np.concatenate([path_edges(2), path_edges(4) + 2, inverse[path_edges(6)] + 6], axis=1)
    )
    batch = torch.repeat_interleave(torch.arange(3), torch.tensor([2, 4, 6]))
    tokens = torch.randn(12, 3)
    model = ASTBoostPE(
        variant=variant,
        heads=2,
        pe_dim=4,
        sign_hidden=8,
        sign_layers=1,
        k0_pairs=4,
        kernel_degree=3,
        token_dim=None,
        field_scaling=field_scaling,
    ).eval()
    with torch.no_grad():
        model.kernel.coefficients.copy_(
            torch.tensor([[0.2, 0.6, -0.4, 1.1], [0.8, -0.2, 0.5, 1.0]])
        )
        single, _, _ = model.forward_batch(tokens, edges, spectra, batch)
        packed = prepare_spectrum_batch(spectra)
        actual, bias, _, _ = model.forward_padded_batch(
            tokens, edges, packed, batch, contiguous=True
        )
        permuted_tokens = torch.cat((tokens[:6], tokens[6:][permutation]))
        transformed, transformed_bias, _, _ = model.forward_padded_batch(
            permuted_tokens, changed_edges, prepare_spectrum_batch(changed), batch, contiguous=True
        )
    torch.testing.assert_close(actual, single, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(transformed[:6], actual[:6], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(transformed[6:], actual[6:][permutation], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(transformed_bias[:2], bias[:2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        transformed_bias[2], bias[2][:, permutation][:, :, permutation], rtol=1e-5, atol=1e-6
    )


def test_attention_softmax_has_zero_rows_and_finite_padding_gradients() -> None:
    from ast_boost.spectral.gps_adapter import attention_softmax

    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    bias = torch.randn_like(logits, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
    mask = valid[:, :, None] & valid[:, None, :]
    weights = attention_softmax(logits, bias, attention_mask=mask)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights[0, :, :2].sum(-1), torch.ones(2, 2))
    assert torch.count_nonzero(weights[0, :, 2:]) == 0
    assert torch.count_nonzero(weights[1]) == 0
    weights.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(bias.grad).all()
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_contiguous_path_rejects_descending_graph_ids() -> None:
    spectrum = precompute_spectrum(path_edges(3), n=3, k=2)
    model = ASTBoostPE(variant="lite", pe_dim=2, sign_hidden=4)
    batch = torch.tensor([20, 20, 20, 10, 10, 10])
    with pytest.raises(ValueError, match="contiguous"):
        model.forward_padded_batch(
            torch.zeros(6, 2),
            torch.tensor(np.c_[path_edges(3), path_edges(3) + 3]),
            prepare_spectrum_batch([spectrum, spectrum]),
            batch,
            contiguous=True,
        )


def test_centered_standardization_passes_double_precision_gradcheck() -> None:
    from ast_boost.spectral.kernel import _standardize_torch

    torch.manual_seed(88)
    values = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    mask = ~torch.eye(3, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x: _standardize_torch(x, mask, 1e-8), (values,))


def test_fourth_moment_and_projector_power_theory_boundaries() -> None:
    rng = np.random.default_rng(37)
    u = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    # <w_00, w_01> contains the sign factor s_0^3 s_1, so is not invariant.
    moment = np.dot(u[:, 0] ** 2, u[:, 0] * u[:, 1])
    flipped = u * np.array([1, -1, 1])
    changed = np.dot(flipped[:, 0] ** 2, flipped[:, 0] * flipped[:, 1])
    assert abs(moment) > 1e-4
    np.testing.assert_allclose(changed, -moment)
    np.testing.assert_allclose(
        np.sum((u[:, 0] * u[:, 1]) ** 2), np.sum((flipped[:, 0] * flipped[:, 1]) ** 2)
    )
    projector = u[:, :2] @ u[:, :2].T
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.degeneracy
@pytest.mark.kernel_invariance
def test_cuda_half_tokens_keep_frozen_spectra_full_precision_and_gradients() -> None:
    torch.manual_seed(147)
    rng = np.random.default_rng(59)
    u = np.linalg.qr(rng.normal(size=(7, 4)))[0]
    values = np.array([0.4, 0.403, 0.9, 1.5])
    blocks = np.array([0, 0, 1, 2])
    rotated_u = u.copy()
    rotated_u[:, :2] = u[:, :2] @ np.linalg.qr(rng.normal(size=(2, 2)))[0]
    spectra = [Spectrum(values, u, blocks), Spectrum(values, rotated_u, blocks)]
    packed_cpu = prepare_spectrum_batch(spectra, pin_memory=True)
    assert packed_cpu.is_pinned
    packed = packed_cpu.to("cuda")
    model = ASTBoostPE(
        variant="full", heads=2, pe_dim=4, sign_hidden=8, sign_layers=1, token_dim=3
    ).cuda()
    with torch.no_grad():
        model.kernel.coefficients.copy_(torch.linspace(-0.4, 1.2, 18, device="cuda").reshape(2, 9))
    edges = torch.tensor(np.c_[path_edges(7), path_edges(7) + 7], device="cuda")
    tokens = torch.randn(7, 3, device="cuda", dtype=torch.float16).repeat(2, 1).requires_grad_(True)
    batch = torch.arange(2, device="cuda").repeat_interleave(7)
    with torch.autocast("cuda", dtype=torch.float16):
        assert model._prepare_batch(packed, tokens).dtype == torch.float32
        encoded, bias, _, _ = model.forward_padded_batch(
            tokens, edges, packed, batch, contiguous=True
        )
        loss = encoded.float().square().mean() + bias.square().mean()
    assert bias.dtype == torch.float32
    torch.testing.assert_close(bias[0], bias[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(encoded[:7], encoded[7:], rtol=5e-3, atol=5e-4)
    loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_alias_preserves_cache_identity_and_avoids_cpu_roundtrip(monkeypatch) -> None:
    from ast_boost import TorchSpectrum

    spectrum = prepare_spectrum(precompute_spectrum(path_edges(5), n=5, k=3), device="cuda")
    assert spectrum.to("cuda") is spectrum
    original_to = TorchSpectrum.to
    targets = []

    def recording_to(self, device, **kwargs):
        targets.append(torch.device(device).type)
        return original_to(self, device, **kwargs)

    monkeypatch.setattr(TorchSpectrum, "to", recording_to)
    packed = prepare_spectrum_batch([spectrum, spectrum], device="cuda")
    assert targets == ["cuda", "cuda"]
    assert packed.to("cuda") is packed


@pytest.mark.kernel_invariance
def test_learned_response_has_block_clamp_positive_and_negative_controls() -> None:
    rng = np.random.default_rng(28)
    u = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    changed_u = u.copy()
    changed_u[:, :2] = u[:, :2] @ np.array([[0.6, -0.8], [0.8, 0.6]])
    spectrum = Spectrum(np.array([0.5, 0.503, 1.3]), u, np.array([0, 0, 1]))
    rotated = Spectrum(spectrum.eigenvalues, changed_u, spectrum.block_ids)
    kernel = SpectralKernelBias(heads=1, degree=1, standardize=False).double()
    with torch.no_grad():
        kernel.coefficients.copy_(torch.tensor([[0.0, 2.0]]))
    torch.testing.assert_close(kernel(spectrum), kernel(rotated), rtol=1e-10, atol=1e-12)
    kernel.block_clamp = False
    assert not torch.allclose(kernel(spectrum), kernel(rotated), rtol=1e-7, atol=1e-8)


def test_constant_kernel_zeroes_bias_without_nan_and_keeps_gradients() -> None:
    from ast_boost.spectral.kernel import _standardize_torch

    values = torch.full((2, 1, 3, 3), 0.125, requires_grad=True)
    mask = (~torch.eye(3, dtype=torch.bool))[None, None].expand(2, 1, 3, 3).clone()
    mask[1] = False
    output = _standardize_torch(values, mask, 1e-8)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    output.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_adapter_exposes_kernel_domain_for_combinatorial_ablation() -> None:
    model = ASTBoostPE(variant="lite", heads=1, kernel_degree=1, kernel_domain_max=8.0)
    with torch.no_grad():
        model.kernel.coefficients.copy_(torch.tensor([[0.0, 8.0]]))
    torch.testing.assert_close(
        model.kernel.response(torch.tensor([3.0, 5.0])), torch.tensor([[3.0, 5.0]])
    )
