"""Executable invariance requirements from the AST-Boost v0.3 specification.

These tests deliberately use parameter-free operations on the spectral fields.
That keeps the assertions about the mathematical transformation laws separate
from any learned SignNet/GPS implementation.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from ast_boost.spectral.fields import first_order_fields, second_order_fields, second_order_pairs
from ast_boost.spectral.kernel import filtered_kernel, standardize_offdiagonal
from ast_boost.spectral.precompute import precompute_spectrum
from ast_boost.spectral.types import Spectrum

DTYPE = np.float64
RTOL = 1.0e-7
ATOL = 1.0e-8


def _orthonormal_columns(n: int, k: int) -> np.ndarray:
    """Return a deterministic, dense orthonormal basis for a small fixture."""

    # A non-symmetric Vandermonde seed has full column rank and avoids
    # accidental zero fields in the tests.
    values = np.vander(np.linspace(-0.83, 0.91, n, dtype=DTYPE), N=k, increasing=True)
    q, _ = np.linalg.qr(values, mode="reduced")
    return q


def _spectrum(
    eigenvalues: np.ndarray | list[float],
    eigenvectors: np.ndarray,
    block_ids: np.ndarray | list[int],
) -> Spectrum:
    return Spectrum(
        eigenvalues=np.asarray(eigenvalues, dtype=DTYPE),
        eigenvectors=np.asarray(eigenvectors, dtype=DTYPE),
        block_ids=np.asarray(block_ids, dtype=np.int64),
    )


def _rotate_first_block(spectrum: Spectrum, theta: float = 0.61) -> Spectrum:
    """Apply a non-trivial right O(2) action to the first two eigenvectors."""

    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=DTYPE,
    )
    rotated = spectrum.eigenvectors.copy()
    rotated[:, :2] = rotated[:, :2] @ rotation
    return _spectrum(spectrum.eigenvalues, rotated, spectrum.block_ids)


def _pairs_as_tuples(pairs: object) -> list[tuple[int, int]]:
    """Normalise the public pair container for concise mathematical checks."""

    return [tuple(int(index) for index in pair) for pair in np.asarray(pairs).tolist()]


def _field_rows(fields: np.ndarray, pair_count: int, n: int) -> np.ndarray:
    """Accept either conventional field layout: (pairs, nodes) or its transpose."""

    if fields.shape == (pair_count, n):
        return fields
    if fields.shape == (n, pair_count):
        return fields.T
    raise AssertionError(
        "second_order_fields must return one scalar graph signal per selected pair; "
        f"got shape {tuple(fields.shape)}, expected {(pair_count, n)} or {(n, pair_count)}"
    )


def _linear_response(eigenvalues: np.ndarray) -> np.ndarray:
    """A fixed, non-constant response with no trainable parameters."""

    return 1.0 + 17.0 * eigenvalues


@pytest.fixture
def simple_spectrum() -> Spectrum:
    return _spectrum(
        [0.16, 0.43, 0.91, 1.37],
        _orthonormal_columns(n=6, k=4),
        [0, 1, 2, 3],
    )


@pytest.mark.sign_invariance
def test_sign_flips_leave_filtered_kernel_and_parameter_free_field_encoding_invariant(
    simple_spectrum: Spectrum,
) -> None:
    """U -> UD leaves K fixed and makes every w_pq flip only by s_p s_q."""

    signs = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=DTYPE)
    signed = _spectrum(
        simple_spectrum.eigenvalues,
        simple_spectrum.eigenvectors * signs,
        simple_spectrum.block_ids,
    )

    kernel = filtered_kernel(simple_spectrum, response=_linear_response, block_clamp=True)
    signed_kernel = filtered_kernel(signed, response=_linear_response, block_clamp=True)
    np.testing.assert_allclose(signed_kernel, kernel, rtol=RTOL, atol=ATOL)

    # Standardisation is part of the bias path and must not reintroduce a sign
    # dependence after the (already invariant) kernel was formed.
    np.testing.assert_allclose(
        standardize_offdiagonal(signed_kernel),
        standardize_offdiagonal(kernel),
        rtol=RTOL,
        atol=ATOL,
    )

    pairs = _pairs_as_tuples(second_order_pairs(simple_spectrum, k0_pairs=4))
    fields = _field_rows(
        second_order_fields(simple_spectrum, pairs=pairs),
        len(pairs),
        simple_spectrum.eigenvectors.shape[0],
    )
    signed_fields = _field_rows(
        second_order_fields(signed, pairs=pairs), len(pairs), simple_spectrum.eigenvectors.shape[0]
    )

    pair_signs = np.asarray([signs[p] * signs[q] for p, q in pairs], dtype=DTYPE)
    assert np.any(pair_signs == -1), "fixture must contain a field that reverses sign"
    np.testing.assert_allclose(signed_fields, pair_signs[:, None] * fields, rtol=RTOL, atol=ATOL)

    # Squaring each signal is a parameter-free sign-symmetrisation.  This
    # checks the SignNet premise without depending on a learned encoder.
    np.testing.assert_allclose(signed_fields**2, fields**2, rtol=RTOL, atol=ATOL)


@pytest.mark.degeneracy
def test_degenerate_projection_is_rotation_invariant_and_block_internal_pairs_are_excluded() -> (
    None
):
    """A two-dimensional eigenspace is represented by its projector, not vectors."""

    spectrum = _spectrum(
        [0.30, 0.30, 0.88, 1.42],
        _orthonormal_columns(n=7, k=4),
        [0, 0, 1, 2],
    )
    rotated = _rotate_first_block(spectrum)

    projector = spectrum.eigenvectors[:, :2] @ spectrum.eigenvectors[:, :2].T
    rotated_projector = rotated.eigenvectors[:, :2] @ rotated.eigenvectors[:, :2].T
    np.testing.assert_allclose(rotated_projector, projector, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        np.diagonal(rotated_projector), np.diagonal(projector), rtol=RTOL, atol=ATOL
    )
    # The actual absolute-field input to Kern/Full replaces B's arbitrary
    # columns by diag(P_B), so its NumPy precursor is also rotation-invariant.
    np.testing.assert_allclose(
        first_order_fields(rotated), first_order_fields(spectrum), rtol=RTOL, atol=ATOL
    )

    # Exact repeated eigenvalues are safe even before clamping; this verifies
    # the expected P_B form of the kernel for a true degenerate block.
    kernel = filtered_kernel(spectrum, response=_linear_response, block_clamp=True)
    rotated_kernel = filtered_kernel(rotated, response=_linear_response, block_clamp=True)
    np.testing.assert_allclose(rotated_kernel, kernel, rtol=RTOL, atol=ATOL)

    pairs = _pairs_as_tuples(second_order_pairs(spectrum, k0_pairs=4))
    block_sizes = Counter(spectrum.block_ids.tolist())
    assert (0, 0) not in pairs and (0, 1) not in pairs and (1, 1) not in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs and (0, 3) not in pairs
    assert (2, 2) in pairs, "diagonal pair of a singleton block remains well-defined"
    assert all(
        block_sizes[spectrum.block_ids[p]] == 1 and block_sizes[spectrum.block_ids[q]] == 1
        for p, q in pairs
    )

    # A caller must not be able to bypass the safe pair selector with a manual
    # pair either: u_p * u_q is basis-dependent as soon as *either* endpoint
    # lies in an O(m), m > 1, eigenspace.
    for unsafe_pair in ((0, 0), (0, 2), (1, 3)):
        try:
            second_order_fields(spectrum, pairs=[unsafe_pair])
        except ValueError:
            pass
        else:
            raise AssertionError(
                "second_order_fields must reject every pair touching a non-singleton block"
            )

    fields = _field_rows(
        second_order_fields(spectrum, pairs=pairs), len(pairs), spectrum.eigenvectors.shape[0]
    )
    expected = np.stack(
        [spectrum.eigenvectors[:, p] * spectrum.eigenvectors[:, q] for p, q in pairs]
    )
    np.testing.assert_allclose(fields, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.kernel_invariance
def test_block_clamp_makes_a_near_degenerate_kernel_rotation_invariant() -> None:
    """The v0.3 negative/positive pair for the block-clamp safety condition."""

    # The first two values are close enough to be deliberately represented as
    # one block, while a non-constant response still assigns them different
    # raw weights.  This models the near-degenerate case covered by v0.3.
    spectrum = _spectrum(
        [0.500, 0.503, 1.10, 1.61],
        _orthonormal_columns(n=7, k=4),
        [0, 0, 1, 2],
    )
    rotated = _rotate_first_block(spectrum, theta=0.79)

    unclamped = filtered_kernel(spectrum, response=_linear_response, block_clamp=False)
    unclamped_rotated = filtered_kernel(rotated, response=_linear_response, block_clamp=False)
    assert not np.allclose(unclamped_rotated, unclamped, rtol=1.0e-5, atol=1.0e-7), (
        "without block clamping, unequal filter values inside a rotated block "
        "must expose the basis dependence"
    )

    clamped = filtered_kernel(spectrum, response=_linear_response, block_clamp=True)
    clamped_rotated = filtered_kernel(rotated, response=_linear_response, block_clamp=True)
    np.testing.assert_allclose(clamped_rotated, clamped, rtol=RTOL, atol=ATOL)


@pytest.mark.kernel_invariance
def test_offdiagonal_standardisation_is_per_graph(
    simple_spectrum: Spectrum,
) -> None:
    kernel = filtered_kernel(simple_spectrum, response=_linear_response, block_clamp=True)
    standardised = standardize_offdiagonal(kernel)
    off_diagonal = ~np.eye(kernel.shape[-1], dtype=bool)
    values = standardised[off_diagonal]

    np.testing.assert_allclose(values.mean(), 0.0, atol=ATOL, rtol=0)
    np.testing.assert_allclose(values.std(), 1.0, atol=1.0e-7, rtol=1.0e-7)


def _path_edge_index(n: int) -> np.ndarray:
    source = np.arange(n - 1, dtype=np.int64)
    target = source + 1
    return np.stack((np.concatenate((source, target)), np.concatenate((target, source))))


def _cycle_edge_index(n: int) -> np.ndarray:
    source = np.arange(n, dtype=np.int64)
    target = (source + 1) % n
    return np.stack((np.concatenate((source, target)), np.concatenate((target, source))))


@pytest.mark.degeneracy
def test_precompute_does_not_cut_through_a_degenerate_block_at_nominal_k() -> None:
    """A nominal k is a lower target, never a basis-dependent block cut.

    C4 has normalized-Laplacian spectrum (0, 1, 1, 2).  With the zero mode
    skipped, k=1 lands in the multiplicity-two lambda=1 space; retaining both
    modes is necessary before using its projector or deciding that its fields
    must be omitted.
    """

    spectrum = precompute_spectrum(_cycle_edge_index(4), n=4, k=1, skip_zero=True)

    assert spectrum.k == 2
    np.testing.assert_allclose(spectrum.eigenvalues, [1.0, 1.0], rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(spectrum.block_ids, [0, 0])
    np.testing.assert_allclose(
        spectrum.eigenvectors.T @ spectrum.eigenvectors,
        np.eye(2),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.degeneracy
@pytest.mark.kernel_invariance
def test_torch_kern_and_full_are_basis_invariant_for_a_nontrivial_block() -> None:
    """The train-time adapters must not read a particular basis inside B={0,1}."""

    torch = pytest.importorskip("torch")
    from ast_boost import ASTBoostPE

    spectrum = _spectrum(
        [0.30, 0.30, 0.88, 1.42],
        _orthonormal_columns(n=6, k=4),
        [0, 0, 1, 2],
    )
    rotated = _rotate_first_block(spectrum, theta=0.71)
    edge_index = torch.as_tensor(_path_edge_index(6), dtype=torch.long)
    x = torch.zeros((6, 3), dtype=torch.float32)

    # Reuse each initialized model for both bases.  Evaluation mode makes the
    # comparison about the spectral representation, not stochastic training
    # behaviour.  Kern must discard/aggregate B safely in its first-order path;
    # Full must do the same in both first- and second-order paths.
    for variant in ("kern", "full"):
        model = ASTBoostPE(
            variant=variant,
            heads=2,
            pe_dim=5,
            sign_hidden=8,
            sign_layers=1,
            k0_pairs=4,
            kernel_degree=2,
            block_clamp=True,
            standardize_bias=True,
        ).eval()
        with torch.no_grad():
            encoded, bias = model(x, spectrum, edge_index)
            rotated_encoded, rotated_bias = model(x, rotated, edge_index)

        torch.testing.assert_close(
            rotated_encoded[:, x.shape[1] :], encoded[:, x.shape[1] :], rtol=1.0e-5, atol=1.0e-6
        )
        torch.testing.assert_close(rotated_bias, bias, rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.permutation
def test_precomputed_kernel_and_parameter_free_fields_are_node_permutation_equivariant() -> None:
    """Precomputation and every frozen, parameter-free spectral object commute with Π."""

    n = 6
    edge_index = _path_edge_index(n)
    permutation = np.asarray([3, 0, 5, 1, 4, 2], dtype=np.int64)  # old node -> new node
    permuted_edge_index = permutation[edge_index]

    spectrum = precompute_spectrum(edge_index, n=n, k=4, skip_zero=True)
    permuted_spectrum = precompute_spectrum(permuted_edge_index, n=n, k=4, skip_zero=True)
    kernel = filtered_kernel(spectrum, response=_linear_response, block_clamp=True)
    permuted_kernel = filtered_kernel(
        permuted_spectrum, response=_linear_response, block_clamp=True
    )
    np.testing.assert_allclose(
        permuted_kernel[np.ix_(permutation, permutation)], kernel, rtol=2.0e-6, atol=2.0e-7
    )

    pairs = _pairs_as_tuples(second_order_pairs(spectrum, k0_pairs=4))
    permuted_pairs = _pairs_as_tuples(second_order_pairs(permuted_spectrum, k0_pairs=4))
    assert permuted_pairs == pairs
    fields = _field_rows(second_order_fields(spectrum, pairs=pairs), len(pairs), n)
    permuted_fields = _field_rows(
        second_order_fields(permuted_spectrum, pairs=permuted_pairs), len(permuted_pairs), n
    )

    # Individual eigensolver columns can independently change sign, so compare
    # the parameter-free sign-invariant field representation.  Its node axis
    # still has to transform equivariantly under the graph permutation.
    np.testing.assert_allclose(
        (permuted_fields**2)[:, permutation], fields**2, rtol=2.0e-6, atol=2.0e-7
    )
