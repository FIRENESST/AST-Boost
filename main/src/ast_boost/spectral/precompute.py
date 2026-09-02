"""Offline, deterministic spectrum preprocessing for AST-Boost.

The output is deliberately a NumPy :class:`~ast_boost.spectral.types.Spectrum`.
It should be serialized with the dataset and treated as constant during model
training; no eigendecomposition is placed in the PyTorch autograd graph.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from .types import Spectrum

LaplacianKind = Literal["sym", "comb", "rw"]


def _to_numpy(value: object, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert NumPy, PyTorch, or array-like values without requiring PyTorch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value, dtype=dtype)
    return array


def undirected_adjacency(
    edge_index: object,
    *,
    n: int,
    edge_weight: object | None = None,
) -> np.ndarray:
    """Build an undirected dense adjacency matrix from a PyG-style edge list.

    This project targets small graphs for its initial experiments.  Duplicated
    reciprocal PyG edges are not double-counted: entries are mirrored with the
    largest observed non-negative weight.  Parallel edges with different weights
    should be coalesced by the dataset loader before calling this helper.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    indices = _to_numpy(edge_index, dtype=np.int64)
    if indices.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if indices.ndim != 2:
        raise ValueError("edge_index must be two-dimensional")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError("edge_index has a node id outside [0, n)")

    edge_count = indices.shape[1]
    if edge_weight is None:
        weights = np.ones(edge_count, dtype=np.float64)
    else:
        weights = _to_numpy(edge_weight, dtype=np.float64).reshape(-1)
        if weights.size != edge_count:
            raise ValueError("edge_weight must have one entry per edge")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("edge_weight must be finite and non-negative")

    adjacency = np.zeros((n, n), dtype=np.float64)
    for source, target, weight in zip(indices[0], indices[1], weights, strict=True):
        if source != target:
            adjacency[source, target] = max(adjacency[source, target], weight)
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def build_laplacian(
    edge_index: object,
    *,
    n: int,
    edge_weight: object | None = None,
    kind: LaplacianKind = "sym",
    isolated_degree: float = 1e-6,
) -> np.ndarray:
    """Construct a graph Laplacian from an undirected edge list.

    ``sym`` is the default required by the AST-Boost kernel.  ``comb`` is kept
    for the README's ablation.  ``rw`` returns the conventional random-walk
    Laplacian, but its non-orthonormal right eigenvectors are not fed directly to
    the invariant kernel; :func:`precompute_spectrum` uses its symmetric similar
    representation instead.
    """
    if isolated_degree <= 0:
        raise ValueError("isolated_degree must be positive")
    adjacency = undirected_adjacency(edge_index, n=n, edge_weight=edge_weight)
    degrees = adjacency.sum(axis=1)
    if kind == "comb":
        return np.diag(degrees) - adjacency
    if kind == "sym":
        safe_degrees = np.where(degrees > 0, degrees, isolated_degree)
        inverse_sqrt = 1.0 / np.sqrt(safe_degrees)
        laplacian = np.eye(n, dtype=np.float64) - (
            inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
        )
        return 0.5 * (laplacian + laplacian.T)
    if kind == "rw":
        safe_degrees = np.where(degrees > 0, degrees, isolated_degree)
        return np.eye(n, dtype=np.float64) - adjacency / safe_degrees[:, None]
    raise ValueError(f"unknown Laplacian kind {kind!r}; expected sym, comb, or rw")


def build_sparse_laplacian(
    edge_index: object,
    *,
    n: int,
    edge_weight: object | None = None,
    kind: LaplacianKind = "sym",
    isolated_degree: float = 1e-6,
):
    """Build a CSR Laplacian without materialising an ``N x N`` dense matrix.

    It is used automatically above ``dense_threshold``.  The public dense
    :func:`build_laplacian` remains convenient for small graph fixtures.
    """
    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError("sparse Laplacians require `scipy>=1.10`.") from error
    if n < 0:
        raise ValueError("n must be non-negative")
    if isolated_degree <= 0:
        raise ValueError("isolated_degree must be positive")
    indices = _to_numpy(edge_index, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError("edge_index has a node id outside [0, n)")
    edge_count = indices.shape[1]
    if edge_weight is None:
        weights = np.ones(edge_count, dtype=np.float64)
    else:
        weights = _to_numpy(edge_weight, dtype=np.float64).reshape(-1)
        if weights.size != edge_count:
            raise ValueError("edge_weight must have one entry per edge")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("edge_weight must be finite and non-negative")
    keep = indices[0] != indices[1]
    adjacency = sparse.coo_matrix(
        (weights[keep], (indices[0, keep], indices[1, keep])), shape=(n, n), dtype=np.float64
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if kind == "comb":
        return (sparse.diags(degrees, format="csr") - adjacency).tocsr()
    safe_degrees = np.where(degrees > 0, degrees, isolated_degree)
    if kind == "sym":
        inverse_sqrt = sparse.diags(1.0 / np.sqrt(safe_degrees), format="csr")
        return (sparse.eye(n, format="csr") - inverse_sqrt @ adjacency @ inverse_sqrt).tocsr()
    if kind == "rw":
        inverse = sparse.diags(1.0 / safe_degrees, format="csr")
        return (sparse.eye(n, format="csr") - inverse @ adjacency).tocsr()
    raise ValueError(f"unknown Laplacian kind {kind!r}; expected sym, comb, or rw")


def degeneracy_blocks(
    eigenvalues: object,
    *,
    relative_gap: float = 1e-2,
    tau: float = 1e-6,
) -> np.ndarray:
    """Group sorted eigenvalues using the v0.3 relative-gap rule.

    The current value is compared with the running block mean, which avoids a
    long near-degenerate chain incorrectly joining unrelated frequencies.
    """
    if relative_gap < 0:
        raise ValueError("relative_gap must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    values = _to_numpy(eigenvalues, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(values) < -1e-10):
        raise ValueError("eigenvalues must be sorted before grouping")

    block_ids = np.zeros(values.size, dtype=np.int64)
    block = 0
    running_sum = float(values[0])
    count = 1
    for index in range(1, values.size):
        mean = running_sum / count
        threshold = relative_gap * max(abs(mean), tau)
        if abs(values[index] - mean) > threshold:
            block += 1
            running_sum = float(values[index])
            count = 1
        else:
            running_sum += float(values[index])
            count += 1
        block_ids[index] = block
    return block_ids


def _low_spectrum(
    matrix: object,
    *,
    requested: int,
    dense_threshold: int,
    sparse_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low eigenpairs with dense eigh or v0.3's shift-invert eigsh."""
    n = matrix.shape[0]
    if n <= dense_threshold or requested >= n - 1:
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        return np.linalg.eigh(dense)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "large-graph spectrum preprocessing requires SciPy; install `scipy>=1.10` "
            "or increase dense_threshold for this graph."
        ) from error
    values, vectors = eigsh(
        csr_matrix(matrix),
        k=requested,
        sigma=sparse_sigma,
        which="LM",
    )
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _candidate_positions(
    values: np.ndarray,
    *,
    skip_zero: bool,
    zero_tolerance: float,
) -> np.ndarray:
    return np.flatnonzero(values > zero_tolerance) if skip_zero else np.arange(values.size)


def _complete_block_selection(
    values: np.ndarray,
    *,
    k: int,
    skip_zero: bool,
    zero_tolerance: float,
    degeneracy_eps: float,
    degeneracy_tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select at least ``k`` modes without cutting through a spectral block."""
    candidates = _candidate_positions(values, skip_zero=skip_zero, zero_tolerance=zero_tolerance)
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    blocks = degeneracy_blocks(values[candidates], relative_gap=degeneracy_eps, tau=degeneracy_tau)
    cutoff = min(k, candidates.size) - 1
    chosen_mask = blocks <= blocks[cutoff]
    return candidates[chosen_mask], blocks[chosen_mask]


def _cutoff_block_reaches_observed_end(
    values: np.ndarray,
    *,
    k: int,
    skip_zero: bool,
    zero_tolerance: float,
    degeneracy_eps: float,
    degeneracy_tau: float,
) -> bool:
    """Whether a sparse solve needs more modes to prove a block is complete."""
    candidates = _candidate_positions(values, skip_zero=skip_zero, zero_tolerance=zero_tolerance)
    if candidates.size < k:
        return True
    blocks = degeneracy_blocks(values[candidates], relative_gap=degeneracy_eps, tau=degeneracy_tau)
    return bool(blocks[-1] == blocks[k - 1])


def precompute_spectrum(
    edge_index: object,
    *,
    n: int,
    k: int = 8,
    edge_weight: object | None = None,
    laplacian: LaplacianKind = "sym",
    skip_zero: bool = True,
    zero_tolerance: float = 1e-8,
    degeneracy_eps: float = 1e-2,
    degeneracy_tau: float = 1e-6,
    dense_threshold: int = 256,
    sparse_sigma: float = 1e-5,
) -> Spectrum:
    """Precompute the smallest positive Laplacian modes and degeneracy blocks.

    ZINC-scale graphs use dense ``eigh``.  Larger graphs use
    ``scipy.sparse.linalg.eigsh(sigma=1e-5, which='LM')`` (shift-invert) as
    prescribed in v0.3; the nonzero shift avoids factoring an exactly singular
    normalized Laplacian.  The solver asks for extra modes and expands the
    request when disconnected components consume the zero eigenspace.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        raise ValueError("k must be non-negative")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")
    if dense_threshold < 1:
        raise ValueError("dense_threshold must be positive")
    if sparse_sigma <= 0:
        raise ValueError("sparse_sigma must be positive")
    if n == 0 or k == 0:
        stored_kind: LaplacianKind = "sym" if laplacian == "rw" else laplacian
        return Spectrum(
            eigenvalues=np.empty(0),
            eigenvectors=np.empty((n, 0)),
            block_ids=np.empty(0, dtype=np.int64),
            laplacian=stored_kind,
            zero_tolerance=zero_tolerance,
        )

    solve_kind: LaplacianKind = laplacian
    if laplacian == "rw":
        # L_rw is similar to L_sym.  The latter supplies an orthonormal basis,
        # which is required by U diag(g) U^T and by the O(m) block argument.
        warnings.warn(
            "rw requested: using the orthonormal L_sym representation for spectral PE; "
            "the cached spectrum is labelled 'sym' to avoid a false random-walk attribution.",
            RuntimeWarning,
            stacklevel=2,
        )
        solve_kind = "sym"
    if n <= dense_threshold:
        matrix = build_laplacian(
            edge_index,
            n=n,
            edge_weight=edge_weight,
            kind=solve_kind,
        )
        values, vectors = np.linalg.eigh(matrix)
    else:
        matrix = build_sparse_laplacian(
            edge_index,
            n=n,
            edge_weight=edge_weight,
            kind=solve_kind,
        )
        # eigsh requires k < N.  Requesting extra modes makes the common
        # connected case a single solve, while the loop handles many components.
        requested = min(n - 1, max(2 * k, k + 8, 1))
        while True:
            values, vectors = _low_spectrum(
                matrix,
                requested=requested,
                dense_threshold=dense_threshold,
                sparse_sigma=sparse_sigma,
            )
            values = np.maximum(values, 0.0)
            if requested >= n - 1 or not _cutoff_block_reaches_observed_end(
                values,
                k=k,
                skip_zero=skip_zero,
                zero_tolerance=zero_tolerance,
                degeneracy_eps=degeneracy_eps,
                degeneracy_tau=degeneracy_tau,
            ):
                break
            requested = min(n - 1, max(requested + 1, requested * 2))
    values = np.maximum(values, 0.0)  # suppress harmless solver roundoff

    chosen, block_ids = _complete_block_selection(
        values,
        k=k,
        skip_zero=skip_zero,
        zero_tolerance=zero_tolerance,
        degeneracy_eps=degeneracy_eps,
        degeneracy_tau=degeneracy_tau,
    )
    selected_values = values[chosen]
    selected_vectors = vectors[:, chosen]
    return Spectrum(
        eigenvalues=selected_values,
        eigenvectors=selected_vectors,
        block_ids=block_ids,
        laplacian=solve_kind,
        zero_tolerance=zero_tolerance,
    )


def save_spectrum(spectrum: Spectrum, path: str | Path) -> Path:
    """Persist a frozen spectrum in a portable compressed NPZ file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        eigenvalues=spectrum.eigenvalues,
        eigenvectors=spectrum.eigenvectors,
        block_ids=spectrum.block_ids,
        laplacian=np.asarray(spectrum.laplacian),
        zero_tolerance=np.asarray(spectrum.zero_tolerance),
    )
    return destination


def load_spectrum(path: str | Path) -> Spectrum:
    """Load a :class:`Spectrum` produced by :func:`save_spectrum`."""
    with np.load(Path(path), allow_pickle=False) as cache:
        return Spectrum(
            eigenvalues=cache["eigenvalues"],
            eigenvectors=cache["eigenvectors"],
            block_ids=cache["block_ids"],
            laplacian=str(cache["laplacian"].item()),
            zero_tolerance=float(cache["zero_tolerance"].item()),
        )
