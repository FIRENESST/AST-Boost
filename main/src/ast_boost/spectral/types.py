"""Immutable containers shared by preprocessing and train-time modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_float_array(value: object, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_int_array(value: object, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.int64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Spectrum:
    """A frozen, truncated orthonormal eigensystem of a graph Laplacian.

    ``eigenvectors`` is shaped ``(num_nodes, k)``.  ``block_ids[p]`` identifies
    the (near-)degenerate block containing frequency ``p``.  A block is a
    subspace, not a choice of basis; callers should use :attr:`clamped_eigenvalues`
    or :attr:`block_projectors` whenever a calculation involves a whole block.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    block_ids: np.ndarray
    laplacian: str = "sym"
    zero_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.eigenvalues, ndim=1, name="eigenvalues")
        vectors = _readonly_float_array(self.eigenvectors, ndim=2, name="eigenvectors")
        blocks = _readonly_int_array(self.block_ids, ndim=1, name="block_ids")

        if vectors.shape[1] != values.size:
            raise ValueError(
                "eigenvectors must have one column per eigenvalue "
                f"({vectors.shape[1]} != {values.size})"
            )
        if blocks.shape != values.shape:
            raise ValueError("block_ids must have one value per eigenvalue")
        if np.any(blocks < 0):
            raise ValueError("block_ids must be non-negative")
        if values.size and np.any(np.diff(values) < -1e-10):
            raise ValueError("eigenvalues must be sorted in nondecreasing order")
        if values.size:
            expected = np.arange(blocks.max() + 1, dtype=np.int64)
            if not np.array_equal(np.unique(blocks), expected):
                raise ValueError("block_ids must be contiguous and start at zero")

        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "block_ids", blocks)

    @property
    def num_nodes(self) -> int:
        return int(self.eigenvectors.shape[0])

    @property
    def k(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def num_blocks(self) -> int:
        return int(self.block_ids.max() + 1) if self.k else 0

    @property
    def block_means(self) -> np.ndarray:
        """Mean eigenvalue for each block, ordered by block id."""
        if not self.k:
            return np.empty(0, dtype=np.float64)
        means = np.empty(self.num_blocks, dtype=np.float64)
        for block in range(self.num_blocks):
            means[block] = self.eigenvalues[self.block_ids == block].mean()
        means.setflags(write=False)
        return means

    @property
    def clamped_eigenvalues(self) -> np.ndarray:
        """Per-frequency block means required for a basis-invariant kernel."""
        if not self.k:
            return np.empty(0, dtype=np.float64)
        values = self.block_means[self.block_ids]
        values.setflags(write=False)
        return values

    def block_indices(self, block: int) -> np.ndarray:
        """Return frequency indices belonging to a near-degenerate block."""
        if not 0 <= block < self.num_blocks:
            raise IndexError(f"unknown block {block}")
        return np.flatnonzero(self.block_ids == block)

    def block_projector(self, block: int) -> np.ndarray:
        """Return ``U_B U_B.T``, the basis-invariant projector for one block."""
        indices = self.block_indices(block)
        basis = self.eigenvectors[:, indices]
        return basis @ basis.T

    def as_dict(self) -> dict[str, object]:
        """Make a serializable mapping suitable for an NPZ cache."""
        return {
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "block_ids": self.block_ids,
            "laplacian": self.laplacian,
            "zero_tolerance": self.zero_tolerance,
        }
