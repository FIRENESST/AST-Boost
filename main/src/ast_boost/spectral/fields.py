"""Construction of the first- and second-order graph-signal fields."""

from __future__ import annotations

import numpy as np

from .types import Spectrum


def nontrivial_block_mask(spectrum: Spectrum) -> np.ndarray:
    """Mark modes whose individual columns are not basis-invariant.

    A mode in an ``O(m)`` block with ``m > 1`` cannot safely be used on its own,
    nor can it be multiplied by a mode from another block: that product field
    mixes under a block rotation.  This conservative mask is the v1 guard until
    a full BasisNet-style equivariant block encoder is introduced.
    """
    sizes = np.bincount(spectrum.block_ids, minlength=spectrum.num_blocks)
    return sizes[spectrum.block_ids] > 1 if spectrum.k else np.empty(0, dtype=bool)


def first_order_fields(spectrum: Spectrum) -> np.ndarray:
    """Construct the basis-safe inputs for the absolute SignNet path.

    Singleton blocks contribute their signed eigenvector field ``u_p``.  A
    non-singleton block contributes exactly one invariant projector-diagonal
    field ``diag(P_B) = sum_{p in B} u_p**2``.  The latter is unchanged under
    both individual sign flips and arbitrary ``U_B -> U_B Q`` rotations.
    """
    fields: list[np.ndarray] = []
    for block in range(spectrum.num_blocks):
        indices = spectrum.block_indices(block)
        vectors = spectrum.eigenvectors[:, indices]
        if indices.size == 1:
            fields.append(vectors[:, 0])
        else:
            fields.append(np.square(vectors).sum(axis=1))
    if not fields:
        return np.empty((0, spectrum.num_nodes), dtype=np.float64)
    return np.stack(fields, axis=0)


def second_order_pairs(
    spectrum: Spectrum,
    *,
    k0_pairs: int = 4,
) -> np.ndarray:
    """Choose the low-frequency field pairs used by AST-Full.

    The order is deterministic: ``(0, 0), (0, 1), ...``.  Every pair touching a
    non-singleton near-degenerate block is omitted.  Even a cross-block product
    ``u_p * u_q`` mixes under ``U_B -> U_B Q`` when only ``p`` belongs to ``B``;
    summing a nonlinear SignNet over those fields is not generally O(m)-invariant.
    A diagonal pair survives only for a singleton block.
    """
    if k0_pairs < 0:
        raise ValueError("k0_pairs must be non-negative")
    limit = min(k0_pairs, spectrum.k)
    forbidden = nontrivial_block_mask(spectrum)
    pairs: list[tuple[int, int]] = []
    for p in range(limit):
        for q in range(p, limit):
            if forbidden[p] or forbidden[q]:
                continue
            pairs.append((p, q))
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def second_order_fields(
    spectrum: Spectrum,
    pairs: object | None = None,
    *,
    k0_pairs: int = 4,
) -> np.ndarray:
    """Return ``w_pq = u_p * u_q`` as an array shaped ``(num_fields, N)``."""
    selected = (
        second_order_pairs(spectrum, k0_pairs=k0_pairs) if pairs is None else np.asarray(pairs)
    )
    if selected.size == 0:
        return np.empty((0, spectrum.num_nodes), dtype=np.float64)
    if selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError("pairs must have shape (num_pairs, 2)")
    selected = selected.astype(np.int64, copy=False)
    if selected.min() < 0 or selected.max() >= spectrum.k:
        raise IndexError("a field pair refers to a missing frequency")
    forbidden = nontrivial_block_mask(spectrum)
    # Any product touching an arbitrary direction inside an O(m) block is
    # basis-dependent, including products with a singleton mode outside B.
    invalid = forbidden[selected[:, 0]] | forbidden[selected[:, 1]]
    if np.any(invalid):
        raise ValueError("second-order fields cannot use a mode from a non-singleton block")
    vectors = spectrum.eigenvectors
    return (vectors[:, selected[:, 0]] * vectors[:, selected[:, 1]]).T
