"""Basis-safe filtered spectral kernels and optional PyTorch bias module."""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np

from .types import Spectrum

Response = np.ndarray | Callable[[np.ndarray], np.ndarray]


def _response_values(response: Response, values: np.ndarray) -> np.ndarray:
    output = response(values) if callable(response) else response
    weights = np.asarray(output, dtype=np.float64).reshape(-1)
    if weights.shape != values.shape:
        raise ValueError(
            "spectral response must return one scalar per selected frequency "
            f"(expected {values.shape}, got {weights.shape})"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("spectral response must be finite")
    return weights


def blockwise_mean(values: object, block_ids: object) -> np.ndarray:
    """Tie a frequency response within every degenerate block."""
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    blocks = np.asarray(block_ids, dtype=np.int64).reshape(-1)
    if weights.shape != blocks.shape:
        raise ValueError("values and block_ids must have the same shape")
    result = weights.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        result[mask] = weights[mask].mean()
    return result


def filtered_kernel(
    spectrum: Spectrum,
    response: Response,
    *,
    block_clamp: bool = True,
) -> np.ndarray:
    """Construct ``U diag(g(lambda)) U.T`` with the v0.3 safety condition.

    When ``block_clamp`` is true (the default), the response is evaluated at the
    mean eigenvalue of each near-degenerate block and then explicitly tied.  The
    latter protects against small numerical differences in a user-supplied
    callable.  Turning it off is useful only for the negative unit test that
    demonstrates the basis-dependence of unequal block weights.
    """
    evaluation_values = spectrum.clamped_eigenvalues if block_clamp else spectrum.eigenvalues
    weights = _response_values(response, evaluation_values)
    if block_clamp:
        weights = blockwise_mean(weights, spectrum.block_ids)
    return (spectrum.eigenvectors * weights[None, :]) @ spectrum.eigenvectors.T


def standardize_offdiagonal(
    kernel: object,
    *,
    eps: float = 1e-8,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Standardize each graph/head over off-diagonal entries only.

    Input may be ``(N, N)`` or ``(H, N, N)``.  By default the diagonal is reset
    to zero because the documented attention bias is a relative-node prior;
    setting ``zero_diagonal=False`` preserves its original values.
    """
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-1] != values.shape[-2]:
        raise ValueError("kernel must have shape (N, N) or (H, N, N)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    original_shape = values.shape
    heads = values[None, ...] if values.ndim == 2 else values
    n = heads.shape[-1]
    output = heads.copy()
    if n <= 1:
        if zero_diagonal:
            output[...] = 0.0
        return output[0] if len(original_shape) == 2 else output

    mask = ~np.eye(n, dtype=bool)
    for head in range(output.shape[0]):
        selected = output[head][mask]
        mean = selected.mean()
        std = selected.std()
        output[head][mask] = (selected - mean) / max(std, eps)
        if zero_diagonal:
            np.fill_diagonal(output[head], 0.0)
    return output[0] if len(original_shape) == 2 else output


def bernstein_basis(values: object, *, degree: int, domain_max: float = 2.0) -> np.ndarray:
    """Evaluate a Bernstein basis on the Laplacian spectrum (normally [0, 2])."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if domain_max <= 0:
        raise ValueError("domain_max must be positive")
    x = np.clip(np.asarray(values, dtype=np.float64) / domain_max, 0.0, 1.0)
    return np.stack(
        [
            comb(degree, index) * x**index * (1.0 - x) ** (degree - index)
            for index in range(degree + 1)
        ],
        axis=-1,
    )


try:  # Keep preprocessing importable on machines that do not yet have PyTorch.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only in a preprocessing-only env
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:
    from .torch_spectrum import SpectrumInput, ensure_torch_spectrum

    def _standardize_torch(kernels: Tensor, mask: Tensor, eps: float) -> Tensor:
        """Masked two-pass population moments without diagonal cancellation.

        A detached common offset cancels algebraically from both the value and
        its gradient. It prevents a large common mean from losing small, real
        off-diagonal differences. No graph/head is coupled to another.
        """
        stats = kernels.float() if kernels.dtype in {torch.float16, torch.bfloat16} else kernels
        if stats.shape[-1] <= 1:
            return stats * 0.0
        count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        anchor = stats.masked_fill(~mask, float("inf")).amin(dim=(-2, -1), keepdim=True)
        anchor = torch.where(mask.any(dim=(-2, -1), keepdim=True), anchor, 0.0).detach()
        shifted = (stats - anchor).masked_fill(~mask, 0.0)
        mean = shifted.sum(dim=(-2, -1), keepdim=True) / count
        centered = (shifted - mean).masked_fill(~mask, 0.0)
        variance = centered.square().sum(dim=(-2, -1), keepdim=True) / count
        return centered * torch.rsqrt(variance.clamp_min(eps**2))

    class SpectralKernelBias(nn.Module):
        """Per-attention-head Bernstein spectral kernels.

        The only learned part is the response ``g_h(lambda)`` and per-head
        ``alpha``.  Eigenvectors are copied from :class:`Spectrum` and never
        become trainable parameters.
        """

        def __init__(
            self,
            *,
            heads: int,
            degree: int = 8,
            block_clamp: bool = True,
            standardize: bool = True,
            alpha_init: float = 0.1,
            domain_max: float = 2.0,
            eps: float = 1e-8,
        ) -> None:
            super().__init__()
            if heads <= 0:
                raise ValueError("heads must be positive")
            if degree < 0:
                raise ValueError("degree must be non-negative")
            if not np.isfinite(domain_max) or domain_max <= 0:
                raise ValueError("domain_max must be finite and positive")
            if not np.isfinite(eps) or eps <= 0:
                raise ValueError("eps must be finite and positive")
            self.heads = heads
            self.degree = degree
            self.block_clamp = block_clamp
            self.standardize = standardize
            self.domain_max = domain_max
            self.eps = eps
            # A constant initial response makes all heads well behaved before
            # task-specific frequency selection begins.
            self.coefficients = nn.Parameter(torch.ones(heads, degree + 1))
            self.alpha = nn.Parameter(torch.full((heads,), float(alpha_init)))
            self.register_buffer(
                "_bernstein_indices",
                torch.arange(degree + 1, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "_binomial_coefficients",
                torch.tensor([comb(degree, index) for index in range(degree + 1)]),
                persistent=False,
            )

        def response(self, eigenvalues: Tensor) -> Tensor:
            """Return responses shaped ``(..., heads, k)``."""
            x = (eigenvalues / self.domain_max).clamp(0.0, 1.0)
            indices = self._bernstein_indices
            coefficients = self._binomial_coefficients.to(dtype=x.dtype)
            basis = (
                coefficients
                * x.unsqueeze(-1).pow(indices)
                * (1.0 - x).unsqueeze(-1).pow(self.degree - indices)
            )
            if eigenvalues.ndim == 1:
                # Keep the small single-graph GEMM in its faster HxD @ DxK
                # orientation; the batched path below handles (...,K,D).
                return self.coefficients.to(dtype=basis.dtype) @ basis.T
            return torch.matmul(basis, self.coefficients.to(dtype=basis.dtype).T).movedim(-1, -2)

        def raw_kernel(
            self,
            spectrum: SpectrumInput,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> Tensor:
            """Build unscaled, unstandardized kernels with shape ``(H, N, N)``."""
            parameter = self.coefficients
            device = device or parameter.device
            dtype = dtype or parameter.dtype
            if dtype in {torch.float16, torch.bfloat16}:
                dtype = torch.float32
            prepared = ensure_torch_spectrum(
                spectrum,
                k0_pairs=0,
                device=device,
                dtype=dtype,
            )
            vectors = prepared.eigenvectors
            eigenvalues = prepared.clamped_eigenvalues if self.block_clamp else prepared.eigenvalues
            # Orthogonal-projector contractions and their subsequent variance
            # are precision-sensitive; AMP remains enabled for the neural PE.
            with torch.autocast(device_type=vectors.device.type, enabled=False):
                weights = self.response(eigenvalues)
                weighted_vectors = vectors.unsqueeze(0) * weights.unsqueeze(1)
                return torch.matmul(weighted_vectors, vectors.T)

        def raw_kernel_padded(
            self,
            eigenvalues: Tensor,
            eigenvectors: Tensor,
            frequency_mask: Tensor,
        ) -> Tensor:
            """Build ``(B,H,N,N)`` kernels for an already padded spectrum batch."""
            if eigenvalues.ndim != 2 or eigenvectors.ndim != 3:
                raise ValueError("padded eigenvalues/eigenvectors must have shapes (B,K)/(B,N,K)")
            if frequency_mask.shape != eigenvalues.shape:
                raise ValueError("frequency_mask must match eigenvalues")
            if (
                eigenvectors.shape[0] != eigenvalues.shape[0]
                or eigenvectors.shape[2] != eigenvalues.shape[1]
            ):
                raise ValueError("padded spectrum dimensions do not agree")
            with torch.autocast(device_type=eigenvectors.device.type, enabled=False):
                if eigenvectors.dtype in {torch.float16, torch.bfloat16}:
                    eigenvectors = eigenvectors.float()
                eigenvalues = eigenvalues.to(dtype=eigenvectors.dtype)
                weights = self.response(eigenvalues)
                weights = weights * frequency_mask[:, None, :].to(dtype=weights.dtype)
                weighted_vectors = eigenvectors[:, None, :, :] * weights[:, :, None, :]
                return torch.matmul(weighted_vectors, eigenvectors.transpose(-1, -2)[:, None, :, :])

        def forward_padded(
            self,
            eigenvalues: Tensor,
            eigenvectors: Tensor,
            frequency_mask: Tensor,
            valid_nodes: Tensor,
        ) -> Tensor:
            """Return standardized/scaled bias for a padded spectrum batch."""
            kernels = self.raw_kernel_padded(eigenvalues, eigenvectors, frequency_mask)
            if valid_nodes.shape != (kernels.shape[0], kernels.shape[-1]):
                raise ValueError("valid_nodes must have shape (batch, max_nodes)")
            pair_mask = valid_nodes[:, :, None] & valid_nodes[:, None, :]
            diagonal = torch.eye(
                kernels.shape[-1], dtype=torch.bool, device=kernels.device
            ).unsqueeze(0)
            offdiagonal_mask = pair_mask & ~diagonal
            if self.standardize:
                kernels = _standardize_torch(kernels, offdiagonal_mask[:, None, :, :], self.eps)
            else:
                kernels = kernels * pair_mask[:, None, :, :].to(dtype=kernels.dtype)
            return self.alpha.to(dtype=kernels.dtype)[None, :, None, None] * kernels

        def forward(
            self,
            spectrum: SpectrumInput,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> Tensor:
            kernels = self.raw_kernel(spectrum, device=device, dtype=dtype)
            if self.standardize:
                n = kernels.shape[-1]
                mask = ~torch.eye(n, dtype=torch.bool, device=kernels.device)
                kernels = _standardize_torch(kernels, mask, self.eps)
            return self.alpha.to(dtype=kernels.dtype)[:, None, None] * kernels


else:

    class SpectralKernelBias:  # pragma: no cover - simple dependency error path
        """Placeholder that gives a clear error in a preprocessing-only environment."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "SpectralKernelBias requires PyTorch. Install AST-Boost's training extras "
                "with `pip install -e .`."
            )
