"""An exactly closed, basis-invariant full-spectrum residual for a RWSE baseline."""
import torch
from torch import nn

from .kernel import SpectralKernelBias


class GatedFullSpectrumKernel(SpectralKernelBias):
    """offdiag(g(L)) * tanh(gamma), without variance or size normalization.

    gamma=0 closes the residual exactly. Nonconstant Bernstein coefficients
    exp(-2*j/degree) approximate exp(-lambda) at initialization, allowing a
    gate gradient immediately. The coefficients start learning after the gate
    opens. A constant full-spectrum response and a closed gate would deadlock.
    The signed gate is deliberate: finite sigmoid logits cannot give exact zero.
    """

    def __init__(self, *, heads: int, degree: int = 8):
        if degree < 1:
            raise ValueError("a nonconstant initial kernel requires degree >= 1")
        super().__init__(heads=heads, degree=degree, standardize=False, block_clamp=False)
        del self.alpha
        self.gamma = nn.Parameter(torch.zeros(heads))
        with torch.no_grad():
            self.coefficients.copy_(torch.exp(-torch.linspace(0, 2, degree + 1))[None])

    def forward_padded(self, eigenvalues, eigenvectors, frequency_mask, valid_nodes):
        if any(value is None for value in (eigenvalues, eigenvectors, frequency_mask)):
            raise ValueError("gated residual requires a cached complete spectrum including zero modes")
        if valid_nodes.shape != eigenvectors.shape[:2] or not torch.equal(
            frequency_mask.sum(-1), valid_nodes.sum(-1)
        ):
            raise ValueError("gated residual requires every node's spectral mode")
        raw = self.raw_kernel_padded(eigenvalues, eigenvectors, frequency_mask, complete=True)
        n = valid_nodes.shape[1]
        pairs = valid_nodes[:, :, None] & valid_nodes[:, None, :]
        pairs = pairs & ~torch.eye(n, dtype=torch.bool, device=raw.device)
        raw = raw.masked_fill(~pairs[:, None], 0)
        return self.gamma.tanh()[None, :, None, None] * raw

    def forward(self, *args, **kwargs):
        raise ValueError("use forward_padded with an explicit complete spectrum and node mask")
