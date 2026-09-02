"""AST-Boost: invariant spectral encodings for graph transformers.

The package keeps the spectrum outside the training graph.  NumPy utilities are
available without PyTorch; neural modules are imported lazily so a preprocessing
environment does not need the full training stack installed.
"""

from .spectral.precompute import (
    build_laplacian,
    build_sparse_laplacian,
    load_spectrum,
    precompute_spectrum,
    save_spectrum,
)
from .spectral.types import Spectrum

__all__ = [
    "Spectrum",
    "build_laplacian",
    "build_sparse_laplacian",
    "precompute_spectrum",
    "save_spectrum",
    "load_spectrum",
    "TorchSpectrum",
    "TorchSpectrumBatch",
    "prepare_spectrum",
    "prepare_spectra",
    "prepare_spectrum_batch",
]


def __getattr__(name: str):
    """Delay importing optional PyTorch modules until they are actually used."""
    if name == "ASTBoostPE":
        from .spectral.gps_adapter import ASTBoostPE

        return ASTBoostPE
    if name == "SpectralKernelBias":
        from .spectral.kernel import SpectralKernelBias

        return SpectralKernelBias
    if name == "SignInvariantFieldEncoder":
        from .spectral.signnet import SignInvariantFieldEncoder

        return SignInvariantFieldEncoder
    if name in {
        "TorchSpectrum",
        "TorchSpectrumBatch",
        "prepare_spectrum",
        "prepare_spectra",
        "prepare_spectrum_batch",
    }:
        from .spectral import torch_spectrum

        return getattr(torch_spectrum, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
