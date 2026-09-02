"""Spectrum preprocessing and invariant spectral encoders."""

from .fields import (
    first_order_fields,
    nontrivial_block_mask,
    second_order_fields,
    second_order_pairs,
)
from .kernel import filtered_kernel, standardize_offdiagonal
from .precompute import (
    build_laplacian,
    build_sparse_laplacian,
    load_spectrum,
    precompute_spectrum,
    save_spectrum,
)
from .torch_spectrum import (
    TorchSpectrum,
    TorchSpectrumBatch,
    prepare_spectra,
    prepare_spectrum,
    prepare_spectrum_batch,
)
from .types import Spectrum

__all__ = [
    "Spectrum",
    "build_laplacian",
    "build_sparse_laplacian",
    "precompute_spectrum",
    "save_spectrum",
    "load_spectrum",
    "filtered_kernel",
    "standardize_offdiagonal",
    "first_order_fields",
    "nontrivial_block_mask",
    "second_order_pairs",
    "second_order_fields",
    "TorchSpectrum",
    "TorchSpectrumBatch",
    "prepare_spectrum",
    "prepare_spectra",
    "prepare_spectrum_batch",
]
