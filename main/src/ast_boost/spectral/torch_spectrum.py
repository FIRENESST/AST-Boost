"""Reusable tensor spectra for GPU training without repeated host transfers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .fields import first_order_fields, second_order_pairs
from .types import Spectrum

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - preprocessing-only environment
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]


if torch is not None:

    def _canonical_device(device: torch.device | str) -> torch.device:
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        if target.type == "cpu":
            return torch.device("cpu")
        return target

    @dataclass(frozen=True, slots=True)
    class TorchSpectrum:
        """A device-ready spectrum and its invariant fields.

        Prepare these objects once in a dataset cache or collate worker.  Keeping
        them on CUDA avoids all per-step NumPy-to-tensor copies; keeping them in
        pinned CPU memory allows asynchronous ``non_blocking`` batch transfers.
        """

        eigenvalues: Tensor
        clamped_eigenvalues: Tensor
        eigenvectors: Tensor
        block_ids: Tensor
        first_order: Tensor
        pairs: Tensor
        pair_limit: int
        num_blocks: int
        laplacian: str
        zero_tolerance: float

        @property
        def num_nodes(self) -> int:
            return int(self.eigenvectors.shape[0])

        @property
        def k(self) -> int:
            return int(self.eigenvalues.numel())

        @property
        def device(self) -> torch.device:
            return self.eigenvectors.device

        @property
        def dtype(self) -> torch.dtype:
            return self.eigenvectors.dtype

        @property
        def is_pinned(self) -> bool:
            return self.device.type == "cpu" and self.eigenvectors.is_pinned()

        def pairs_for(self, k0_pairs: int) -> Tensor:
            if k0_pairs < 0:
                raise ValueError("k0_pairs must be non-negative")
            if k0_pairs > self.pair_limit:
                raise ValueError(
                    f"TorchSpectrum was prepared with k0_pairs={self.pair_limit}, "
                    f"but the model requested {k0_pairs}"
                )
            if not self.pairs.numel() or k0_pairs == self.pair_limit:
                return self.pairs
            return self.pairs[(self.pairs[:, 0] < k0_pairs) & (self.pairs[:, 1] < k0_pairs)]

        def pin_memory(self) -> TorchSpectrum:
            """Return a pinned CPU copy suitable for asynchronous CUDA transfer."""
            if self.device.type != "cpu" or self.is_pinned:
                return self
            if not torch.cuda.is_available():
                return self
            return TorchSpectrum(
                eigenvalues=self.eigenvalues.pin_memory(),
                clamped_eigenvalues=self.clamped_eigenvalues.pin_memory(),
                eigenvectors=self.eigenvectors.pin_memory(),
                block_ids=self.block_ids.pin_memory(),
                first_order=self.first_order.pin_memory(),
                pairs=self.pairs.pin_memory(),
                pair_limit=self.pair_limit,
                num_blocks=self.num_blocks,
                laplacian=self.laplacian,
                zero_tolerance=self.zero_tolerance,
            )

        def to(
            self,
            device: torch.device | str,
            *,
            dtype: torch.dtype | None = None,
            non_blocking: bool = True,
        ) -> TorchSpectrum:
            """Move floating data and integer metadata together."""
            target = _canonical_device(device)
            target_dtype = dtype or self.dtype
            if self.device == target and self.dtype == target_dtype:
                return self
            return TorchSpectrum(
                eigenvalues=self.eigenvalues.to(
                    device=target, dtype=target_dtype, non_blocking=non_blocking
                ),
                clamped_eigenvalues=self.clamped_eigenvalues.to(
                    device=target, dtype=target_dtype, non_blocking=non_blocking
                ),
                eigenvectors=self.eigenvectors.to(
                    device=target, dtype=target_dtype, non_blocking=non_blocking
                ),
                block_ids=self.block_ids.to(device=target, non_blocking=non_blocking),
                first_order=self.first_order.to(
                    device=target, dtype=target_dtype, non_blocking=non_blocking
                ),
                pairs=self.pairs.to(device=target, non_blocking=non_blocking),
                pair_limit=self.pair_limit,
                num_blocks=self.num_blocks,
                laplacian=self.laplacian,
                zero_tolerance=self.zero_tolerance,
            )

    @dataclass(frozen=True, slots=True)
    class TorchSpectrumBatch:
        """Padded spectra and invariant fields for one reusable graph batch."""

        eigenvalues: Tensor
        clamped_eigenvalues: Tensor
        eigenvectors: Tensor
        frequency_mask: Tensor
        first_order: Tensor
        first_order_mask: Tensor
        second_order: Tensor
        second_order_mask: Tensor
        valid_nodes: Tensor
        node_counts: tuple[int, ...]
        pair_limit: int

        @property
        def batch_size(self) -> int:
            return len(self.node_counts)

        @property
        def max_nodes(self) -> int:
            return int(self.valid_nodes.shape[1])

        @property
        def device(self) -> torch.device:
            return self.eigenvectors.device

        @property
        def dtype(self) -> torch.dtype:
            return self.eigenvectors.dtype

        @property
        def is_pinned(self) -> bool:
            return self.device.type == "cpu" and self.eigenvectors.is_pinned()

        def pin_memory(self) -> TorchSpectrumBatch:
            if self.device.type != "cpu" or self.is_pinned or not torch.cuda.is_available():
                return self
            return TorchSpectrumBatch(
                eigenvalues=self.eigenvalues.pin_memory(),
                clamped_eigenvalues=self.clamped_eigenvalues.pin_memory(),
                eigenvectors=self.eigenvectors.pin_memory(),
                frequency_mask=self.frequency_mask.pin_memory(),
                first_order=self.first_order.pin_memory(),
                first_order_mask=self.first_order_mask.pin_memory(),
                second_order=self.second_order.pin_memory(),
                second_order_mask=self.second_order_mask.pin_memory(),
                valid_nodes=self.valid_nodes.pin_memory(),
                node_counts=self.node_counts,
                pair_limit=self.pair_limit,
            )

        def to(
            self,
            device: torch.device | str,
            *,
            dtype: torch.dtype | None = None,
            non_blocking: bool = True,
        ) -> TorchSpectrumBatch:
            target = _canonical_device(device)
            target_dtype = dtype or self.dtype
            if self.device == target and self.dtype == target_dtype:
                return self

            def move_float(value: Tensor) -> Tensor:
                return value.to(device=target, dtype=target_dtype, non_blocking=non_blocking)

            def move_metadata(value: Tensor) -> Tensor:
                return value.to(device=target, non_blocking=non_blocking)

            return TorchSpectrumBatch(
                eigenvalues=move_float(self.eigenvalues),
                clamped_eigenvalues=move_float(self.clamped_eigenvalues),
                eigenvectors=move_float(self.eigenvectors),
                frequency_mask=move_metadata(self.frequency_mask),
                first_order=move_float(self.first_order),
                first_order_mask=move_metadata(self.first_order_mask),
                second_order=move_float(self.second_order),
                second_order_mask=move_metadata(self.second_order_mask),
                valid_nodes=move_metadata(self.valid_nodes),
                node_counts=self.node_counts,
                pair_limit=self.pair_limit,
            )

    SpectrumInput = Spectrum | TorchSpectrum
    SpectrumBatchInput = Sequence[SpectrumInput] | TorchSpectrumBatch

    def _float_tensor(value: object, dtype: torch.dtype) -> Tensor:
        array = np.array(value, dtype=np.float32 if dtype == torch.float32 else None, copy=True)
        return torch.from_numpy(array).to(dtype=dtype)

    def prepare_spectrum(
        spectrum: Spectrum,
        *,
        k0_pairs: int = 4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> TorchSpectrum:
        """Convert and precompute all train-time spectral tensors exactly once."""
        prepared = TorchSpectrum(
            eigenvalues=_float_tensor(spectrum.eigenvalues, dtype),
            clamped_eigenvalues=_float_tensor(spectrum.clamped_eigenvalues, dtype),
            eigenvectors=_float_tensor(spectrum.eigenvectors, dtype),
            block_ids=torch.from_numpy(np.array(spectrum.block_ids, dtype=np.int64, copy=True)),
            first_order=_float_tensor(first_order_fields(spectrum), dtype),
            pairs=torch.from_numpy(
                np.array(second_order_pairs(spectrum, k0_pairs=k0_pairs), dtype=np.int64, copy=True)
            ),
            pair_limit=k0_pairs,
            num_blocks=spectrum.num_blocks,
            laplacian=spectrum.laplacian,
            zero_tolerance=spectrum.zero_tolerance,
        )
        if pin_memory:
            prepared = prepared.pin_memory()
        return prepared.to(device, dtype=dtype, non_blocking=non_blocking)

    def prepare_spectra(
        spectra: Sequence[Spectrum],
        *,
        k0_pairs: int = 4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> list[TorchSpectrum]:
        """Prepare a collection with one consistent device/dtype policy."""
        return [
            prepare_spectrum(
                spectrum,
                k0_pairs=k0_pairs,
                device=device,
                dtype=dtype,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            for spectrum in spectra
        ]

    def prepare_spectrum_batch(
        spectra: Sequence[SpectrumInput],
        *,
        k0_pairs: int = 4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> TorchSpectrumBatch:
        """Pad spectra once so a full graph batch can run through fused GPU kernels."""
        if k0_pairs < 0:
            raise ValueError("k0_pairs must be non-negative")
        target = _canonical_device(device)
        build_device = target
        if not spectra or not all(
            isinstance(spectrum, TorchSpectrum) and spectrum.device == target
            for spectrum in spectra
        ):
            # Raw NumPy spectra are cheapest to assemble on CPU, followed by one
            # batched transfer instead of many small per-graph CUDA transfers.
            build_device = torch.device("cpu")
        prepared = [
            ensure_torch_spectrum(
                spectrum,
                k0_pairs=k0_pairs,
                device=build_device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            for spectrum in spectra
        ]
        node_counts = tuple(spectrum.num_nodes for spectrum in prepared)
        max_nodes = max(node_counts, default=0)
        max_frequencies = max((spectrum.k for spectrum in prepared), default=0)
        max_first_order = max((spectrum.first_order.shape[0] for spectrum in prepared), default=0)
        pair_lists = [spectrum.pairs_for(k0_pairs) for spectrum in prepared]
        max_second_order = max((pairs.shape[0] for pairs in pair_lists), default=0)
        batch_size = len(prepared)

        eigenvalues = torch.zeros((batch_size, max_frequencies), device=build_device, dtype=dtype)
        clamped_eigenvalues = torch.zeros_like(eigenvalues)
        eigenvectors = torch.zeros(
            (batch_size, max_nodes, max_frequencies), device=build_device, dtype=dtype
        )
        frequency_mask = torch.zeros(
            (batch_size, max_frequencies), device=build_device, dtype=torch.bool
        )
        first_order = torch.zeros(
            (batch_size, max_first_order, max_nodes), device=build_device, dtype=dtype
        )
        first_order_mask = torch.zeros(
            (batch_size, max_first_order), device=build_device, dtype=torch.bool
        )
        second_order = torch.zeros(
            (batch_size, max_second_order, max_nodes), device=build_device, dtype=dtype
        )
        second_order_mask = torch.zeros(
            (batch_size, max_second_order), device=build_device, dtype=torch.bool
        )
        valid_nodes = torch.zeros((batch_size, max_nodes), device=build_device, dtype=torch.bool)
        for index, (spectrum, pairs) in enumerate(zip(prepared, pair_lists, strict=True)):
            n, k = spectrum.num_nodes, spectrum.k
            first_count = spectrum.first_order.shape[0]
            second_count = pairs.shape[0]
            eigenvalues[index, :k] = spectrum.eigenvalues
            clamped_eigenvalues[index, :k] = spectrum.clamped_eigenvalues
            eigenvectors[index, :n, :k] = spectrum.eigenvectors
            frequency_mask[index, :k] = True
            first_order[index, :first_count, :n] = spectrum.first_order
            first_order_mask[index, :first_count] = True
            if second_count:
                vectors = spectrum.eigenvectors
                second_order[index, :second_count, :n] = (
                    vectors[:, pairs[:, 0]] * vectors[:, pairs[:, 1]]
                ).T
                second_order_mask[index, :second_count] = True
            valid_nodes[index, :n] = True

        batch = TorchSpectrumBatch(
            eigenvalues=eigenvalues,
            clamped_eigenvalues=clamped_eigenvalues,
            eigenvectors=eigenvectors,
            frequency_mask=frequency_mask,
            first_order=first_order,
            first_order_mask=first_order_mask,
            second_order=second_order,
            second_order_mask=second_order_mask,
            valid_nodes=valid_nodes,
            node_counts=node_counts,
            pair_limit=k0_pairs,
        )
        if pin_memory:
            batch = batch.pin_memory()
        return batch.to(target, dtype=dtype, non_blocking=non_blocking)

    def ensure_torch_spectrum(
        spectrum: SpectrumInput,
        *,
        k0_pairs: int,
        device: torch.device | str,
        dtype: torch.dtype,
        non_blocking: bool = True,
    ) -> TorchSpectrum:
        """Use an existing tensor cache or provide the backwards-compatible conversion."""
        if isinstance(spectrum, TorchSpectrum):
            if k0_pairs > spectrum.pair_limit:
                raise ValueError(
                    f"prepared pair limit {spectrum.pair_limit} is smaller than model limit "
                    f"{k0_pairs}"
                )
            return spectrum.to(device, dtype=dtype, non_blocking=non_blocking)
        return prepare_spectrum(
            spectrum,
            k0_pairs=k0_pairs,
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )


else:
    SpectrumInput = Spectrum
    SpectrumBatchInput = Sequence[SpectrumInput]

    class TorchSpectrum:  # pragma: no cover - dependency error path
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("TorchSpectrum requires PyTorch; install `pip install -e .`.")

    class TorchSpectrumBatch:  # pragma: no cover - dependency error path
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("TorchSpectrumBatch requires PyTorch; install `pip install -e .`.")

    def prepare_spectrum(*args: object, **kwargs: object) -> object:  # pragma: no cover
        raise ImportError("prepare_spectrum requires PyTorch; install `pip install -e .`.")

    def prepare_spectra(*args: object, **kwargs: object) -> object:  # pragma: no cover
        raise ImportError("prepare_spectra requires PyTorch; install `pip install -e .`.")

    def prepare_spectrum_batch(*args: object, **kwargs: object) -> object:  # pragma: no cover
        raise ImportError("prepare_spectrum_batch requires PyTorch; install `pip install -e .`.")

    def ensure_torch_spectrum(*args: object, **kwargs: object) -> object:  # pragma: no cover
        raise ImportError("ensure_torch_spectrum requires PyTorch; install `pip install -e .`.")
