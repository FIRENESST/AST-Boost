"""AST-Boost positional/structural encoding adapter for a GraphGPS backbone.

This module intentionally does *not* implement another Transformer.  It gives a
GPS-compatible caller augmented node tokens plus a dense per-head attention bias
that can be added immediately before attention softmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .kernel import SpectralKernelBias
from .signnet import SignInvariantFieldEncoder
from .torch_spectrum import (
    SpectrumBatchInput,
    SpectrumInput,
    TorchSpectrum,
    TorchSpectrumBatch,
    ensure_torch_spectrum,
    prepare_spectrum_batch,
)

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on training extras
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


Variant = Literal["lite", "kern", "full"]


if torch is not None:

    class ASTBoostPE(nn.Module):
        """Produce AST-Lite, AST-Kern, or AST-Full features and spectral bias.

        ``forward`` operates on one graph.  ``forward_batch`` handles a PyG-style
        disjoint batch and returns a block-diagonal dense bias; the downstream GPS
        attention mask must still prohibit cross-graph attention.
        """

        def __init__(
            self,
            *,
            variant: Variant = "full",
            heads: int = 8,
            pe_dim: int = 32,
            sign_hidden: int = 64,
            sign_layers: int = 2,
            k0_pairs: int = 4,
            kernel_degree: int = 8,
            block_clamp: bool = True,
            standardize_bias: bool = True,
            alpha_init: float = 0.1,
            token_dim: int | None = None,
            share_second_order_psi: bool = True,
            non_blocking: bool = True,
        ) -> None:
            super().__init__()
            if variant not in {"lite", "kern", "full"}:
                raise ValueError("variant must be 'lite', 'kern', or 'full'")
            if pe_dim <= 0:
                raise ValueError("pe_dim must be positive")
            if k0_pairs < 0:
                raise ValueError("k0_pairs must be non-negative")
            self.variant = variant
            self.pe_dim = pe_dim
            self.k0_pairs = k0_pairs
            self.token_dim = token_dim
            self.non_blocking = non_blocking
            if token_dim is not None and token_dim <= 0:
                raise ValueError("token_dim must be positive when provided")
            self.kernel = SpectralKernelBias(
                heads=heads,
                degree=kernel_degree,
                block_clamp=block_clamp,
                standardize=standardize_bias,
                alpha_init=alpha_init,
            )
            if variant == "lite":
                # v0.3's light local baseline: a function of |x_i|/squared
                # coordinates, not a claim of new cross-frequency information.
                self.lite_encoder = nn.Sequential(
                    nn.Linear(2, sign_hidden),
                    nn.GELU(),
                    nn.Linear(sign_hidden, pe_dim),
                )
            else:
                self.first_order_encoder = SignInvariantFieldEncoder(
                    hidden_dim=sign_hidden,
                    out_dim=pe_dim,
                    layers=sign_layers,
                )
            if variant == "full":
                self.second_order_encoder = SignInvariantFieldEncoder(
                    hidden_dim=sign_hidden,
                    out_dim=pe_dim,
                    layers=sign_layers,
                )
                if share_second_order_psi:
                    # Preserve separate first/second readout heads while sharing
                    # the expensive graph-signal encoder across every field.
                    self.second_order_encoder.psi = self.first_order_encoder.psi
            self.token_fusion = (
                nn.Linear(token_dim + self.appended_dim, token_dim)
                if token_dim is not None
                else None
            )

        @property
        def appended_dim(self) -> int:
            """Number of spectral channels appended to every input node token."""
            return self.pe_dim if self.variant in {"lite", "kern"} else 2 * self.pe_dim

        @staticmethod
        def _local_edge_index(edge_index: Tensor, nodes: Tensor, *, total_nodes: int) -> Tensor:
            """Extract/reindex edges induced by a selected graph in a batch."""
            mapping = torch.full((total_nodes,), -1, dtype=torch.long, device=edge_index.device)
            mapping[nodes] = torch.arange(nodes.numel(), device=edge_index.device)
            source, target = edge_index
            keep = (mapping[source] >= 0) & (mapping[target] >= 0)
            return mapping[edge_index[:, keep]]

        def _lite_features(self, spectrum: TorchSpectrum) -> Tensor:
            vectors = spectrum.eigenvectors
            if vectors.shape[1] == 0:
                summary = vectors.new_zeros((vectors.shape[0], 2))
            else:
                squared = vectors.square()
                values = spectrum.clamped_eigenvalues
                summary = torch.stack(
                    [squared.sum(dim=1).sqrt(), (squared * values[None, :]).sum(dim=1)],
                    dim=-1,
                )
            return self.lite_encoder(summary)

        def _second_order_features(
            self,
            spectrum: TorchSpectrum,
            edge_index: Tensor,
        ) -> Tensor:
            fields = self._second_order_fields(spectrum)
            return self.second_order_encoder(fields, edge_index)

        def _second_order_fields(self, spectrum: TorchSpectrum) -> Tensor:
            vectors = spectrum.eigenvectors
            pairs = spectrum.pairs_for(self.k0_pairs)
            if not pairs.numel():
                return vectors.new_zeros((0, spectrum.num_nodes))
            return (vectors[:, pairs[:, 0]] * vectors[:, pairs[:, 1]]).T

        def _first_order_features(
            self,
            spectrum: TorchSpectrum,
            edge_index: Tensor,
        ) -> Tensor:
            return self.first_order_encoder(spectrum.first_order, edge_index)

        def _match_parameter_dtype(self, x: Tensor) -> Tensor:
            """Make direct fp16/bf16 calls safe outside an autocast context."""
            if torch.is_autocast_enabled(x.device.type):
                return x
            parameter_dtype = next(self.parameters()).dtype
            return x if x.dtype == parameter_dtype else x.to(dtype=parameter_dtype)

        def _prepare_spectrum(self, spectrum: SpectrumInput, x: Tensor) -> TorchSpectrum:
            return ensure_torch_spectrum(
                spectrum,
                k0_pairs=self.k0_pairs,
                device=x.device,
                dtype=x.dtype,
                non_blocking=self.non_blocking,
            )

        def _prepare_batch(self, spectra: SpectrumBatchInput, x: Tensor) -> TorchSpectrumBatch:
            if isinstance(spectra, TorchSpectrumBatch):
                if self.k0_pairs != spectra.pair_limit:
                    raise ValueError(
                        f"prepared pair limit {spectra.pair_limit} must match model limit "
                        f"{self.k0_pairs}; rebuild the spectrum batch"
                    )
                return spectra.to(x.device, dtype=x.dtype, non_blocking=self.non_blocking)
            return prepare_spectrum_batch(
                spectra,
                k0_pairs=self.k0_pairs,
                device=x.device,
                dtype=x.dtype,
                non_blocking=self.non_blocking,
            )

        def forward(
            self, x: Tensor, spectrum: SpectrumInput, edge_index: Tensor
        ) -> tuple[Tensor, Tensor]:
            """Return ``(x_with_pe, attn_bias)`` for one graph.

            ``attn_bias`` has shape ``(heads, N, N)`` and is already standardized
            per graph and scaled by its learnable ``alpha``.  Add it to attention
            logits before softmax; do not concatenate it to node tokens.
            """
            if x.ndim != 2:
                raise ValueError("x must have shape (num_nodes, feature_dim)")
            if not x.is_floating_point():
                raise TypeError(
                    "ASTBoostPE expects floating node tokens. Apply the dataset/node encoder "
                    "before attaching spectral features."
                )
            x = self._match_parameter_dtype(x)
            prepared = self._prepare_spectrum(spectrum, x)
            if x.shape[0] != prepared.num_nodes:
                raise ValueError("x and spectrum have different node counts")
            if self.token_dim is not None and x.shape[1] != self.token_dim:
                raise ValueError(
                    f"x has width {x.shape[1]}, but token_dim={self.token_dim}; "
                    "attach ASTBoostPE after the matching node encoder."
                )
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, num_edges)")
            local_edges = edge_index.to(device=x.device, dtype=torch.long)

            if self.variant == "lite":
                node_pe = self._lite_features(prepared)
            else:
                first = self._first_order_features(prepared, local_edges)
                node_pe = first
                if self.variant == "full":
                    second = self._second_order_features(prepared, local_edges)
                    node_pe = torch.cat((first, second), dim=-1)
            bias = self.kernel(prepared, device=x.device, dtype=x.dtype)
            tokens = torch.cat([x, node_pe], dim=-1)
            if self.token_fusion is not None:
                tokens = self.token_fusion(tokens)
            return tokens, bias

        def forward_batch(
            self,
            x: Tensor,
            edge_index: Tensor,
            spectra: Sequence[SpectrumInput],
            batch: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            """Apply the adapter independently and return its required attention mask.

            The boolean third result has shape ``(N, N)`` and is true only for
            node pairs in the same graph.  Callers must mask false positions out
            of attention logits; zero cross-graph bias is not an attention mask.
            """
            if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
                raise ValueError("batch must contain one graph id per node")
            if not x.is_floating_point():
                raise TypeError(
                    "ASTBoostPE expects floating node tokens. Apply the dataset/node encoder "
                    "before attaching spectral features."
                )
            x = self._match_parameter_dtype(x)
            edge_index = edge_index.to(device=x.device, dtype=torch.long)
            batch = batch.to(device=x.device)
            if not spectra and x.shape[0]:
                raise ValueError("non-empty x requires at least one spectrum")
            graph_ids = torch.unique(batch, sorted=True)
            if graph_ids.numel() != len(spectra):
                raise ValueError("len(spectra) must equal the number of graphs in batch")
            output_width = (
                self.token_dim if self.token_dim is not None else x.shape[1] + self.appended_dim
            )
            output: Tensor | None = None
            bias: Tensor | None = None
            for index, graph_id in enumerate(graph_ids):
                nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
                spectrum = spectra[index]
                if nodes.numel() != spectrum.num_nodes:
                    raise ValueError("a spectrum node count does not match its batch graph")
                local_edges = self._local_edge_index(edge_index, nodes, total_nodes=x.shape[0])
                encoded, local_bias = self(x[nodes], spectrum, local_edges)
                if output is None:
                    # Autocast can make module outputs lower precision even when
                    # the input tokens are float32.  Allocate from the computed
                    # tensors so indexed writes preserve that mixed-precision dtype.
                    output = encoded.new_zeros((x.shape[0], output_width))
                    bias = local_bias.new_zeros((self.kernel.heads, x.shape[0], x.shape[0]))
                output[nodes] = encoded
                assert bias is not None
                bias[:, nodes[:, None], nodes[None, :]] = local_bias
            if output is None or bias is None:
                output = x.new_zeros((x.shape[0], output_width))
                bias = x.new_zeros((self.kernel.heads, x.shape[0], x.shape[0]))
            attention_mask = batch[:, None] == batch[None, :]
            return output, bias, attention_mask

        def forward_padded_batch(
            self,
            x: Tensor,
            edge_index: Tensor,
            spectra: SpectrumBatchInput,
            batch: Tensor,
            *,
            contiguous: bool = False,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            """Vectorized GPU batch with ``(B,H,max_N,max_N)`` structural bias.

            This avoids the quadratic ``(sum N)^2`` allocation used by the
            backwards-compatible disjoint bias and runs all graphs through each
            SignNet/kernel layer together. Pass a reusable ``TorchSpectrumBatch``
            to avoid rebuilding padded tensors. Set ``contiguous=True`` for the
            standard ``Batch.from_data_list`` node layout to skip sorting. The
            fourth return value is a ``(B,max_N)`` valid-node mask for pooling or
            dense token packing.
            """
            if batch.ndim != 1 or batch.shape[0] != x.shape[0]:
                raise ValueError("batch must contain one graph id per node")
            if not x.is_floating_point():
                raise TypeError(
                    "ASTBoostPE expects floating node tokens. Apply the dataset/node encoder "
                    "before attaching spectral features."
                )
            x = self._match_parameter_dtype(x)
            edge_index = edge_index.to(device=x.device, dtype=torch.long)
            batch = batch.to(device=x.device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, num_edges)")
            prepared = self._prepare_batch(spectra, x)
            max_nodes = prepared.max_nodes
            output_width = (
                self.token_dim if self.token_dim is not None else x.shape[1] + self.appended_dim
            )
            if prepared.batch_size == 0:
                if x.shape[0]:
                    raise ValueError("non-empty x requires at least one spectrum")
                empty_bias = x.new_zeros((0, self.kernel.heads, 0, 0))
                empty_mask = torch.zeros((0, 0, 0), dtype=torch.bool, device=x.device)
                return x.new_zeros((0, output_width)), empty_bias, empty_mask, prepared.valid_nodes

            expected_counts = torch.as_tensor(
                prepared.node_counts, device=x.device, dtype=torch.long
            )
            if x.shape[0] != sum(prepared.node_counts):
                raise ValueError("a spectrum node count does not match its batch graph")
            starts = torch.cat((expected_counts.new_zeros(1), expected_counts.cumsum(dim=0)[:-1]))
            if contiguous:
                graph_index = torch.repeat_interleave(
                    torch.arange(prepared.batch_size, device=x.device), expected_counts
                )
                local_index = torch.arange(x.shape[0], device=x.device) - starts[graph_index]
                if x.device.type == "cpu":
                    graph_ids, counts = torch.unique_consecutive(batch, return_counts=True)
                    if graph_ids.numel() != prepared.batch_size or not torch.equal(
                        counts, expected_counts
                    ):
                        raise ValueError("contiguous=True requires one contiguous block per graph")
                else:
                    block_ids = batch[starts]
                    valid_layout = torch.all(batch == block_ids[graph_index])
                    if block_ids.numel() > 1:
                        valid_layout = valid_layout & torch.all(block_ids[1:] > block_ids[:-1])
                    torch._assert_async(
                        valid_layout,
                        "contiguous=True requires ascending contiguous graph blocks",
                    )
            else:
                # Stable sorting supports arbitrary graph IDs and node interleaving.
                sorted_nodes = torch.argsort(batch, stable=True)
                sorted_batch = batch[sorted_nodes]
                graph_ids, counts = torch.unique_consecutive(sorted_batch, return_counts=True)
                if graph_ids.numel() != prepared.batch_size:
                    raise ValueError("the spectrum batch must match the graphs in batch")
                if counts.device.type == "cpu":
                    if not torch.equal(counts, expected_counts):
                        raise ValueError("a spectrum node count does not match its batch graph")
                else:
                    torch._assert_async(
                        torch.all(counts == expected_counts),
                        "a spectrum node count does not match its batch graph",
                    )
                rank = torch.empty_like(sorted_nodes)
                rank[sorted_nodes] = torch.arange(x.shape[0], device=x.device)
                graph_index = torch.searchsorted(graph_ids, batch)
                local_index = rank - starts[graph_index]
            padded_index = graph_index * max_nodes + local_index

            padded_x_flat = x.new_zeros((prepared.batch_size * max_nodes, x.shape[1]))
            padded_x_flat[padded_index] = x
            padded_x = padded_x_flat.reshape(prepared.batch_size, max_nodes, x.shape[1])
            source, target = edge_index
            same_graph_edge = graph_index[source] == graph_index[target]
            padded_edges = padded_index[edge_index[:, same_graph_edge]]

            if self.variant == "lite":
                squared = prepared.eigenvectors.square()
                summary = torch.stack(
                    [
                        squared.sum(dim=-1).sqrt(),
                        (squared * prepared.clamped_eigenvalues[:, None, :]).sum(dim=-1),
                    ],
                    dim=-1,
                )
                node_pe = self.lite_encoder(summary)
            else:
                first = self.first_order_encoder.forward_padded(
                    prepared.first_order, padded_edges, prepared.first_order_mask
                )
                node_pe = first
                if self.variant == "full":
                    second = self.second_order_encoder.forward_padded(
                        prepared.second_order,
                        padded_edges,
                        prepared.second_order_mask,
                    )
                    node_pe = torch.cat((first, second), dim=-1)

            kernel_values = (
                prepared.clamped_eigenvalues if self.kernel.block_clamp else prepared.eigenvalues
            )
            padded_bias = self.kernel.forward_padded(
                kernel_values,
                prepared.eigenvectors,
                prepared.frequency_mask,
                prepared.valid_nodes,
            )
            padded_tokens = torch.cat((padded_x, node_pe), dim=-1)
            if self.token_fusion is not None:
                padded_tokens = self.token_fusion(padded_tokens)
            padded_tokens = padded_tokens.masked_fill(~prepared.valid_nodes[:, :, None], 0.0)
            output = padded_tokens.reshape(-1, output_width)[padded_index]
            attention_mask = prepared.valid_nodes[:, :, None] & prepared.valid_nodes[:, None, :]
            return output, padded_bias, attention_mask, prepared.valid_nodes

    def add_attention_bias(
        logits: Tensor,
        bias: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Add bias and optionally enforce a same-graph attention mask."""
        if logits.ndim == 3:
            if logits.shape != bias.shape:
                raise ValueError("3-D logits and bias must have identical shapes")
            output = logits + bias
        elif logits.ndim == 4:
            if bias.ndim == 3 and logits.shape[1:] == bias.shape:
                output = logits + bias.unsqueeze(0)
            elif bias.ndim == 4 and logits.shape == bias.shape:
                output = logits + bias
            else:
                raise ValueError("4-D logits require bias shaped (H,N,N) or matching (B,H,N,N)")
        else:
            raise ValueError("logits must have shape (H,N,N) or (B,H,N,N)")
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape == output.shape[-2:]:
                broadcast_mask = attention_mask
            elif (
                output.ndim == 4
                and attention_mask.ndim == 3
                and attention_mask.shape == (output.shape[0], *output.shape[-2:])
            ):
                broadcast_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("attention_mask must have shape (N,N) or (B,N,N)")
            output = output.masked_fill(
                ~broadcast_mask.to(device=output.device, dtype=torch.bool), float("-inf")
            )
        return output


else:

    class ASTBoostPE:  # pragma: no cover - simple dependency error path
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("ASTBoostPE requires PyTorch; install `pip install -e .`.")

    def add_attention_bias(*args: object, **kwargs: object) -> object:  # pragma: no cover
        raise ImportError("add_attention_bias requires PyTorch; install `pip install -e .`.")
