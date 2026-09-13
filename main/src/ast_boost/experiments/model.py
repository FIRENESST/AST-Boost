"""Matched GPS-style GINE + attention backbone for controlled local ablations.

This standalone runner follows the GPS parallel local/global recipe. It is not
an exact reproduction of the public GraphGym trainer or its published scores.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ast_boost import ASTBoostPE, SpectralKernelBias
from ast_boost.spectral.residual_kernel import GatedFullSpectrumKernel

from .data import PackedGraphBatch
from .graphgps import (
    CategoricalFeatureEncoder,
    GraphGPSGatedLayer,
    GraphGPSLapPE,
    GraphGPSLayer,
    GraphGPSRWSE,
    GraphGPSSignNet,
)

METHODS = ("none", "rwse", "lappe", "signnet_local", "lite", "kern", "full")
REFERENCE_METHODS = (
    "rwse_graphgps",
    "lappe_graphgps",
    "signnet_graphgps",
)
GRAPHGPS_ONLY_METHODS = (*REFERENCE_METHODS, "rwse_kernel_graphgps", "rwse_gated_full_graphgps")
ALL_METHODS = METHODS + GRAPHGPS_ONLY_METHODS
BACKBONES = ("compact", "graphgps")


class MaskedBatchNorm(nn.Module):
    """BatchNorm over real nodes only, without dynamic CUDA boolean compaction."""

    def __init__(self, width: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.eps, self.momentum = eps, momentum

    def forward(self, x: Tensor, valid: Tensor, node_index: Tensor | None = None) -> Tensor:
        if node_index is None:
            node_index = valid.reshape(-1).nonzero().reshape(-1)
        flat = x.reshape(-1, x.shape[-1])
        real = flat.index_select(0, node_index)
        normalized = self.forward_compact(real)
        return torch.zeros_like(flat).index_copy(0, node_index, normalized).reshape_as(x)

    def forward_compact(self, real: Tensor) -> Tensor:
        """The same normalization on an already compact real-node tensor."""
        return F.batch_norm(
            real,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training and real.shape[0] > 1,
            momentum=self.momentum,
            eps=self.eps,
        )


class GPSLayer(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.local = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )
        self.local_norm = MaskedBatchNorm(width)
        self.attn_norm = MaskedBatchNorm(width)
        self.ff_norm = MaskedBatchNorm(width)

    def forward(self, x, edge_index, edge_embedding, valid, bias, node_index=None):
        size, nodes, width = x.shape
        flat = x.reshape(-1, width)
        source, target = edge_index
        message = F.relu(flat[source] + edge_embedding)
        aggregate = torch.zeros_like(flat).index_add_(0, target, message.to(flat.dtype))
        local = self.local((flat + aggregate).reshape_as(x))
        local = self.local_norm(
            x + F.dropout(local, self.dropout, self.training), valid, node_index
        )
        qkv = self.qkv(x).reshape(size, nodes, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Only keys are masked: every real graph has at least one valid key.
        # Padding queries are discarded after attention, so no all--inf rows.
        attention_mask = torch.zeros(size, 1, 1, nodes, device=x.device)
        attention_mask = attention_mask.masked_fill(~valid[:, None, None, :], float("-inf"))
        if bias is not None:
            attention_mask = attention_mask + bias
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attention = self.out(attention.transpose(1, 2).reshape(size, nodes, width))
        global_x = self.attn_norm(
            x + F.dropout(attention, self.dropout, self.training), valid, node_index
        )
        combined = local + global_x
        return self.ff_norm(
            combined + F.dropout(self.ff(combined), self.dropout, self.training), valid, node_index
        )

    def forward_compact(self, x, edge_index, edge_embedding, node_index, shape, attention_mask):
        """Keep GINE/BatchNorm/FF on real nodes; pack only attention Q/K/V.

        Uses exactly the same parameters as the padded reference above. With
        dropout enabled the distribution is unchanged, but node-wise dropout
        consumes fewer random values and is not seed-by-seed identical.
        """
        size, nodes = shape
        width = x.shape[-1]
        source, target = edge_index
        message = F.relu(x.index_select(0, source) + edge_embedding)
        aggregate = torch.zeros_like(x).index_add_(0, target, message.to(x.dtype))
        local = self.local_norm.forward_compact(
            x + F.dropout(self.local(x + aggregate), self.dropout, self.training)
        )
        real_qkv = self.qkv(x)
        qkv = real_qkv.new_zeros(size * nodes, 3 * width).index_copy(0, node_index, real_qkv)
        qkv = qkv.reshape(size, nodes, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attention = attention.transpose(1, 2).reshape(size * nodes, width)
        attention = self.out(attention.index_select(0, node_index))
        global_x = self.attn_norm.forward_compact(
            x + F.dropout(attention, self.dropout, self.training)
        )
        combined = local + global_x
        return self.ff_norm.forward_compact(
            combined + F.dropout(self.ff(combined), self.dropout, self.training)
        )


class GPSRegressor(nn.Module):
    def __init__(
        self,
        method: str,
        *,
        width=64,
        layers=10,
        heads=4,
        pe_dim=16,
        sign_hidden=32,
        sign_layers=2,
        k=8,
        pairs=4,
        rw_steps=20,
        dropout=0.0,
        attention_dropout=0.5,
        kernel_eps=1e-6,
        node_layout="compact",
        field_scaling="none",
        signal_backend="sparse",
        frequency_labels="none",
        kernel_spectrum="pe",
        kernel_diagonal=False,
        backbone="compact",
        node_feature_dims: tuple[int, ...] | None = None,
        edge_feature_dims: tuple[int, ...] | None = None,
        output_dim: int = 1,
        pooling: str = "sum",
        head_type: str = "zinc",
        local_gnn: str = "gine",
        reference_pe_dim: int | None = None,
    ):
        super().__init__()
        if method not in ALL_METHODS:
            raise ValueError(f"unknown method: {method}")
        if width < 4 or heads < 1 or width % heads or layers < 1:
            raise ValueError("positive layers and width divisible by heads are required")
        self.method = method
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        if method in GRAPHGPS_ONLY_METHODS and backbone != "graphgps":
            raise ValueError("GraphGPS reference encoders require backbone='graphgps'")
        if node_layout not in {"compact", "padded"}:
            raise ValueError("node_layout must be compact or padded")
        if backbone == "graphgps" and node_layout != "compact":
            raise ValueError("the GraphGPS backbone uses its native compact-node layout")
        self.node_layout = node_layout
        self.backbone = backbone
        if signal_backend not in {"sparse", "dense"}:
            raise ValueError("signal_backend must be sparse or dense")
        self.signal_backend = signal_backend
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        if pooling not in {"sum", "mean"}:
            raise ValueError("pooling must be sum or mean")
        if head_type not in {"zinc", "linear"}:
            raise ValueError("head_type must be zinc or linear")
        if local_gnn not in {"gine", "gatedgcn"}:
            raise ValueError("local_gnn must be gine or gatedgcn")
        if local_gnn == "gatedgcn" and backbone != "graphgps":
            raise ValueError("CustomGatedGCN requires backbone='graphgps'")
        self.output_dim = output_dim
        self.pooling = pooling
        # Build the common backbone FIRST: paired seeds align shared weights
        # for equal embedding widths, including local SignNet/Kern/Full.
        # Public PE concatenation can change the atom embedding width.
        default_reference_pe_dim = {
            "rwse_graphgps": 28,
            "rwse_kernel_graphgps": 28,
            "rwse_gated_full_graphgps": 28,
            "lappe_graphgps": 8,
            "signnet_graphgps": 8,
        }.get(method, 0)
        reference_pe_dim = (
            default_reference_pe_dim if reference_pe_dim is None else reference_pe_dim
        )
        if method not in GRAPHGPS_ONLY_METHODS and reference_pe_dim:
            raise ValueError("reference_pe_dim only applies to GraphGPS reference encoders")
        if reference_pe_dim >= width:
            raise ValueError("GraphGPS PE dimension must be smaller than backbone width")
        atom_width = width - reference_pe_dim
        self.atom = (
            CategoricalFeatureEncoder(node_feature_dims, atom_width)
            if node_feature_dims is not None
            else nn.Embedding(28, atom_width)
        )
        self.bond = (
            CategoricalFeatureEncoder(edge_feature_dims, width)
            if edge_feature_dims is not None
            else nn.Embedding(4, width)
        )
        layer_type = (
            GraphGPSGatedLayer
            if backbone == "graphgps" and local_gnn == "gatedgcn"
            else GraphGPSLayer
            if backbone == "graphgps"
            else GPSLayer
        )
        self.layers = nn.ModuleList(
            layer_type(width, heads, dropout, attention_dropout) for _ in range(layers)
        )
        self.head = (
            nn.Linear(width, output_dim)
            if head_type == "linear"
            else nn.Sequential(
                nn.Linear(width, width // 2),
                nn.ReLU(),
                nn.Linear(width // 2, width // 4),
                nn.ReLU(),
                nn.Linear(width // 4, output_dim),
            )
        )
        self.pe = None
        self.reference_pe = None
        self.kernel_only = None
        if method in {"lite", "kern", "full", "signnet_local"}:
            self.pe = ASTBoostPE(
                variant="kern" if method == "signnet_local" else method,
                heads=heads,
                pe_dim=pe_dim,
                sign_hidden=sign_hidden,
                sign_layers=sign_layers,
                k0_pairs=pairs,
                token_dim=width,
                kernel_eps=kernel_eps,
                field_scaling=field_scaling,
                frequency_labels=frequency_labels,
                kernel_spectrum=kernel_spectrum,
                kernel_diagonal=kernel_diagonal,
            )
            if method == "signnet_local":
                self.pe.kernel.requires_grad_(False)
        elif method in {"rwse", "lappe"}:
            feature_width = rw_steps if method == "rwse" else k
            self.feature_norm = MaskedBatchNorm(feature_width)
            self.feature_projection = nn.Linear(feature_width, pe_dim)
            self.fusion = nn.Linear(width + pe_dim, width)
        elif method in {"rwse_graphgps", "rwse_kernel_graphgps", "rwse_gated_full_graphgps"}:
            self.reference_pe = GraphGPSRWSE(rw_steps, output_dim=reference_pe_dim)
            if method == "rwse_kernel_graphgps":
                self.kernel_only = SpectralKernelBias(
                    heads=heads, degree=8, eps=kernel_eps
                )
            elif method == "rwse_gated_full_graphgps":
                if kernel_spectrum != "all":
                    raise ValueError("gated full kernel requires kernel_spectrum='all'")
                self.kernel_only = GatedFullSpectrumKernel(heads=heads)
        elif method == "lappe_graphgps":
            self.reference_pe = GraphGPSLapPE(output_dim=reference_pe_dim)
        elif method == "signnet_graphgps":
            self.reference_pe = GraphGPSSignNet(
                frequencies=k, output_dim=reference_pe_dim
            )

    def forward(self, batch: PackedGraphBatch) -> Tensor:
        valid = batch.spectra.valid_nodes
        # Compact valid node IDs once; every normalization reuses this layout.
        node_index = batch.node_index
        if node_index is None:
            node_index = valid.reshape(-1).nonzero().reshape(-1)
        mapping = node_index.new_full((valid.numel(),), -1)
        mapping.index_copy_(0, node_index, torch.arange(len(node_index), device=node_index.device))
        compact_edges = mapping[batch.edge_index]
        x = self.atom(batch.node_types)
        bias = None
        if self.pe is not None:
            if self.signal_backend == "dense" and batch.dense_adjacency is None:
                raise ValueError("dense signal backend requires a dense-signals GraphBank")
            x, bias = self.pe.forward_packed(
                x,
                batch.edge_index,
                batch.spectra,
                compute_bias=self.method != "signnet_local",
                dense_adjacency=batch.dense_adjacency if self.signal_backend == "dense" else None,
            )
        elif self.method in {"rwse", "lappe"}:
            features = batch.rwse if self.method == "rwse" else batch.lappe
            if self.method == "lappe" and self.training:
                # Independent per-graph, per-frequency signs; constant over nodes.
                signs = torch.randint(
                    0, 2, (features.shape[0], 1, features.shape[2]), device=x.device
                )
                features = features * (2 * signs - 1)
            encoded = self.feature_projection(self.feature_norm(features, valid, node_index))
            x = self.fusion(torch.cat((x, encoded), dim=-1))
        elif self.reference_pe is not None:
            if self.method in {"rwse_graphgps", "rwse_kernel_graphgps", "rwse_gated_full_graphgps"}:
                features = batch.rwse.reshape(-1, batch.rwse.shape[-1]).index_select(
                    0, node_index
                )
                encoded = self.reference_pe(features)
            elif self.method == "lappe_graphgps":
                encoded = self.reference_pe(
                    batch.graphgps_eigenvalues,
                    batch.graphgps_eigenvectors,
                    batch.graphgps_frequency_mask,
                    node_index,
                )
            else:
                encoded = self.reference_pe(
                    batch.graphgps_eigenvectors, compact_edges, node_index
                )
            padded_encoded = encoded.new_zeros(valid.numel(), encoded.shape[-1])
            padded_encoded.index_copy_(0, node_index, encoded)
            x = torch.cat((x, padded_encoded.reshape(*valid.shape, -1)), dim=-1)
            if self.method == "rwse_gated_full_graphgps":
                bias = self.kernel_only.forward_padded(
                    batch.spectra.kernel_eigenvalues,
                    batch.spectra.kernel_eigenvectors,
                    batch.spectra.kernel_frequency_mask,
                    valid,
                )
            elif self.kernel_only is not None:
                bias = self.kernel_only.forward_padded(
                    batch.spectra.eigenvalues,
                    batch.spectra.eigenvectors,
                    batch.spectra.frequency_mask,
                    valid,
                )
        x = x.masked_fill(~valid[..., None], 0)
        edge_embedding = self.bond(batch.edge_types)
        if self.node_layout == "padded":
            for layer in self.layers:
                x = layer(x, batch.edge_index, edge_embedding, valid, bias, node_index)
            pool_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            pooled = x.to(pool_dtype).masked_fill(~valid[..., None], 0).sum(1)
        else:
            x = x.reshape(-1, x.shape[-1]).index_select(0, node_index)
            # The kernel is shared across layers: its padding mask and gradient
            # accumulation can also be shared instead of rebuilt ten times.
            attention_mask = torch.zeros(
                valid.shape[0], 1, 1, valid.shape[1], device=x.device
            ).masked_fill(~valid[:, None, None, :], float("-inf"))
            if bias is not None:
                attention_mask = attention_mask + bias
            for layer in self.layers:
                layer_output = layer.forward_compact(
                    x, compact_edges, edge_embedding, node_index, valid.shape, attention_mask
                )
                if isinstance(layer_output, tuple):
                    x, edge_embedding = layer_output
                else:
                    x = layer_output
            # Keep the original padded sum reduction order, not a CUDA atomic
            # scatter sum, to avoid introducing another nondeterministic op.
            pool_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            pooled = x.new_zeros(valid.numel(), x.shape[-1], dtype=pool_dtype)
            pooled = pooled.index_copy(0, node_index, x.to(pool_dtype))
            pooled = pooled.reshape(*valid.shape, -1).sum(1)
        if self.pooling == "mean":
            counts = valid.sum(1, keepdim=True).clamp_min(1).to(pooled.dtype)
            pooled = pooled / counts
        prediction = self.head(pooled)
        return prediction.reshape(-1) if self.output_dim == 1 else prediction

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
