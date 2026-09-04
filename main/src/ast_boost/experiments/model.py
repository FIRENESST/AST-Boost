"""Matched GPS-style GINE + attention backbone for controlled local ablations.

This standalone runner follows the GPS parallel local/global recipe. It is not
an exact reproduction of the public GraphGym trainer or its published scores.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ast_boost import ASTBoostPE

from .data import PackedGraphBatch

METHODS = ("none", "rwse", "lappe", "signnet_local", "lite", "kern", "full")


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
        k=8,
        pairs=4,
        rw_steps=20,
        dropout=0.0,
        attention_dropout=0.5,
        kernel_eps=1e-6,
        node_layout="compact",
        field_scaling="none",
        signal_backend="sparse",
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        if width < 4 or heads < 1 or width % heads or layers < 1:
            raise ValueError("positive layers and width divisible by heads are required")
        self.method = method
        if node_layout not in {"compact", "padded"}:
            raise ValueError("node_layout must be compact or padded")
        self.node_layout = node_layout
        if signal_backend not in {"sparse", "dense"}:
            raise ValueError("signal_backend must be sparse or dense")
        self.signal_backend = signal_backend
        # Build the common backbone FIRST: paired seeds give identical initial
        # atom/bond embeddings, GPS layers, and prediction head for every method.
        self.atom = nn.Embedding(28, width)
        self.bond = nn.Embedding(4, width)
        self.layers = nn.ModuleList(
            GPSLayer(width, heads, dropout, attention_dropout) for _ in range(layers)
        )
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, width // 4),
            nn.ReLU(),
            nn.Linear(width // 4, 1),
        )
        self.pe = None
        if method in {"lite", "kern", "full", "signnet_local"}:
            self.pe = ASTBoostPE(
                variant="kern" if method == "signnet_local" else method,
                heads=heads,
                pe_dim=pe_dim,
                sign_hidden=sign_hidden,
                sign_layers=2,
                k0_pairs=pairs,
                token_dim=width,
                kernel_eps=kernel_eps,
                field_scaling=field_scaling,
            )
            if method == "signnet_local":
                self.pe.kernel.requires_grad_(False)
        elif method in {"rwse", "lappe"}:
            feature_width = rw_steps if method == "rwse" else k
            self.feature_norm = MaskedBatchNorm(feature_width)
            self.feature_projection = nn.Linear(feature_width, pe_dim)
            self.fusion = nn.Linear(width + pe_dim, width)

    def forward(self, batch: PackedGraphBatch) -> Tensor:
        valid = batch.spectra.valid_nodes
        # Compact valid node IDs once; every normalization reuses this layout.
        node_index = batch.node_index
        if node_index is None:
            node_index = valid.reshape(-1).nonzero().reshape(-1)
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
        x = x.masked_fill(~valid[..., None], 0)
        edge_embedding = self.bond(batch.edge_types)
        if self.node_layout == "padded":
            for layer in self.layers:
                x = layer(x, batch.edge_index, edge_embedding, valid, bias, node_index)
            pool_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            pooled = x.to(pool_dtype).masked_fill(~valid[..., None], 0).sum(1)
        else:
            x = x.reshape(-1, x.shape[-1]).index_select(0, node_index)
            mapping = node_index.new_full((valid.numel(),), -1)
            mapping.index_copy_(0, node_index, torch.arange(len(node_index), device=x.device))
            compact_edges = mapping[batch.edge_index]
            # The kernel is shared across layers: its padding mask and gradient
            # accumulation can also be shared instead of rebuilt ten times.
            attention_mask = torch.zeros(
                valid.shape[0], 1, 1, valid.shape[1], device=x.device
            ).masked_fill(~valid[:, None, None, :], float("-inf"))
            if bias is not None:
                attention_mask = attention_mask + bias
            for layer in self.layers:
                x = layer.forward_compact(
                    x, compact_edges, edge_embedding, node_index, valid.shape, attention_mask
                )
            # Keep the original padded sum reduction order, not a CUDA atomic
            # scatter sum, to avoid introducing another nondeterministic op.
            pool_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            pooled = x.new_zeros(valid.numel(), x.shape[-1], dtype=pool_dtype)
            pooled = pooled.index_copy(0, node_index, x.to(pool_dtype))
            pooled = pooled.reshape(*valid.shape, -1).sum(1)
        return self.head(pooled).reshape(-1)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
