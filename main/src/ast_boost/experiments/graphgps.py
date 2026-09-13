"""Auditable ports of the public GraphGPS ZINC layer and PE baselines.

The equations and hyperparameters follow GraphGPS commit
``28015707cbab7f8ad72bed0ee872d068ea59c94b``.  They live in the standalone
runner so AST variants and public baselines can share one modern PyG runtime;
this module does not claim GraphGym trainer or published-score equivalence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

GRAPHGPS_REFERENCE_COMMIT = "28015707cbab7f8ad72bed0ee872d068ea59c94b"
GRAPHGPS_REFERENCE_URL = "https://github.com/rampasek/GraphGPS"


class GraphGPSLayer(nn.Module):
    """GINE + Transformer layer matching the public GraphGPS data flow."""

    def __init__(self, width: int, heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        try:
            from torch_geometric.nn import GINEConv
        except ImportError as error:  # pragma: no cover - dependency-specific path
            raise ImportError("the GraphGPS backbone requires torch-geometric") from error
        self.heads = heads
        gine_mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.local_model = GINEConv(gine_mlp)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(dropout)
        self.norm_local = nn.BatchNorm1d(width)
        self.norm_attention = nn.BatchNorm1d(width)
        self.ff_linear1 = nn.Linear(width, width * 2)
        self.ff_linear2 = nn.Linear(width * 2, width)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.norm_ff = nn.BatchNorm1d(width)

    def forward_compact(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_embedding: Tensor,
        node_index: Tensor,
        shape: tuple[int, int],
        attention_mask: Tensor,
    ) -> Tensor:
        """Run GraphGPS on real nodes and use padding only for attention."""
        batch_size, max_nodes = shape
        width = x.shape[-1]
        residual = x

        local = self.local_model(x, edge_index, edge_embedding)
        local = self.norm_local(residual + self.dropout_local(local))

        dense = x.new_zeros(batch_size * max_nodes, width).index_copy(0, node_index, x)
        dense = dense.reshape(batch_size, max_nodes, width)
        # MultiheadAttention accepts one additive matrix per graph and head.
        # The caller already folds the key-padding mask into this tensor.
        expanded_mask = attention_mask.expand(
            batch_size, self.heads, max_nodes, max_nodes
        ).reshape(batch_size * self.heads, max_nodes, max_nodes)
        attended = self.self_attention(
            dense,
            dense,
            dense,
            attn_mask=expanded_mask,
            need_weights=False,
        )[0]
        attended = attended.reshape(batch_size * max_nodes, width).index_select(0, node_index)
        attended = self.norm_attention(residual + self.dropout_attention(attended))

        combined = local + attended
        feed_forward = self.ff_linear2(
            self.ff_dropout1(torch.relu(self.ff_linear1(combined)))
        )
        return self.norm_ff(combined + self.ff_dropout2(feed_forward))


class GraphGPSGatedLayer(nn.Module):
    """CustomGatedGCN + Transformer used by the public Peptides configs.

    The local branch follows GraphGPS' ``GatedGCNLayer`` exactly: it updates
    node and edge states, applies its own normalization/dropout/residual, and
    the surrounding GPS layer applies the additional local-branch BatchNorm.
    """

    def __init__(self, width: int, heads: int, dropout: float, attention_dropout: float):
        super().__init__()
        try:
            from torch_geometric.nn import Linear
        except ImportError as error:  # pragma: no cover - dependency-specific path
            raise ImportError("the GraphGPS backbone requires torch-geometric") from error
        self.heads = heads
        self.dropout = dropout
        self.A = Linear(width, width, bias=True)
        self.B = Linear(width, width, bias=True)
        self.C = Linear(width, width, bias=True)
        self.D = Linear(width, width, bias=True)
        self.E = Linear(width, width, bias=True)
        self.gated_node_norm = nn.BatchNorm1d(width)
        self.gated_edge_norm = nn.BatchNorm1d(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=attention_dropout, batch_first=True
        )
        self.norm_local = nn.BatchNorm1d(width)
        self.norm_attention = nn.BatchNorm1d(width)
        self.ff_linear1 = nn.Linear(width, width * 2)
        self.ff_linear2 = nn.Linear(width * 2, width)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.norm_ff = nn.BatchNorm1d(width)

    def _local(self, x: Tensor, edge_index: Tensor, edge_embedding: Tensor):
        source, target = edge_index
        edge_state = (
            self.D(x).index_select(0, target)
            + self.E(x).index_select(0, source)
            + self.C(edge_embedding)
        )
        gates = torch.sigmoid(edge_state)
        weighted = gates * self.B(x).index_select(0, source)
        numerator = torch.zeros_like(x, dtype=weighted.dtype).index_add_(
            0, target, weighted
        )
        denominator = torch.zeros_like(x, dtype=gates.dtype).index_add_(0, target, gates)
        node_state = self.A(x) + numerator / (denominator + 1e-6)
        node_state = torch.relu(self.gated_node_norm(node_state))
        edge_state = torch.relu(self.gated_edge_norm(edge_state))
        node_state = x + torch.dropout(node_state, self.dropout, self.training)
        edge_state = edge_embedding + torch.dropout(edge_state, self.dropout, self.training)
        return node_state, edge_state

    def forward_compact(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_embedding: Tensor,
        node_index: Tensor,
        shape: tuple[int, int],
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, max_nodes = shape
        width = x.shape[-1]
        residual = x

        local, edge_embedding = self._local(x, edge_index, edge_embedding)
        local = self.norm_local(local)

        dense = x.new_zeros(batch_size * max_nodes, width).index_copy(0, node_index, x)
        dense = dense.reshape(batch_size, max_nodes, width)
        expanded_mask = attention_mask.expand(
            batch_size, self.heads, max_nodes, max_nodes
        ).reshape(batch_size * self.heads, max_nodes, max_nodes)
        attended = self.self_attention(
            dense,
            dense,
            dense,
            attn_mask=expanded_mask,
            need_weights=False,
        )[0]
        attended = attended.reshape(batch_size * max_nodes, width).index_select(0, node_index)
        attended = self.norm_attention(residual + torch.dropout(
            attended, self.dropout, self.training
        ))

        combined = local + attended
        feed_forward = self.ff_linear2(
            self.ff_dropout1(torch.relu(self.ff_linear1(combined)))
        )
        output = self.norm_ff(combined + self.ff_dropout2(feed_forward))
        return output, edge_embedding


class CategoricalFeatureEncoder(nn.Module):
    """OGB-style sum of one Xavier-initialized embedding per feature column."""

    def __init__(self, dimensions: tuple[int, ...], output_dim: int):
        super().__init__()
        if not dimensions or min(dimensions) < 1:
            raise ValueError("categorical feature dimensions must be positive")
        self.dimensions = tuple(dimensions)
        self.embeddings = nn.ModuleList(nn.Embedding(size, output_dim) for size in dimensions)
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != len(self.embeddings):
            raise ValueError("categorical feature width differs from configured dimensions")
        encoded = self.embeddings[0](features[..., 0])
        for column, embedding in enumerate(self.embeddings[1:], start=1):
            encoded = encoded + embedding(features[..., column])
        return encoded


class GraphGPSRWSE(nn.Module):
    """Official ZINC RWSE encoder: BatchNorm over 20 steps, then Linear to 28."""

    def __init__(self, steps: int, output_dim: int = 28):
        super().__init__()
        self.output_dim = output_dim
        self.raw_norm = nn.BatchNorm1d(steps)
        self.projection = nn.Linear(steps, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(self.raw_norm(features))


class GraphGPSLapPE(nn.Module):
    """Official ZINC random-sign LapPE DeepSet with eigenvalue input."""

    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(2, 2 * output_dim)
        self.encoder = nn.Sequential(
            nn.ReLU(), nn.Linear(2 * output_dim, output_dim), nn.ReLU()
        )

    def forward(
        self,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        frequency_mask: Tensor,
        node_index: Tensor,
    ) -> Tensor:
        batch_size, max_nodes, frequencies = eigenvectors.shape
        graph_index = node_index // max_nodes
        vectors = eigenvectors.reshape(batch_size * max_nodes, frequencies).index_select(
            0, node_index
        )
        values = eigenvalues.index_select(0, graph_index)
        mask = frequency_mask.index_select(0, graph_index)
        if self.training:
            # This intentionally matches the public GraphGPS implementation:
            # one random sign per frequency column for the whole minibatch.
            signs = torch.where(
                torch.rand(frequencies, device=vectors.device) >= 0.5,
                vectors.new_ones(frequencies),
                -vectors.new_ones(frequencies),
            )
            vectors = vectors * signs
        encoded = self.encoder(self.input_projection(torch.stack((vectors, values), dim=-1)))
        return encoded.masked_fill(~mask[..., None], 0).sum(dim=1)


class _SignNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        *,
        batch_norm: bool,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("SignNet MLP must have at least one layer")
        dimensions = (
            [input_dim, output_dim]
            if layers == 1
            else [input_dim, *([hidden_dim] * (layers - 1)), output_dim]
        )
        self.linears = nn.ModuleList(
            nn.Linear(source, target) for source, target in zip(dimensions, dimensions[1:])
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_dim) for _ in range(max(0, layers - 1))
        ) if batch_norm else None

    def forward(self, x: Tensor) -> Tensor:
        for index, linear in enumerate(self.linears[:-1]):
            x = torch.relu(linear(x))
            if self.norms is not None:
                x = (
                    self.norms[index](x)
                    if x.ndim == 2
                    else self.norms[index](x.transpose(1, 2)).transpose(1, 2)
                )
        return self.linears[-1](x)


class _SignNetGIN(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, layers: int):
        super().__init__()
        try:
            from torch_geometric.nn import GINConv
        except ImportError as error:  # pragma: no cover - dependency-specific path
            raise ImportError("the public SignNet baseline requires torch-geometric") from error
        if layers < 2:
            raise ValueError("the public SignNet GIN requires at least two layers")
        networks = [
            _SignNetMLP(1, hidden_dim, hidden_dim, 2, batch_norm=True),
            *[
                _SignNetMLP(hidden_dim, hidden_dim, hidden_dim, 2, batch_norm=True)
                for _ in range(layers - 2)
            ],
            _SignNetMLP(hidden_dim, hidden_dim, output_dim, 2, batch_norm=True),
        ]
        self.layers = nn.ModuleList(GINConv(network) for network in networks)
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers - 1))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            if index:
                x = self.norms[index - 1](x.transpose(1, 2)).transpose(1, 2)
            x = layer(x, edge_index)
        return x


class GraphGPSSignNet(nn.Module):
    """Public GraphGPS SignNet-MLP ZINC encoder with its published defaults."""

    def __init__(
        self,
        frequencies: int = 8,
        output_dim: int = 8,
        hidden_dim: int = 64,
        phi_output_dim: int = 4,
        phi_layers: int = 8,
        rho_layers: int = 2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.frequencies = frequencies
        self.phi = _SignNetGIN(hidden_dim, phi_output_dim, phi_layers)
        self.rho = _SignNetMLP(
            frequencies * phi_output_dim,
            hidden_dim,
            output_dim,
            rho_layers,
            batch_norm=True,
        )

    def forward(self, eigenvectors: Tensor, edge_index: Tensor, node_index: Tensor) -> Tensor:
        batch_size, max_nodes, frequencies = eigenvectors.shape
        if frequencies != self.frequencies:
            raise ValueError(
                "SignNet eigenvector count differs from its configured frequency count"
            )
        vectors = eigenvectors.reshape(batch_size * max_nodes, frequencies).index_select(
            0, node_index
        )
        signals = vectors.transpose(0, 1).unsqueeze(-1)
        invariant = self.phi(signals, edge_index) + self.phi(-signals, edge_index)
        return self.rho(invariant.transpose(0, 1).reshape(vectors.shape[0], -1))
