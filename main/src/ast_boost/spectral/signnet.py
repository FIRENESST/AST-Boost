"""Shared GIN encoders implementing the SignNet symmetrization pattern."""

from __future__ import annotations

from .fields import first_order_fields
from .types import Spectrum

try:  # Neural modules are optional for offline spectrum preprocessing.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - depends on the caller's environment
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:

    class _GINLayer(nn.Module):
        """Small batched GIN layer for a stack of scalar graph signals."""

        def __init__(self, width: int) -> None:
            super().__init__()
            self.eps = nn.Parameter(torch.zeros(()))
            self.mlp = nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
            )

        def forward(self, signals: Tensor, edge_index: Tensor) -> Tensor:
            # signals: (num_fields, num_nodes, width)
            sources, targets = edge_index
            aggregate = torch.zeros_like(signals)
            if sources.numel():
                aggregate.index_add_(1, targets, signals[:, sources, :])
            return self.mlp((1.0 + self.eps) * signals + aggregate)

    class SharedSignalGIN(nn.Module):
        """A two-layer GIN ``psi`` shared by all first- or second-order fields."""

        def __init__(self, *, hidden_dim: int = 64, layers: int = 2) -> None:
            super().__init__()
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if layers <= 0:
                raise ValueError("layers must be positive")
            self.hidden_dim = hidden_dim
            self.input_projection = nn.Linear(1, hidden_dim)
            self.layers = nn.ModuleList(_GINLayer(hidden_dim) for _ in range(layers))

        def forward(self, fields: Tensor, edge_index: Tensor) -> Tensor:
            """Encode ``(num_fields, N)`` fields into ``(num_fields, N, hidden)``."""
            if fields.ndim == 2:
                fields = fields.unsqueeze(-1)
            if fields.ndim != 3 or fields.shape[-1] != 1:
                raise ValueError("fields must have shape (num_fields, num_nodes) or (..., 1)")
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, num_edges)")
            edge_index = edge_index.to(device=fields.device, dtype=torch.long)
            # Tensor-to-bool checks synchronize CUDA.  Keep the defensive range
            # validation on CPU fixtures and trust prevalidated GPU batches.
            if (
                edge_index.device.type == "cpu"
                and edge_index.numel()
                and (edge_index.min() < 0 or edge_index.max() >= fields.shape[1])
            ):
                raise ValueError("edge_index contains a node outside the field tensor")
            hidden = self.input_projection(fields)
            for layer in self.layers:
                hidden = layer(hidden, edge_index)
            return hidden

    class SignInvariantFieldEncoder(nn.Module):
        """Encode graph signals with ``psi(w) + psi(-w)`` then aggregate fields.

        This is the shared implementation for one eigenvector per field
        (AST-Kern) and product fields ``u_p * u_q`` (AST-Full).  It is invariant
        to a global sign change of every individual input field, while ``psi``
        itself is a graph-aware GIN rather than the invalid default pointwise MLP.
        """

        def __init__(
            self,
            *,
            hidden_dim: int = 64,
            out_dim: int = 32,
            layers: int = 2,
        ) -> None:
            super().__init__()
            if out_dim <= 0:
                raise ValueError("out_dim must be positive")
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.psi = SharedSignalGIN(hidden_dim=hidden_dim, layers=layers)
            self.rho = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def encode_invariant(self, fields: Tensor, edge_index: Tensor) -> Tensor:
            """Return per-field invariant hidden states shaped ``(F,N,H)``."""
            if fields.ndim != 2:
                raise ValueError("fields must have shape (num_fields, num_nodes)")
            if fields.shape[0] == 0:
                return fields.new_zeros((0, fields.shape[1], self.hidden_dim))
            # Process both signs in one larger field batch.  GIN never mixes the
            # field dimension, so this is identical but launches each layer once.
            encoded = self.psi(torch.cat((fields, -fields), dim=0), edge_index)
            positive, negative = encoded.chunk(2, dim=0)
            return positive + negative

        def readout(self, invariant: Tensor) -> Tensor:
            """Aggregate invariant field states and apply this encoder's readout."""
            if invariant.ndim != 3 or invariant.shape[-1] != self.hidden_dim:
                raise ValueError("invariant must have shape (num_fields, num_nodes, hidden)")
            if invariant.shape[0] == 0:
                return invariant.new_zeros((invariant.shape[1], self.out_dim))
            return self.rho(invariant.sum(dim=0))

        def forward(self, fields: Tensor, edge_index: Tensor) -> Tensor:
            """Return one invariant node feature vector per node.

            ``fields`` is ``(num_fields, N)``.  An empty set of legal fields is
            represented as ``(0, N)`` and deterministically returns zeros, rather
            than injecting the output bias of ``rho`` into a graph with no fields.
            """
            return self.readout(self.encode_invariant(fields, edge_index))

        def encode_padded_invariant(self, fields: Tensor, edge_index: Tensor) -> Tensor:
            """Return fused padded states shaped ``(F,B,N,H)``."""
            if fields.ndim != 3:
                raise ValueError("fields must have shape (batch, num_fields, max_nodes)")
            batch_size, num_fields, max_nodes = fields.shape
            if num_fields == 0:
                return fields.new_zeros((0, batch_size, max_nodes, self.hidden_dim))
            flat_fields = fields.permute(1, 0, 2).reshape(num_fields, batch_size * max_nodes)
            encoded = self.psi(torch.cat((flat_fields, -flat_fields), dim=0), edge_index)
            positive, negative = encoded.chunk(2, dim=0)
            return (positive + negative).reshape(num_fields, batch_size, max_nodes, self.hidden_dim)

        def readout_padded(self, invariant: Tensor, field_mask: Tensor) -> Tensor:
            """Apply a field mask and this encoder's readout to fused padded states."""
            if invariant.ndim != 4 or invariant.shape[-1] != self.hidden_dim:
                raise ValueError("invariant must have shape (num_fields, batch, nodes, hidden)")
            num_fields, batch_size, max_nodes, _ = invariant.shape
            if field_mask.shape != (batch_size, num_fields):
                raise ValueError("field_mask must have shape (batch, num_fields)")
            if num_fields == 0:
                return invariant.new_zeros((batch_size, max_nodes, self.out_dim))
            valid_fields = field_mask.T[:, :, None, None].to(dtype=invariant.dtype)
            return self.rho((invariant * valid_fields).sum(dim=0))

        def forward_padded(
            self,
            fields: Tensor,
            edge_index: Tensor,
            field_mask: Tensor,
        ) -> Tensor:
            """Encode padded ``(B,F,N)`` fields with one fused disjoint-graph pass."""
            invariant = self.encode_padded_invariant(fields, edge_index)
            return self.readout_padded(invariant, field_mask)

        def first_order(
            self,
            spectrum: Spectrum,
            edge_index: Tensor,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> Tensor:
            """Encode singleton modes and projector-diagonal block fields safely.

            Passing raw columns of ``U`` would be correct only for singleton
            eigenvalues.  ``first_order_fields`` replaces every nontrivial block
            with ``diag(U_B U_B.T)``, so this path remains invariant under an
            arbitrary orthogonal change of basis in that block.
            """
            parameter = next(self.parameters())
            device = device or parameter.device
            dtype = dtype or parameter.dtype
            fields = torch.as_tensor(first_order_fields(spectrum), device=device, dtype=dtype)
            return self(fields, edge_index.to(device=device, dtype=torch.long))


else:

    class SharedSignalGIN:  # pragma: no cover - dependency error path
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("SharedSignalGIN requires PyTorch; install `pip install -e .`.")

    class SignInvariantFieldEncoder:  # pragma: no cover - dependency error path
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "SignInvariantFieldEncoder requires PyTorch; install `pip install -e .`."
            )
