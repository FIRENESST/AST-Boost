"""ZINC's official splits, exact RWSE, and a reusable GPU-resident graph bank."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ast_boost import precompute_spectrum, prepare_spectrum_batch
from ast_boost.spectral.precompute import undirected_adjacency
from ast_boost.spectral.torch_spectrum import TorchSpectrumBatch

CACHE_VERSION = 1
SPECTRAL_KEYS = tuple(
    f.name for f in fields(TorchSpectrumBatch) if f.name not in {"node_counts", "pair_limit"}
)


def random_walk_diagonal(edge_index, n: int, steps: int = 20) -> np.ndarray:
    """diag(P^t), t=1..steps, computed from the full graph (not truncated spectrum)."""
    if n < 1 or steps < 1:
        raise ValueError("nodes and walk steps must be positive")
    adjacency = undirected_adjacency(edge_index, n=n)
    degree = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(adjacency, degree, out=np.zeros_like(adjacency), where=degree > 0)
    power = np.eye(n)
    result = []
    for _ in range(steps):
        power = power @ transition
        result.append(np.diag(power))
    return np.stack(result, axis=-1).astype(np.float32)


@dataclass
class PackedGraphBatch:
    node_types: Tensor
    edge_index: Tensor
    edge_types: Tensor
    targets: Tensor
    rwse: Tensor
    lappe: Tensor
    spectra: TorchSpectrumBatch
    node_index: Tensor | None = None
    dense_adjacency: Tensor | None = None


class GraphBank:
    """Cache graph topology and frozen features once; minibatches use index_select.

    Split-local padding avoids repeated Python per-graph collation, spectra
    copies, and token layout construction. All caches contain tensors/primitives
    only and can be loaded with weights_only=True.
    """

    def __init__(self, payload: dict, device: str | torch.device = "cpu", *, dense_signals=False):
        self.metadata = payload["metadata"]
        self.counts = tuple(payload["node_counts"])
        self.pair_limit = payload["pair_limit"]

        # Shapes are known from CPU cache metadata. Reusing positions and the
        # final occupied extent avoids CUDA nonzero/synchronization and lets a
        # minibatch discard only all-zero padding before its device copies.
        def mask_layout(name):
            rows = payload["tensors"][name].cpu().numpy()
            positions = tuple(np.flatnonzero(row) for row in rows)
            extents = tuple(int(row[-1]) + 1 if len(row) else 0 for row in positions)
            return positions, extents

        self._node_positions, self._node_extents = mask_layout("valid_nodes")
        self._edge_positions, self._edge_extents = mask_layout("edge_mask")
        _, self._frequency_extents = mask_layout("frequency_mask")
        _, self._first_order_extents = mask_layout("first_order_mask")
        _, self._second_order_extents = mask_layout("second_order_mask")
        if tuple(map(len, self._node_positions)) != self.counts:
            raise ValueError("node counts disagree with the cached valid-node mask")
        self.tensors = {key: value.to(device) for key, value in payload["tensors"].items()}
        self.device = self.tensors["node_types"].device
        if dense_signals:
            size, nodes = self.tensors["node_types"].shape
            source, target = self.tensors["edges"].unbind(1)
            adjacency = torch.zeros(size, nodes * nodes, device=self.device)
            adjacency.scatter_add_(1, target * nodes + source, self.tensors["edge_mask"].float())
            self.tensors["dense_adjacency"] = adjacency.reshape(size, nodes, nodes)

    def __len__(self) -> int:
        return len(self.counts)

    def batch(
        self,
        indices,
        *,
        static_layout: bool = True,
        trim_padding: bool = False,
    ) -> PackedGraphBatch:
        ids = [int(i) for i in indices]
        if not ids:
            raise ValueError("cannot construct an empty training batch")
        if min(ids) < 0 or max(ids) >= len(self):
            raise IndexError("graph index out of range")
        selection = torch.tensor(ids, device=self.device)
        tensors = self.tensors
        if trim_padding:
            node_extent = max(self._node_extents[i] for i in ids)
            edge_extent = max(self._edge_extents[i] for i in ids)
            frequency_extent = max(self._frequency_extents[i] for i in ids)
            first_extent = max(self._first_order_extents[i] for i in ids)
            second_extent = max(self._second_order_extents[i] for i in ids)
            tensors = {
                **tensors,
                "eigenvalues": tensors["eigenvalues"][:, :frequency_extent],
                "clamped_eigenvalues": tensors["clamped_eigenvalues"][:, :frequency_extent],
                "eigenvectors": tensors["eigenvectors"][:, :node_extent, :frequency_extent],
                "frequency_mask": tensors["frequency_mask"][:, :frequency_extent],
                "first_order": tensors["first_order"][:, :first_extent, :node_extent],
                "first_order_mask": tensors["first_order_mask"][:, :first_extent],
                "second_order": tensors["second_order"][:, :second_extent, :node_extent],
                "second_order_mask": tensors["second_order_mask"][:, :second_extent],
                "valid_nodes": tensors["valid_nodes"][:, :node_extent],
                "node_types": tensors["node_types"][:, :node_extent],
                "edges": tensors["edges"][:, :, :edge_extent],
                "edge_mask": tensors["edge_mask"][:, :edge_extent],
                "edge_types": tensors["edge_types"][:, :edge_extent],
                "rwse": tensors["rwse"][:, :node_extent],
                "lappe": tensors["lappe"][:, :node_extent],
            }
            if "dense_adjacency" in tensors:
                tensors["dense_adjacency"] = tensors["dense_adjacency"][
                    :, :node_extent, :node_extent
                ]
        selected = {key: value.index_select(0, selection) for key, value in tensors.items()}
        spectra = TorchSpectrumBatch(
            **{key: selected[key] for key in SPECTRAL_KEYS},
            node_counts=tuple(self.counts[i] for i in ids),
            pair_limit=self.pair_limit,
        )
        offsets = torch.arange(len(ids), device=self.device)[:, None, None] * spectra.max_nodes
        edges = selected["edges"] + offsets
        edge_mask = selected["edge_mask"]
        flat_edges = edges.permute(1, 0, 2).reshape(2, -1)
        if static_layout:
            edge_positions = torch.as_tensor(
                np.concatenate(
                    [self._edge_positions[g] + i * edge_mask.shape[1] for i, g in enumerate(ids)]
                ),
                device=self.device,
            )
            node_index = torch.as_tensor(
                np.concatenate(
                    [self._node_positions[g] + i * spectra.max_nodes for i, g in enumerate(ids)]
                ),
                device=self.device,
            )
            edge_index = flat_edges.index_select(1, edge_positions)
            edge_types = selected["edge_types"].reshape(-1).index_select(0, edge_positions)
        else:
            # Retained as an independently checkable legacy layout reference.
            edge_index = flat_edges[:, edge_mask.reshape(-1)]
            edge_types = selected["edge_types"][edge_mask]
            node_index = None
        return PackedGraphBatch(
            node_types=selected["node_types"],
            edge_index=edge_index,
            edge_types=edge_types,
            targets=selected["targets"],
            rwse=selected["rwse"],
            lappe=selected["lappe"],
            spectra=spectra,
            node_index=node_index,
            dense_adjacency=selected.get("dense_adjacency"),
        )


def build_payload(graphs, *, k: int = 8, pairs: int = 4, rw_steps: int = 20) -> dict:
    start = time.perf_counter()
    spectra = []
    digest = hashlib.sha256()
    degenerate_graphs = 0
    for graph in graphs:
        spectrum = precompute_spectrum(graph.edge_index, n=graph.num_nodes, k=k)
        spectra.append(spectrum)
        degenerate_graphs += int(spectrum.num_blocks < spectrum.k)
        for value in (graph.x, graph.edge_index, graph.edge_attr, graph.y):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    prepared = prepare_spectrum_batch(spectra, k0_pairs=pairs)
    size, nodes = len(graphs), prepared.max_nodes
    max_edges = max(graph.edge_index.shape[1] for graph in graphs)
    tensors = {key: getattr(prepared, key) for key in SPECTRAL_KEYS}
    tensors.update(
        node_types=torch.zeros(size, nodes, dtype=torch.long),
        edges=torch.zeros(size, 2, max_edges, dtype=torch.long),
        edge_mask=torch.zeros(size, max_edges, dtype=torch.bool),
        edge_types=torch.zeros(size, max_edges, dtype=torch.long),
        targets=torch.zeros(size),
        rwse=torch.zeros(size, nodes, rw_steps),
        lappe=torch.zeros(size, nodes, k),
    )
    for i, (graph, spectrum) in enumerate(zip(graphs, spectra, strict=True)):
        n, e = graph.num_nodes, graph.edge_index.shape[1]
        tensors["node_types"][i, :n] = graph.x.reshape(-1).long()
        tensors["edges"][i, :, :e] = graph.edge_index
        tensors["edge_mask"][i, :e] = True
        tensors["edge_types"][i, :e] = graph.edge_attr.reshape(-1).long()
        tensors["targets"][i] = graph.y.reshape(())
        tensors["rwse"][i, :n] = torch.from_numpy(
            random_walk_diagonal(graph.edge_index, n, rw_steps)
        )
        retained = min(k, spectrum.k)
        tensors["lappe"][i, :n, :retained] = torch.tensor(
            spectrum.eigenvectors[:, :retained].copy()
        )
    return {
        "tensors": tensors,
        "node_counts": list(prepared.node_counts),
        "pair_limit": pairs,
        "metadata": {
            "version": CACHE_VERSION,
            "k": k,
            "pairs": pairs,
            "rw_steps": rw_steps,
            "graphs": size,
            "dataset_sha256": digest.hexdigest(),
            "graphs_with_retained_degenerate_blocks": degenerate_graphs,
            "max_retained_frequencies": prepared.eigenvalues.shape[1],
            "mean_legal_pairs": prepared.second_order_mask.sum(1).float().mean().item(),
            "precompute_seconds": time.perf_counter() - start,
        },
    }


def load_zinc_banks(
    root: Path, cache: Path, *, device="cpu", k=8, pairs=4, rw_steps=20, dense_signals=False
):
    from torch_geometric.datasets import ZINC

    cache.mkdir(parents=True, exist_ok=True)
    banks = {}
    for split in ("train", "val", "test"):
        path = cache / f"zinc-{split}-k{k}-p{pairs}-rw{rw_steps}-v{CACHE_VERSION}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            dataset = ZINC(root=str(root), subset=True, split=split)
            print(f"Precomputing {split}: {len(dataset)} graphs", flush=True)
            payload = build_payload(list(dataset), k=k, pairs=pairs, rw_steps=rw_steps)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
        expected = {"version": CACHE_VERSION, "k": k, "pairs": pairs, "rw_steps": rw_steps}
        if any(payload["metadata"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"incompatible cache: {path}")
        banks[split] = GraphBank(payload, device, dense_signals=dense_signals)
        print(f"{split}: {banks[split].metadata}", flush=True)
    return banks
