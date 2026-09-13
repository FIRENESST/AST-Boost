"""Scalable Peptides-func/struct preprocessing and validation metrics.

Unlike the small ZINC bank, this cache stores one graph at a time and pads only
the selected minibatch.  This avoids materialising the full Peptides split at
its maximum 444-node size on the GPU.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ast_boost import precompute_spectrum, prepare_spectrum
from ast_boost.spectral.precompute import build_laplacian, build_sparse_laplacian
from ast_boost.spectral.torch_spectrum import (
    TorchSpectrum,
    prepare_spectrum_batch,
)

from .data import PackedGraphBatch, random_walk_diagonal

LRGB_CACHE_VERSION = 2
PEPTIDES_ATOM_FEATURE_DIMS = (119, 4, 12, 12, 10, 6, 6, 2, 2)
PEPTIDES_BOND_FEATURE_DIMS = (5, 6, 2)
PEPTIDES_TARGET_DIMS = {"Peptides-func": 10, "Peptides-struct": 11}


def graphgps_comb_spectrum(
    edge_index, n: int, frequencies: int, *, dense_threshold: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lowest combinatorial-Laplacian modes, including zero, as in GraphGPS."""
    if n < 1 or frequencies < 1:
        raise ValueError("nodes and frequencies must be positive")
    count = min(n, frequencies)
    if n <= dense_threshold or count >= n:
        values, vectors = np.linalg.eigh(build_laplacian(edge_index, n=n, kind="comb"))
    else:
        from scipy.sparse.linalg import eigsh

        matrix = build_sparse_laplacian(edge_index, n=n, kind="comb")
        # A fixed nonconstant start vector makes preprocessing reproducible and
        # avoids choosing an arbitrary RNG state inside ARPACK.
        start = np.linspace(1.0, 2.0, n, dtype=np.float64)
        start /= np.linalg.norm(start)
        values, vectors = eigsh(
            matrix,
            k=count,
            sigma=-1e-5,
            which="LM",
            v0=start,
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    values = np.maximum(values[:count], 0.0).astype(np.float32)
    vectors = vectors[:, :count].astype(np.float32)
    mask = np.ones(count, dtype=np.bool_)
    return values, vectors, mask


def _spectrum_record(spectrum: TorchSpectrum) -> dict:
    return {field.name: getattr(spectrum, field.name) for field in fields(TorchSpectrum)}


def _spectrum_from_record(record: dict) -> TorchSpectrum:
    return TorchSpectrum(**record)


def _update_topology_digest(digest, graph) -> None:
    for value in (graph.x, graph.edge_index, graph.edge_attr):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())


def lrgb_topology_digest(graphs) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        _update_topology_digest(digest, graph)
    return digest.hexdigest()


def build_lrgb_records(
    graphs,
    *,
    k: int = 10,
    pairs: int = 4,
    rw_steps: int = 0,
    dense_threshold: int = 64,
    include_targets: bool = True,
) -> tuple[list[dict], str]:
    """Precompute one cache record per graph and return a content digest."""
    records = []
    digest = hashlib.sha256()
    for graph in graphs:
        n = int(graph.num_nodes)
        spectrum = precompute_spectrum(
            graph.edge_index,
            n=n,
            k=k,
            dense_threshold=dense_threshold,
        )
        prepared = prepare_spectrum(spectrum, k0_pairs=pairs)
        gps_values, gps_vectors, gps_mask = graphgps_comb_spectrum(
            graph.edge_index, n, k, dense_threshold=dense_threshold
        )
        node_types = graph.x.detach().cpu().long().contiguous()
        edge_index = graph.edge_index.detach().cpu().long().contiguous()
        edge_types = graph.edge_attr.detach().cpu().long().contiguous()
        targets = graph.y.detach().cpu().float().reshape(-1).contiguous()
        # Both Peptides tasks use the same molecules and split order.  The
        # digest intentionally covers topology/features only so the expensive
        # spectral shard can be shared after an exact per-task verification.
        _update_topology_digest(digest, graph)
        records.append(
            {
                "node_types": node_types,
                "edge_index": edge_index,
                "edge_types": edge_types,
                **({"targets": targets} if include_targets else {}),
                "rwse": torch.from_numpy(random_walk_diagonal(edge_index, n, rw_steps))
                if rw_steps
                else torch.empty(n, 0),
                "spectrum": _spectrum_record(prepared),
                "graphgps_eigenvalues": torch.from_numpy(gps_values),
                "graphgps_eigenvectors": torch.from_numpy(gps_vectors),
                "graphgps_frequency_mask": torch.from_numpy(gps_mask),
            }
        )
    return records, digest.hexdigest()


class LRGBGraphBank:
    """CPU per-graph records with minibatch-local padding and device transfer."""

    def __init__(self, records: list[dict], metadata: dict, device="cpu"):
        if not records:
            raise ValueError("LRGB graph bank cannot be empty")
        self.records = records
        self.metadata = metadata
        self.counts = tuple(int(record["node_types"].shape[0]) for record in records)
        self.pair_limit = int(metadata["pairs"])
        self.k = int(metadata["k"])
        self.rw_steps = int(metadata["rw_steps"])
        self.device = torch.device(device)

    def __len__(self):
        return len(self.records)

    def batch(self, indices, **_ignored) -> PackedGraphBatch:
        ids = [int(index) for index in indices]
        if not ids:
            raise ValueError("cannot construct an empty training batch")
        if min(ids) < 0 or max(ids) >= len(self):
            raise IndexError("graph index out of range")
        selected = [self.records[index] for index in ids]
        spectra = prepare_spectrum_batch(
            [_spectrum_from_record(record["spectrum"]) for record in selected],
            k0_pairs=self.pair_limit,
            device=self.device,
        )
        batch_size, max_nodes = spectra.valid_nodes.shape
        node_features = selected[0]["node_types"].shape[-1]
        edge_features = selected[0]["edge_types"].shape[-1]
        target_dim = selected[0]["targets"].numel()
        node_types = torch.zeros(batch_size, max_nodes, node_features, dtype=torch.long)
        rwse = torch.zeros(batch_size, max_nodes, self.rw_steps)
        lappe_width = max(record["spectrum"]["eigenvalues"].numel() for record in selected)
        lappe = torch.zeros(batch_size, max_nodes, lappe_width)
        gps_values = torch.zeros(batch_size, self.k)
        gps_vectors = torch.zeros(batch_size, max_nodes, self.k)
        gps_mask = torch.zeros(batch_size, self.k, dtype=torch.bool)
        targets = torch.zeros(batch_size, target_dim)
        edges = []
        edge_types = []
        node_indices = []
        for index, record in enumerate(selected):
            n = record["node_types"].shape[0]
            node_types[index, :n] = record["node_types"]
            rwse[index, :n] = record["rwse"]
            vectors = record["spectrum"]["eigenvectors"]
            lappe[index, :n, : vectors.shape[1]] = vectors
            count = record["graphgps_eigenvalues"].numel()
            gps_values[index, :count] = record["graphgps_eigenvalues"]
            gps_vectors[index, :n, :count] = record["graphgps_eigenvectors"]
            gps_mask[index, :count] = record["graphgps_frequency_mask"]
            targets[index] = record["targets"]
            edges.append(record["edge_index"] + index * max_nodes)
            edge_types.append(record["edge_types"])
            node_indices.append(torch.arange(n) + index * max_nodes)

        def move(value):
            return value.to(self.device, non_blocking=True)

        return PackedGraphBatch(
            node_types=move(node_types),
            edge_index=move(torch.cat(edges, dim=1)),
            edge_types=move(torch.cat(edge_types, dim=0).reshape(-1, edge_features)),
            targets=move(targets),
            rwse=move(rwse),
            lappe=move(lappe),
            graphgps_eigenvalues=move(gps_values),
            graphgps_eigenvectors=move(gps_vectors),
            graphgps_frequency_mask=move(gps_mask),
            spectra=spectra,
            node_index=move(torch.cat(node_indices)),
        )


def _cache_directory(
    cache: Path,
    split: str,
    *,
    k: int,
    pairs: int,
    rw_steps: int,
    limit: int,
) -> Path:
    extent = f"limit{limit}" if limit else "full"
    name = f"{split}-k{k}-p{pairs}-rw{rw_steps}-{extent}-v{LRGB_CACHE_VERSION}"
    return cache / "peptides-common" / name


def load_lrgb_banks(
    root: Path,
    cache: Path,
    *,
    dataset: str,
    device="cpu",
    k=10,
    pairs=4,
    rw_steps=0,
    train_limit=0,
    val_limit=0,
    dense_threshold=64,
    shard_size=128,
    splits=("train", "val"),
):
    """Load Peptides banks. The test split must be explicitly requested."""
    from torch_geometric.datasets import LRGBDataset

    if dataset not in PEPTIDES_TARGET_DIMS:
        raise ValueError("dataset must be Peptides-func or Peptides-struct")
    if any(split not in {"train", "val", "test"} for split in splits):
        raise ValueError("unknown LRGB split")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    cache.mkdir(parents=True, exist_ok=True)
    limits = {"train": train_limit, "val": val_limit, "test": 0}
    banks = {}
    for split in splits:
        dataset_split = LRGBDataset(root=str(root), name=dataset, split=split)
        requested_limit = int(limits[split])
        size = min(len(dataset_split), requested_limit or len(dataset_split))
        directory = _cache_directory(
            cache,
            split,
            k=k,
            pairs=pairs,
            rw_steps=rw_steps,
            limit=requested_limit,
        )
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        shard_digests = []
        cache_digests = []
        target_digest = hashlib.sha256()
        start_time = time.perf_counter()
        for start in range(0, size, shard_size):
            end = min(start + shard_size, size)
            path = directory / f"shard-{start:05d}-{end:05d}.pt"
            expected = {
                "version": LRGB_CACHE_VERSION,
                "dataset_family": "Peptides",
                "split": split,
                "start": start,
                "end": end,
                "k": k,
                "pairs": pairs,
                "rw_steps": rw_steps,
                "dense_threshold": dense_threshold,
            }
            graphs = [dataset_split[index] for index in range(start, end)]
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload.get("metadata") != expected:
                    raise ValueError(f"incompatible cache shard: {path}")
            else:
                print(f"Precomputing {dataset} {split} [{start}:{end}]", flush=True)
                shard, digest = build_lrgb_records(
                    graphs,
                    k=k,
                    pairs=pairs,
                    rw_steps=rw_steps,
                    dense_threshold=dense_threshold,
                    include_targets=False,
                )
                payload = {"metadata": expected, "records": shard, "digest": digest}
                temporary = path.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(path)
            if len(payload["records"]) != end - start:
                raise ValueError(f"incomplete cache shard: {path}")
            current_digest = lrgb_topology_digest(graphs)
            if current_digest != payload["digest"]:
                raise ValueError(
                    f"{dataset} topology does not match the shared Peptides cache: {path}"
                )
            records.extend(
                {
                    **record,
                    "targets": graph.y.detach().cpu().float().reshape(-1).contiguous(),
                }
                for record, graph in zip(payload["records"], graphs, strict=True)
            )
            shard_digests.append(payload["digest"])
            cache_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            for graph in graphs:
                labels = graph.y.detach().cpu().float().contiguous().numpy()
                target_digest.update(str((labels.shape, labels.dtype)).encode())
                target_digest.update(labels.tobytes())
        target_dim = PEPTIDES_TARGET_DIMS[dataset]
        metadata = {
            "version": LRGB_CACHE_VERSION,
            "dataset": dataset,
            "split": split,
            "graphs": size,
            "source_graphs": len(dataset_split),
            "limit": requested_limit,
            "k": k,
            "pairs": pairs,
            "rw_steps": rw_steps,
            "dense_threshold": dense_threshold,
            "target_dim": target_dim,
            "atom_feature_dims": PEPTIDES_ATOM_FEATURE_DIMS,
            "bond_feature_dims": PEPTIDES_BOND_FEATURE_DIMS,
            "dataset_sha256": hashlib.sha256("".join(shard_digests).encode()).hexdigest(),
            "target_sha256": target_digest.hexdigest(),
            "spectral_cache_sha256": hashlib.sha256("".join(cache_digests).encode()).hexdigest(),
            "precompute_or_load_seconds": time.perf_counter() - start_time,
            "test_used": split == "test",
        }
        banks[split] = LRGBGraphBank(records, metadata, device)
        print(f"{split}: {metadata}", flush=True)
    return banks


def average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro AP at distinct score thresholds, including tied predictions."""
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must be matching rank-2 tensors")
    logits, targets = logits.detach().cpu(), targets.detach().cpu()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite predictions in average precision")
    scores = []
    for task in range(targets.shape[1]):
        valid = torch.isfinite(targets[:, task])
        labels = targets[valid, task]
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("average precision requires binary targets")
        positives = int((labels > 0.5).sum())
        if positives == 0:
            continue
        order = torch.argsort(logits[valid, task], descending=True, stable=True)
        sorted_scores = logits[valid, task].index_select(0, order)
        sorted_positive = (labels.index_select(0, order) > 0.5).to(torch.float64)
        ends = torch.cat((sorted_scores[:-1] != sorted_scores[1:], torch.tensor([True])))
        ranks = torch.arange(1, len(labels) + 1, dtype=torch.float64)[ends]
        true_positive = sorted_positive.cumsum(0)[ends]
        increments = torch.diff(true_positive, prepend=torch.zeros(1, dtype=torch.float64))
        scores.append(float(((true_positive / ranks) * increments).sum() / positives))
    if not scores:
        raise ValueError("average precision is undefined without a positive target")
    return float(sum(scores) / len(scores))


def validation_metric(logits: torch.Tensor, targets: torch.Tensor, dataset: str) -> float:
    if dataset == "Peptides-func":
        return average_precision(logits, targets)
    if dataset == "Peptides-struct":
        if logits.shape != targets.shape or logits.ndim != 2:
            raise ValueError("predictions and targets must be matching rank-2 tensors")
        value = torch.abs(logits - targets).mean().item()
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite validation MAE")
        return value
    raise ValueError("unknown Peptides dataset")
