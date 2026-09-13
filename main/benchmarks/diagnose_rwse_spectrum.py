"""Label-free reconstruction of full RWSE from retained normalized-Laplacian modes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ast_boost import build_laplacian


def reconstruction_curve(edge_index, n, steps=20):
    """All low-frequency prefixes, including zero; exact degeneracies kept whole.

    Diagonal similarity makes diag(P**t) equal diag((I-L_sym)**t).
    Isolates follow the repository convention P_ii=0, L_sym_ii=1.
    Errors are global Frobenius errors over nodes and steps, not a ratio at
    individual zero-valued return probabilities (e.g. odd steps on bipartite graphs).
    """
    laplacian = build_laplacian(edge_index, n=n, kind="sym")
    values, vectors = np.linalg.eigh(laplacian)
    transition = np.eye(n) - build_laplacian(edge_index, n=n, kind="rw")
    power = np.eye(n)
    exact = np.empty((n, steps))
    for index in range(steps):
        power = transition @ power
        exact[:, index] = np.diag(power)
    weights = (1 - values[:, None]) ** np.arange(1, steps + 1)[None]
    terms = vectors.square() if hasattr(vectors, "square") else vectors ** 2
    energy = float(np.square(exact).sum())

    def measure(indices):
        predicted = terms[:, indices] @ weights[indices]
        difference = predicted - exact
        return {
            "modes": len(indices), "squared_error": float(np.square(difference).sum()),
            "max_abs_error": float(np.abs(difference).max()),
            "squared_error_by_step": np.square(difference).sum(axis=0).tolist(),
        }

    def prefix(indices, budget):
        count = min(budget, len(indices))
        while count and count < len(indices) and np.isclose(
            values[indices[count]], values[indices[count - 1]], atol=1e-10, rtol=1e-8
        ):
            count += 1
        return indices[:count]

    all_ids = np.arange(n)
    positive = all_ids[values > 1e-8]
    variants = {f"inclusive_k{k}": measure(prefix(all_ids, k)) for k in (4, 8, 16, 32)}
    variants["positive_k8"] = measure(prefix(positive, 8))
    variants["all_positive"] = measure(positive)
    variants["full"] = measure(all_ids)
    needed = {}
    for tolerance in (0.1, 0.05, 0.01):
        needed[str(tolerance)] = next(
            len(prefix(all_ids, k)) for k in range(1, n + 1)
            if measure(prefix(all_ids, k))["squared_error"] <= tolerance ** 2 * energy + 1e-28
        )
    return {"nodes": n, "zero_modes": int((values <= 1e-8).sum()), "energy": energy,
            "energy_by_step": np.square(exact).sum(axis=0).tolist(),
            "variants": variants, "needed_modes": needed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ZINC"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    from torch_geometric.datasets import ZINC
    dataset = ZINC(str(args.root), subset=True, split="train")
    rows, digest = [], hashlib.sha256()
    for index in range(min(args.limit or len(dataset), len(dataset))):
        graph = dataset[index]
        edge = graph.edge_index.numpy()
        digest.update(str((int(graph.num_nodes), edge.shape)).encode())
        digest.update(edge.tobytes())
        rows.append(reconstruction_curve(edge, int(graph.num_nodes)))
        if (index + 1) % 1000 == 0:
            print(f"diagnosed {index + 1} training graphs", flush=True)
    energy = sum(row["energy"] for row in rows)
    variants = {}
    for name in rows[0]["variants"]:
        selected = [row["variants"][name] for row in rows]
        by_step = np.sum([row["squared_error_by_step"] for row in selected], axis=0)
        step_energy = np.sum([row["energy_by_step"] for row in rows], axis=0)
        variants[name] = {
            "relative_frobenius_error": float(np.sqrt(sum(r["squared_error"] for r in selected) / energy)),
            "max_abs_error": max(r["max_abs_error"] for r in selected),
            "mean_modes": float(np.mean([r["modes"] for r in selected])),
            "relative_error_by_step": [float(np.sqrt(e / d)) if d > 0 else None
                                       for e, d in zip(by_step, step_energy, strict=True)],
        }
    report = {
        "dataset": "ZINC subset", "split": "train", "graphs": len(rows), "steps": 20,
        "labels_used": False, "test_evaluations": 0, "topology_sha256": digest.hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "sorted normalized spectrum; exact eigenspace boundary retained; no eigenvalue clamping",
        "interpretation": "reconstruction loss, not proof of task sufficiency, irrecoverability, or MAE causality",
        "variants": variants,
        "needed_modes": {key: {
            "median": float(np.median([r["needed_modes"][key] for r in rows])),
            "p90": float(np.quantile([r["needed_modes"][key] for r in rows], .9)),
            "mean_fraction_of_nodes": float(np.mean([r["needed_modes"][key] / r["nodes"] for r in rows])),
        } for key in rows[0]["needed_modes"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
