"""Numerically compare the SignNet source port with pinned upstream class bodies.

Only upstream MLP/GIN/GINDeepSigns are loaded; this is not a GraphGym runtime
test. Their class bodies are unchanged, using the same installed PyG GINConv.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv

from ast_boost.experiments.graphgps import GRAPHGPS_REFERENCE_COMMIT, GraphGPSSignNet


def upstream_encoder(reference):
    relative = "graphgps/encoder/signnet_pos_encoder.py"
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != GRAPHGPS_REFERENCE_COMMIT:
        raise ValueError("reference checkout has an unexpected revision")
    source = subprocess.check_output(
        ["git", "-C", str(reference), "show", f"HEAD:{relative}"]
    ).decode("utf-8")
    parsed = ast.parse(source)
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef)
               and node.name in {"MLP", "GIN", "GINDeepSigns"}]
    if {node.name for node in classes} != {"MLP", "GIN", "GINDeepSigns"}:
        raise ValueError("upstream encoder classes are missing")
    namespace = {"torch": torch, "nn": nn, "F": F, "GINConv": GINConv}
    exec(compile(ast.Module(body=classes, type_ignores=[]), relative, "exec"), namespace)
    return namespace["GINDeepSigns"], hashlib.sha256(source.encode()).hexdigest()


def reference_name(name):
    return name.replace("phi.", "enc.", 1).replace(".linears.", ".lins.").replace(
        ".norms.", ".bns."
    )


def audit(reference):
    torch.set_num_threads(1)
    reference_type, source_hash = upstream_encoder(reference)
    checks = []
    for training in (False, True):
        torch.manual_seed(42)
        port = GraphGPSSignNet(frequencies=10, output_dim=16).double().train(training)
        original = reference_type(
            in_channels=1, hidden_channels=64, out_channels=4, num_layers=8,
            k=10, dim_pe=16, rho_num_layers=2, use_bn=True, dropout=0.0,
        ).double().train(training)
        original.load_state_dict({
            reference_name(key): value for key, value in port.state_dict().items()
        })
        vectors = torch.randn(2, 5, 10, dtype=torch.float64, requires_grad=True)
        node_index = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9])
        edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7],
                              [1, 0, 2, 1, 4, 3, 5, 4, 6, 5, 7, 6]])
        actual = port(vectors, edges, node_index)
        compact = vectors.reshape(-1, 10).index_select(0, node_index).unsqueeze(-1)
        expected = original(compact, edges, node_index // 5)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.square().sum().backward(retain_graph=True)
        port_input_gradient = vectors.grad.clone()
        vectors.grad = None
        expected.square().sum().backward()
        torch.testing.assert_close(vectors.grad, port_input_gradient, rtol=1e-9, atol=1e-9)
        original_parameters = dict(original.named_parameters())
        gradient_error = 0.0
        for name, parameter in port.named_parameters():
            reference_gradient = original_parameters[reference_name(name)].grad
            torch.testing.assert_close(parameter.grad, reference_gradient, rtol=1e-9, atol=1e-9)
            gradient_error = max(
                gradient_error, float((parameter.grad - reference_gradient).abs().max())
            )
        for name, value in port.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[reference_name(name)])
        checks.append({
            "mode": "train" if training else "eval",
            "output_max_abs_error": float((actual - expected).detach().abs().max()),
            "input_gradient_max_abs_error": float((vectors.grad - port_input_gradient).abs().max()),
            "parameter_gradient_max_abs_error": gradient_error,
            "batchnorm_buffers_match": True,
        })
    port_hash = hashlib.sha256(Path(inspect.getfile(GraphGPSSignNet)).read_bytes()).hexdigest()
    return {"reference_commit": GRAPHGPS_REFERENCE_COMMIT, "upstream_source_sha256": source_hash,
            "port_source_sha256": port_hash,
            "scope": "CPU float64 encoder parity, including gradients and BN buffers; not GraphGym",
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(".cache/official-graphgps"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
