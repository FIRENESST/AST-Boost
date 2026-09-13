"""Check the GatedGCN source port against pinned upstream message passing.

The upstream class body is unchanged. Its torch_scatter sum calls are supplied
by PyG's native scatter adapter because torch_scatter is absent in this runtime.
This checks node/edge updates, not the complete GraphGym training runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch_geometric.nn as pyg_nn
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter as pyg_scatter

from ast_boost.experiments.graphgps import GRAPHGPS_REFERENCE_COMMIT, GraphGPSGatedLayer


def scatter(source, index, dim, out, dim_size, reduce):
    if out is not None or reduce != "sum":
        raise ValueError("audit adapter only supports upstream's sum call")
    return pyg_scatter(source, index, dim=dim, dim_size=dim_size, reduce=reduce)


def audit(reference):
    torch.set_num_threads(1)
    relative = "graphgps/layer/gatedgcn_layer.py"
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != GRAPHGPS_REFERENCE_COMMIT:
        raise ValueError("unexpected reference revision")
    source = subprocess.check_output(
        ["git", "-C", str(reference), "show", f"HEAD:{relative}"]
    ).decode("utf-8")
    local_source = (reference / relative).read_text(encoding="utf-8")
    if source.replace("\r\n", "\n") != local_source.replace("\r\n", "\n"):
        raise ValueError("reference source differs from the pinned commit")
    classes = [node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "GatedGCNLayer"]
    module = types.ModuleType("ast_upstream_gated_audit")
    module.__file__ = str((reference / relative).resolve())
    module.__dict__.update(
        torch=torch, nn=nn, F=F, pyg_nn=pyg_nn, scatter=scatter,
        register=types.SimpleNamespace(act_dict={"relu": nn.ReLU}),
    )
    sys.modules[module.__name__] = module
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), module.__file__, "exec"), module.__dict__
    )
    checks = []
    try:
        for training in (False, True):
            torch.manual_seed(43)
            port = GraphGPSGatedLayer(12, 3, dropout=0, attention_dropout=0)
            port = port.double().train(training)
            original = module.GatedGCNLayer(12, 12, dropout=0, residual=True)
            original = original.double().train(training)
            weights = {}
            for name, value in port.state_dict().items():
                key = name.replace("gated_node_norm.", "bn_node_x.").replace(
                    "gated_edge_norm.", "bn_edge_e."
                )
                if key in original.state_dict():
                    weights[key] = value
            original.load_state_dict(weights)
            x = torch.randn(8, 12, dtype=torch.float64, requires_grad=True)
            edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7],
                                  [1, 0, 2, 1, 4, 3, 5, 4, 6, 5, 7, 6]])
            e = torch.randn(edges.shape[1], 12, dtype=torch.float64, requires_grad=True)
            actual_x, actual_e = port._local(x, edges, e)
            expected = original(types.SimpleNamespace(x=x, edge_attr=e, edge_index=edges))
            torch.testing.assert_close(actual_x, expected.x, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(actual_e, expected.edge_attr, rtol=1e-10, atol=1e-10)
            (actual_x.square().sum() + actual_e.square().sum()).backward(retain_graph=True)
            dx, de = x.grad.clone(), e.grad.clone()
            x.grad, e.grad = None, None
            (expected.x.square().sum() + expected.edge_attr.square().sum()).backward()
            torch.testing.assert_close(dx, x.grad, rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(de, e.grad, rtol=1e-9, atol=1e-9)
            parameters = dict(original.named_parameters())
            max_gradient_error = 0.0
            for name, parameter in port.named_parameters():
                key = name.replace("gated_node_norm.", "bn_node_x.").replace(
                    "gated_edge_norm.", "bn_edge_e."
                )
                if key in parameters:
                    torch.testing.assert_close(parameter.grad, parameters[key].grad,
                                               rtol=1e-9, atol=1e-9)
                    error = float((parameter.grad - parameters[key].grad).abs().max())
                    max_gradient_error = max(max_gradient_error, error)
            for name, value in port.state_dict().items():
                key = name.replace("gated_node_norm.", "bn_node_x.").replace(
                    "gated_edge_norm.", "bn_edge_e."
                )
                if key in original.state_dict():
                    torch.testing.assert_close(value, original.state_dict()[key])
            checks.append({
                "mode": "train" if training else "eval",
                "node_output_max_abs_error": float((actual_x - expected.x).detach().abs().max()),
                "edge_output_max_abs_error": float(
                    (actual_e - expected.edge_attr).detach().abs().max()
                ),
                "node_gradient_max_abs_error": float((dx - x.grad).abs().max()),
                "edge_gradient_max_abs_error": float((de - e.grad).abs().max()),
                "parameter_gradient_max_abs_error": max_gradient_error,
                "batchnorm_buffers_match": True,
            })
    finally:
        sys.modules.pop(module.__name__, None)
    port_hash = hashlib.sha256(Path(inspect.getfile(GraphGPSGatedLayer)).read_bytes()).hexdigest()
    return {
        "reference_commit": revision,
        "upstream_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "port_source_sha256": port_hash,
        "scope": "CPU float64 local node/edge branch; native PyG sum adapter; not GraphGym",
        "checks": checks,
    }


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
