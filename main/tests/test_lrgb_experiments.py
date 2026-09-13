from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ast_boost.experiments.graphgps import CategoricalFeatureEncoder, GraphGPSGatedLayer
from ast_boost.experiments.lrgb import (
    LRGBGraphBank,
    average_precision,
    build_lrgb_records,
    graphgps_comb_spectrum,
    validation_metric,
)
from ast_boost.experiments.model import GPSRegressor


def peptide_graph(n, targets=3):
    nodes = torch.arange(n - 1)
    edge_index = torch.stack(
        (torch.cat((nodes, nodes + 1)), torch.cat((nodes + 1, nodes)))
    )
    x = torch.zeros(n, 9, dtype=torch.long)
    x[:, 0] = torch.arange(n) % 8
    edge_attr = torch.zeros(edge_index.shape[1], 3, dtype=torch.long)
    edge_attr[:, 0] = 1
    return SimpleNamespace(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.arange(targets, dtype=torch.float32)[None] / 10,
        num_nodes=n,
    )


def bank():
    records, digest = build_lrgb_records([peptide_graph(5), peptide_graph(8)], k=4, pairs=3)
    metadata = {"pairs": 3, "k": 4, "rw_steps": 0, "digest": digest}
    return LRGBGraphBank(records, metadata)


def test_categorical_encoder_sums_columns_and_checks_width():
    encoder = CategoricalFeatureEncoder((3, 4), 5)
    features = torch.tensor([[1, 2], [0, 3]])
    expected = encoder.embeddings[0](features[:, 0]) + encoder.embeddings[1](features[:, 1])
    torch.testing.assert_close(encoder(features), expected)
    with pytest.raises(ValueError, match="feature width"):
        encoder(features[:, :1])


def test_graphgps_comb_spectrum_includes_constant_zero_mode():
    graph = peptide_graph(7)
    values, vectors, mask = graphgps_comb_spectrum(graph.edge_index, 7, 4)
    assert values.shape == (4,) and vectors.shape == (7, 4) and mask.all()
    assert values[0] == pytest.approx(0, abs=1e-6)
    assert np.abs(vectors[:, 0]).std() == pytest.approx(0, abs=1e-6)


def test_lrgb_bank_batches_multicolumn_features_without_global_padding():
    batch = bank().batch([1, 0])
    assert batch.node_types.shape == (2, 8, 9)
    assert batch.edge_types.shape == (22, 3)
    assert batch.targets.shape == (2, 3)
    assert batch.node_index.numel() == 13
    assert batch.spectra.node_counts == (8, 5)


@pytest.mark.parametrize("method", ["lappe_graphgps", "signnet_graphgps", "kern", "full"])
def test_peptides_model_shape_gradient_and_gated_edge_updates(method):
    batch = bank().batch([0, 1])
    model = GPSRegressor(
        method,
        width=24,
        layers=1,
        heads=4,
        pe_dim=8,
        sign_hidden=12,
        sign_layers=2,
        k=4,
        pairs=3,
        backbone="graphgps",
        node_feature_dims=(119, 4, 12, 12, 10, 6, 6, 2, 2),
        edge_feature_dims=(5, 6, 2),
        output_dim=3,
        pooling="mean",
        head_type="linear",
        local_gnn="gatedgcn",
        reference_pe_dim=8 if method in {"lappe_graphgps", "signnet_graphgps"} else 0,
        attention_dropout=0,
    )
    prediction = model(batch)
    assert prediction.shape == (2, 3)
    prediction.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert isinstance(model.layers[0], GraphGPSGatedLayer)
    assert model.layers[0].C.weight.grad.abs().sum() > 0
    # Mixed-size random batches must not change evaluation through padding.
    model.eval()
    graph_bank = bank()
    with torch.inference_mode():
        together = model(graph_bank.batch([0, 1]))
        separate = torch.cat([model(graph_bank.batch([index])) for index in (0, 1)])
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)


def test_average_precision_and_structural_mae_are_exact():
    logits = torch.tensor([[0.9, 0.1], [0.8, 0.7], [0.2, 0.6]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    # task 0: (1 + 2/3)/2; task 1: 1
    assert average_precision(logits, targets) == pytest.approx(11 / 12)
    assert validation_metric(logits, targets, "Peptides-func") == pytest.approx(11 / 12)
    assert validation_metric(logits, targets, "Peptides-struct") == pytest.approx(
        torch.abs(logits - targets).mean().item()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_average_precision_groups_ties_independently_of_input_order(dtype):
    logits = torch.tensor([[0.9], [0.9], [0.2], [0.2]], dtype=dtype)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    for order in ([0, 1, 2, 3], [1, 0, 3, 2], [3, 0, 2, 1]):
        assert average_precision(logits[order], targets[order]) == pytest.approx(0.5)
    # A tie contributes at its score threshold, not at an arbitrary sample rank.
    targets = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    assert average_precision(logits, targets) == pytest.approx(11 / 12)


def test_average_precision_rejects_invalid_scores_and_labels():
    with pytest.raises(FloatingPointError, match="nonfinite"):
        average_precision(torch.tensor([[float("nan")]]), torch.ones(1, 1))
    with pytest.raises(ValueError, match="binary"):
        average_precision(torch.ones(2, 1), torch.tensor([[1.0], [0.2]]))
    with pytest.raises(ValueError, match="positive"):
        average_precision(torch.ones(2, 1), torch.zeros(2, 1))


def test_average_precision_matches_precision_recall_integral_with_quantized_scores():
    generator = torch.Generator().manual_seed(42)
    logits = torch.randint(0, 5, (123, 10), generator=generator).float()
    targets = torch.randint(0, 2, logits.shape, generator=generator).float()
    areas = []
    for task in range(10):
        previous_recall, area = 0.0, 0.0
        for threshold in range(4, -1, -1):
            selected = logits[:, task] >= threshold
            true_positive = float(targets[selected, task].sum())
            recall = true_positive / float(targets[:, task].sum())
            precision = true_positive / int(selected.sum())
            area += (recall - previous_recall) * precision
            previous_recall = recall
        areas.append(area)
    expected = sum(areas) / len(areas)
    assert average_precision(logits, targets) == pytest.approx(expected, abs=1e-12)
