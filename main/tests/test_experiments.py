"""Training-integrity tests independent of downloaded data and PyG."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ast_boost import ASTBoostPE, precompute_spectrum, prepare_spectrum_batch
from ast_boost.experiments.data import GraphBank, build_payload, random_walk_diagonal
from ast_boost.experiments.model import METHODS, GPSRegressor, MaskedBatchNorm
from ast_boost.experiments.train import arguments, paired_summary, seed_all


def graphs():
    result = []
    for n in [3, 5, 7]:
        u = torch.arange(n - 1)
        edges = torch.stack((torch.cat((u, u + 1)), torch.cat((u + 1, u))))
        result.append(
            SimpleNamespace(
                x=torch.arange(n)[:, None] % 4,
                num_nodes=n,
                edge_index=edges,
                edge_attr=torch.ones(edges.shape[1]).long(),
                y=torch.tensor([n / 10]),
            )
        )
    return result


def test_exact_rwse_two_node_graph():
    edges = np.array([[0, 1], [1, 0]])
    expected = np.tile([0, 1, 0, 1, 0, 1], (2, 1))
    np.testing.assert_array_equal(random_walk_diagonal(edges, 2, 6), expected)


def test_gpu_bank_selection_does_not_mix_graphs_or_targets():
    bank = GraphBank(build_payload(graphs()))
    batch = bank.batch([2, 0])
    assert batch.spectra.node_counts == (7, 3)
    torch.testing.assert_close(batch.targets, torch.tensor([0.7, 0.3]))
    assert torch.all(batch.edge_index[0] // 7 == batch.edge_index[1] // 7)
    assert batch.spectra.valid_nodes.reshape(-1)[batch.edge_index].all()


@pytest.mark.parametrize("variant", ["lite", "kern", "full"])
def test_packed_fast_path_matches_disjoint_api_and_gradient(variant):
    seed_all(43)
    graph_list = graphs()
    spectra = [precompute_spectrum(g.edge_index, n=g.num_nodes, k=8) for g in graph_list]
    packed = prepare_spectrum_batch(spectra)
    bank = GraphBank(build_payload(graph_list))
    batch = bank.batch([0, 1, 2])
    model = ASTBoostPE(variant=variant, heads=2, pe_dim=4, sign_hidden=8, token_dim=3)
    x = torch.randn(3, 7, 3, requires_grad=True)
    actual, bias = model.forward_packed(x, batch.edge_index, packed)
    offsets = [0, 3, 8]
    flat_edges = torch.cat([g.edge_index + offset for g, offset in zip(graph_list, offsets)], 1)
    membership = torch.repeat_interleave(torch.arange(3), torch.tensor([3, 5, 7]))
    expected, expected_bias, _, _ = model.forward_padded_batch(
        x[packed.valid_nodes], flat_edges, packed, membership, contiguous=True
    )
    torch.testing.assert_close(actual[packed.valid_nodes], expected)
    torch.testing.assert_close(bias, expected_bias)
    grad_a = torch.autograd.grad(actual.square().sum() + bias.square().sum(), x, retain_graph=True)[
        0
    ]
    grad_b = torch.autograd.grad(expected.square().sum() + expected_bias.square().sum(), x)[0]
    torch.testing.assert_close(grad_a, grad_b)


def test_masked_batchnorm_matches_real_node_batchnorm():
    torch.manual_seed(8)
    x = torch.randn(3, 5, 4)
    valid = torch.tensor([[True] * 5, [True, True, False, False, False], [True] * 5])
    module = MaskedBatchNorm(4)
    reference = torch.nn.BatchNorm1d(4)
    expected = reference(x[valid])
    actual = module(x, valid)
    torch.testing.assert_close(actual[valid], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(module.running_mean, reference.running_mean)
    torch.testing.assert_close(module.running_var, reference.running_var)
    assert torch.count_nonzero(actual[~valid]) == 0


@pytest.mark.parametrize("method", METHODS)
def test_end_to_end_predictions_are_batch_independent_and_gradients_finite(method):
    seed_all(91)
    bank = GraphBank(build_payload(graphs()))
    model = GPSRegressor(method, width=16, layers=1, pe_dim=4, sign_hidden=8, attention_dropout=0)
    model.eval()
    expected = model(bank.batch([0, 1, 2]))
    for index in range(3):
        torch.testing.assert_close(
            model(bank.batch([index])), expected[index : index + 1], atol=1e-5, rtol=1e-5
        )
    model.train()
    prediction = model(bank.batch([0, 1, 2]))
    (prediction - torch.tensor([0.3, 0.5, 0.7])).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if method == "full":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.pe.second_order_encoder.rho.parameters()
        )


def test_paired_seeds_have_identical_common_backbone_initialization():
    seed_all(17)
    kern = GPSRegressor("kern", layers=1)
    seed_all(17)
    full = GPSRegressor("full", layers=1)
    for key, value in kern.state_dict().items():
        if not key.startswith("pe."):
            torch.testing.assert_close(value, full.state_dict()[key], rtol=0, atol=0)


def test_full_predictions_really_depend_on_second_order_fields():
    seed_all(91)
    bank = GraphBank(build_payload(graphs()))
    batch = bank.batch([0, 1, 2])
    second = batch.spectra.second_order.detach().clone().requires_grad_(True)
    batch = replace(batch, spectra=replace(batch.spectra, second_order=second))
    model = GPSRegressor(
        "full", width=16, layers=1, pe_dim=4, sign_hidden=8, attention_dropout=0
    ).eval()
    gradient = torch.autograd.grad(model(batch).sum(), second)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    mask = batch.spectra.second_order_mask[:, :, None] & batch.spectra.valid_nodes[:, None, :]
    assert torch.count_nonzero(gradient[~mask]) == 0


def test_full_paired_summary_matches_seeds_not_list_order():
    common = dict(parameters=100, train_epoch_seconds_median=1.0, peak_allocated_mib=20)
    rows = [
        dict(common, method="kern", seed=42, best_val_mae=1.0),
        dict(common, method="kern", seed=43, best_val_mae=2.0),
        dict(common, method="full", seed=43, best_val_mae=1.5),
        dict(common, method="full", seed=42, best_val_mae=1.2),
    ]
    summary = paired_summary(rows)
    assert summary["full_minus_kern_val"]["mean"] == pytest.approx(-0.15)


def test_scheduler_arguments_are_explicit_and_defaults_are_preserved(tmp_path):
    defaults = arguments(["--output", str(tmp_path)])
    cosine = arguments(
        [
            "--output",
            str(tmp_path / "cosine"),
            "--scheduler",
            "cosine",
            "--min-lr",
            "0.00001",
        ]
    )
    assert defaults.scheduler == "plateau" and defaults.plateau_patience == 10
    assert defaults.min_lr == 1e-6
    assert cosine.scheduler == "cosine" and cosine.min_lr == 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_gpu_bf16_optimizer_step_and_no_target_leakage():
    seed_all(73)
    bank = GraphBank(build_payload(graphs()), "cuda")
    batch = bank.batch([0, 1, 2])
    model = GPSRegressor("full", width=16, layers=2, pe_dim=4, sign_hidden=8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.eval()
    torch.testing.assert_close(model(batch), model(replace(batch, targets=batch.targets + 100)))
    model.train()
    before = model.pe.second_order_encoder.rho[0].weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = (model(batch).float() - batch.targets).abs().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before, model.pe.second_order_encoder.rho[0].weight)
