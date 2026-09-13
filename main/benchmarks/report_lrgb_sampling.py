"""Export checked, completed Peptides-struct sampling comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_study(directory):
    manifest = read_json(directory / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "protocol_sha256"}
    protocol_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("manifest protocol hash is invalid")
    rows = read_json(directory / "results.json")
    expected = {(method, seed) for method in manifest["methods"] for seed in manifest["seeds"]}
    assert len(rows) == len(expected), "study is incomplete or duplicated"
    assert {(row["method"], row["seed"]) for row in rows} == expected
    digest = hashlib.sha256((directory / "source_snapshot.zip").read_bytes()).hexdigest()
    assert digest == manifest["source_snapshot_sha256"], "source archive changed"
    snapshot = directory / "runner_snapshot.py"
    if snapshot.exists():
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["runner_sha256"]
    for row in rows:
        assert row["protocol_sha256"] == manifest["protocol_sha256"]
        assert row["test_evaluations"] == 0
        history = [
            json.loads(line) for line in (
                directory / f'{row["method"]}-seed{row["seed"]}' / "metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert [item["epoch"] for item in history] == list(range(1, manifest["epochs"] + 1))
        best = min(history, key=lambda item: item["val_mae"])
        assert (best["epoch"], best["val_mae"]) == (row["best_epoch"], row["best_val_metric"])
        row["last_val_metric"] = history[-1]["val_mae"]
        row["train_epoch_seconds_range"] = [
            min(item["train_seconds"] for item in history),
            max(item["train_seconds"] for item in history),
        ]
    return {"directory": str(directory), "manifest": manifest, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output stem without extension")
    args = parser.parse_args()
    reference, candidate = read_study(args.reference), read_study(args.candidate)
    old, new = reference["manifest"], candidate["manifest"]
    assert old["dataset"] == new["dataset"] == "Peptides-struct"
    for key in (
        "seeds", "epochs", "warmup_epochs", "batch_size", "sign_hidden", "sign_layers",
        "rw_steps", "train_limit", "val_limit", "precision", "model", "optimizer", "ast_controls",
    ):
        assert old[key] == new[key], f"unmatched control: {key}"
    for split in ("train", "val"):
        old_data = read_json(args.reference / "dataset.json")[split]
        new_data = read_json(args.candidate / "dataset.json")[split]
        assert old_data["dataset_sha256"] == new_data["dataset_sha256"]
    previous = {(row["method"], row["seed"]): row for row in reference["results"]}
    changes = []
    for row in candidate["results"]:
        key = (row["method"], row["seed"])
        if key in previous:
            before, after = previous[key]["best_val_metric"], row["best_val_metric"]
            changes.append({
                "method": row["method"], "seed": row["seed"], "before": before, "after": after,
                "delta_mae": after - before, "relative_reduction": (before - after) / before,
            })
    scores = {(row["method"], row["seed"]): row["best_val_metric"] for row in candidate["results"]}
    contrasts = []
    for seed in new["seeds"]:
        for left, right in (
            ("kern", "signnet_local"), ("full", "kern"), ("kern", "lappe_graphgps"),
        ):
            if (left, seed) in scores and (right, seed) in scores:
                contrasts.append({
                    "target": left, "reference": right, "seed": seed,
                    "delta_mae": scores[left, seed] - scores[right, seed],
                })
    payload = {
        "status": "completed short screening; not a converged multi-seed benchmark",
        "reference": reference, "candidate": candidate,
        "sampling_changes": changes, "within_random_sampler": contrasts,
        "test_evaluations": 0,
    }
    diagnostic_path = args.candidate / "sampling_diagnostic.json"
    if diagnostic_path.exists():
        payload["sampling_diagnostic"] = read_json(diagnostic_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# LRGB 随机采样修正实验", "",
        f'Peptides-struct 完整 train/val，{new["epochs"]} epoch，种子 {new["seeds"]}，'
        f'batch {new["batch_size"]}，{new["precision"]}。测试集评估为 0。', "",
        "历史参考按节点数固定分组、每轮只打乱 batch 顺序；新运行逐轮随机置换图。"
        "两组模型容量、谱设置、优化器和训练轮数相同。", "",
        "## 同一模型的采样比较", "",
    ]
    for row in changes:
        lines.append(
            f'- {row["method"]} / seed {row["seed"]}：最佳验证 MAE '
            f'{row["before"]:.6f} → {row["after"]:.6f}，'
            f'相对下降 {row["relative_reduction"]:.2%}。'
        )
    lines += ["", "## 新采样下的模型比较", ""]
    for row in candidate["results"]:
        lines.append(
            f'- {row["method"]} / seed {row["seed"]}：MAE {row["best_val_metric"]:.6f}，'
            f'epoch {row["best_epoch"]}（最后一轮 MAE {row["last_val_metric"]:.6f}），'
            f'{row["parameters"]:,} 参数，'
            f'训练中位 {row["train_epoch_seconds_median"]:.2f} 秒/epoch，'
            f'峰值分配 {row["peak_allocated_mib"]:.1f} MiB。'
        )
    lines += ["", "配对差值（负值有利于前者）：", ""]
    for row in contrasts:
        lines.append(
            f'- {row["target"]} − {row["reference"]} / seed {row["seed"]}：'
            f'{row["delta_mae"]:+.6f}。'
        )
    lines += [
        "", "## 解释边界", "",
        "采样改变同时影响 batch 组成、BatchNorm 统计和优化轨迹。"
        "跨采样的 MAE 改善不能归因于谱核或二阶场，也未单独识别 BatchNorm 的贡献。"
        "H2/H1 应看新采样内部的配对模型差值，随后补多种子和更长训练。", "",
        "公开 LapPE 是固定 GraphGPS 源码的独立移植，当前仍不是 GraphGym 运行时复现。"
        "本轮没有扩充谱模块；核对角仍关闭，Kern/Full 保持相同特征值标签和谱核设置。", "",
        "AP 修正合并同分预测的阈值贡献；断点恢复以 last.pt 为唯一提交点，"
        "可重建 best.pt 与指标日志。新增目标和谱缓存哈希、运行脚本快照。"
        "这些工程修正未改变本次结构回归的 MAE 定义。", "",
        "同一次运行的 epoch 耗时存在明显阶段性变化。报告保留中位数和范围，"
        "仅作资源记录，不把本轮单次计时当作严格吞吐基准。", "",
        f'精确数据、完整协议及源码哈希见 [{args.output.name}.json]({args.output.name}.json)。',
    ]
    if "sampling_diagnostic" in payload:
        lines += ["", "## 采样分布诊断", "",
                  "以下为种子 42、第 1 轮的训练图节点数统计，不使用验证或测试标签。", ""]
        for name, row in payload["sampling_diagnostic"].items():
            lines.append(
                f'- {name}：各 batch 平均节点数的标准差 '
                f'{row["batch_mean_nodes_std"]:.2f}；batch 内节点数标准差的均值 '
                f'{row["mean_within_batch_nodes_std"]:.2f}；'
                f'填充后节点对数量 / 实际节点对数量 {row["attention_padding_ratio"]:.2f}。'
            )
        lines += ["", "填充比仅描述注意力矩阵规模，不等于整模型耗时倍数。"
                  "本轮没有训练 sortish 消融，不能据此给出其 MAE 结论。"]
    command = (
        ".venv/Scripts/python.exe benchmarks/experiment_lrgb_matrix.py"
        " --dataset Peptides-struct --output runs/lrgb-random-reproduction"
        f' --methods {" ".join(new["methods"])}'
        f' --seeds {" ".join(map(str, new["seeds"]))}'
        f' --epochs {new["epochs"]} --warmup-epochs {new["warmup_epochs"]}'
        f' --batch-size {new["batch_size"]} --sampler random'
        f' --sign-hidden {new["sign_hidden"]} --sign-layers {new["sign_layers"]}'
        f' --precision {new["precision"]}'
    )
    lines += ["", "## 复现实验", "",
              "使用记录对应的源码/运行脚本快照和环境，选择新的输出目录。"
              "同一运行续跑时使用原目录并加 `--resume`；源代码或数据指纹改变会被拒绝。", "",
              "```powershell", command, "```", "",
              "本轮仅有 seed 42；下一步先补配对种子和公开 SignNet 对照，"
              "再决定是否扩展至 200 epoch。继续冻结谱模块结构，不评估测试集。"]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"sampling_changes": changes, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
