"""Aggregate only complete, compatible studies from a frozen replication plan."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

if __package__:
    from .report_lrgb_sampling import read_study
else:
    from report_lrgb_sampling import read_study

CONTROL_KEYS = (
    "dataset", "epochs", "warmup_epochs", "batch_size", "sign_hidden", "sign_layers",
    "rw_steps", "train_limit", "val_limit", "precision", "device", "model", "optimizer",
    "ast_controls", "source_sha256", "runner_sha256", "batch_sampler",
    "sortish_window_batches", "environment",
)


def aggregate(plan, studies):
    reference = studies[plan["reference_study"]]
    protocol = reference["manifest"]
    for key in ("dataset", "epochs", "warmup_epochs", "batch_size", "precision"):
        if protocol[key] != plan[key]:
            raise ValueError(f"reference differs from frozen plan: {key}")
    if protocol["batch_sampler"] != plan["sampler"]:
        raise ValueError("reference differs from frozen sampler")
    selected = [
        row for row in reference["results"]
        if row["method"] in plan["reuse"]["methods"] and row["seed"] == plan["reuse"]["seed"]
    ]
    for requested in plan["new_studies"]:
        study = studies[requested["directory"]]
        actual = study["manifest"]
        if actual["methods"] != requested["methods"] or actual["seeds"] != requested["seeds"]:
            raise ValueError("study differs from planned methods or seeds")
        for key in CONTROL_KEYS:
            if actual[key] != protocol[key]:
                raise ValueError(f"incompatible study control: {key}")
        if study["dataset_identity"] != reference["dataset_identity"]:
            raise ValueError("incompatible dataset identity")
        selected.extend(study["results"])
    expected = {(method, seed) for method in plan["methods"] for seed in plan["seeds"]}
    keys = [(row["method"], row["seed"]) for row in selected]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("replication matrix is incomplete or duplicated")
    if any(row["test_evaluations"] != 0 for row in selected):
        raise ValueError("replication plan forbids test evaluations")
    methods = {}
    indexed = {(row["method"], row["seed"]): row for row in selected}
    for method in plan["methods"]:
        rows = [indexed[method, seed] for seed in plan["seeds"]]
        if len({row["parameters"] for row in rows}) != 1:
            raise ValueError("method parameter counts differ between seeds")
        values = [row["best_val_metric"] for row in rows]
        methods[method] = {
            "mean": statistics.mean(values), "sample_std": statistics.stdev(values),
            "parameters": rows[0]["parameters"], "rows": rows,
        }
    contrasts = []
    for comparison in plan["comparisons"]:
        differences = [
            indexed[comparison["target"], seed]["best_val_metric"]
            - indexed[comparison["reference"], seed]["best_val_metric"]
            for seed in plan["seeds"]
        ]
        contrasts.append({
            **comparison, "seeds": plan["seeds"], "differences": differences,
            "mean": statistics.mean(differences), "sample_std": statistics.stdev(differences),
            "target_wins": sum(value < 0 for value in differences),
            "reference_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "interpretation": "negative favors target; descriptive screening, not significance",
        })
    return {"plan": plan, "methods": methods, "contrasts": contrasts, "test_evaluations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    directories = [plan["reference_study"], *[row["directory"] for row in plan["new_studies"]]]
    studies = {}
    for directory in directories:
        path = args.root / directory
        study = read_study(path)
        identity = json.loads((path / "dataset_identity.json").read_text(encoding="utf-8"))
        if set(identity) != {"train", "val"} or any(
            set(values) != {"dataset_sha256", "target_sha256", "spectral_cache_sha256"}
            for values in identity.values()
        ):
            raise ValueError("missing train/val dataset fingerprints")
        study["dataset_identity"] = identity
        studies[directory] = study
    report = aggregate(plan, studies)
    report["evidence"] = {
        directory: {key: study[key] for key in ("manifest", "dataset_identity")}
        for directory, study in studies.items()
    }
    if args.baseline_audit:
        audit = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
        reference = studies[plan["reference_study"]]["manifest"]
        port_hash = next(value for key, value in reference["source_sha256"].items()
                         if key.replace(chr(92), "/").endswith("/experiments/graphgps.py"))
        if audit["port_source_sha256"] != port_hash or (
            audit["reference_commit"] != reference["graphgps_reference"]["commit"]
        ):
            raise ValueError("baseline audit does not cover this experiment's source")
        report["baseline_audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Peptides-struct 三种子复核", "",
        f'完整 train/val；种子 {plan["seeds"]}；每组 {plan["epochs"]} epoch；'
        f'batch {plan["batch_size"]}；随机采样；按最佳验证 MAE 选模。测试集评估为 0。', "",
        "均值后的 ± 为跨种子的样本标准差，不是置信区间。"
        "所有结果均通过源码、运行脚本、训练设置和数据指纹一致性检查。", "",
        "## 全部结果", "",
    ]
    for method, summary in report["methods"].items():
        values = "；".join(
            f'seed {row["seed"]}: {row["best_val_metric"]:.6f}' for row in summary["rows"]
        )
        lines.append(
            f'- {method}：**{summary["mean"]:.6f} ± {summary["sample_std"]:.6f}**；'
            f'{summary["parameters"]:,} 参数。{values}。'
        )
    lines += ["", "## 配对比较", "", "差值定义为前者减后者，负值表示前者更好。", ""]
    for comparison in report["contrasts"]:
        values = ", ".join(f"{value:+.6f}" for value in comparison["differences"])
        lines.append(
            f'- {comparison["target"]} − {comparison["reference"]}：'
            f'均值 **{comparison["mean"]:+.6f}**，'
            f'前者在 {comparison["target_wins"]}/{len(plan["seeds"])} 个种子上更好。'
            f'逐种子差值 [{values}]。'
        )
    lines += ["", "## 结论与边界", ""]
    for comparison in report["contrasts"]:
        direction = "更低" if comparison["mean"] < 0 else "更高或持平"
        lines.append(
            f'- {comparison["target"]} 的平均 MAE 相对 {comparison["reference"]} {direction}；'
            "是否值得扩展训练，应结合上面的逐种子方向和资源开销判断。"
        )
    lines += [
        "", "35 epoch、三个种子仍属于筛查。没有按分数提前终止、增加模块或调整学习率；"
        "不能把描述性均值差当作统计显著性，也不能直接对比公开测试集榜单。", "",
        "公开 SignNet 是固定 GraphGPS 源码的独立移植，尚非历史 GraphGym 运行时复现。"
        "上次 LapPE 只有 seed 42，未混入本轮三种子均值。", "",
        "seed 42 的一阶对照/Kern/Full 复用已冻结记录；本轮补齐其余九次训练。"
        "源代码、协议和数据指纹匹配后才允许合并。seed 42 已参与前期工程开发；"
        "43/44 按本轮冻结计划产生，所有种子共用同一个验证划分。", "",
        f'完整协议和原始数值见 [{args.output.name}.json]({args.output.name}.json)。',
    ]
    if "baseline_audit" in report:
        checks = report["baseline_audit"]["checks"]
        max_error = max(value for check in checks for key, value in check.items()
                        if key.endswith("_max_abs_error"))
        buffers_match = all(check["batchnorm_buffers_match"] for check in checks)
        lines += ["", "## 公开编码器数值对照", "",
                  "另以相同权重比较固定上游版本的 MLP/GIN/GINDeepSigns 类与移植编码器。"
                  "本次 CPU float64、训练/验证模式下，前向输出、输入梯度和参数梯度最大误差"
                  f'为 {max_error:.3g}，BatchNorm 状态一致：{buffers_match}。'
                  "该检查不等于完整 GraphGym 或 GPU 数值复现。", "",
                  f'详见 [{args.baseline_audit.name}]({args.baseline_audit.name})。']
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"methods": report["methods"], "contrasts": report["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
