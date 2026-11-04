"""Audit the saved grid without API calls; reconstruct full-precision outcomes.

Run: .venv/bin/python tool/analyze_temperature_top_p_grid.py
Only analysis files under docs/ are written. Original logs remain unchanged.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))
from audit_proxy_experiments import parse_search
from llm_enas_pkb.search_space import arch_to_str, str_to_arch


def stats(values):
    return {"mean": st.mean(values), "sd": st.stdev(values) if len(values) > 1 else 0,
            "min": min(values), "max": max(values), "median": st.median(values)}


def main():
    cache_path = ROOT / "data/NAS-Bench-201-v1_1-096897.acc_cache.pkl"
    cache = pickle.loads(cache_path.read_bytes())  # Trusted repository benchmark cache.
    idx = cache["datasets"].index("cifar10")
    valid = {str_to_arch(a): v[idx] * 100 for a, v in cache["valid"].items()}
    test = {str_to_arch(a): v[idx] * 100 for a, v in cache["test"].items()}
    vmax, tmax = max(valid.values()), max(test.values())
    paths = sorted((ROOT / "logs/temperature_top_p_grid").glob("search_*.log"))
    runs, initial_sets, configs, keys = [], set(), [], set()
    for path in paths:
        parsed = parse_search(path)
        cfg = parsed["config"]
        t, p = cfg["temperature"], cfg["top_p"]
        rep = int(re.search(r"_rep(\d+)_", path.name)[1])
        key = (t, p, rep)
        assert key not in keys, (path, "duplicate cell/rep")
        keys.add(key)
        assert cfg["dataset"] == "cifar10" and cfg["seed"] == 42 and not cfg["offline"]
        config = {k: v for k, v in cfg.items() if k not in
                  {"temperature", "top_p", "search_log_path"}}
        configs.append(config)
        assert len(parsed["generations"]) == cfg["max_iterations"] == 30
        initial_sets.add(tuple(r["arch"] for r in parsed["initial"]))
        assert len(parsed["initial"]) == cfg["n_init"] == 20
        archive, records, curve, offspring, errors, ages = {}, [], [], [], [], []
        generation_curve = []
        first_offspring, first_preds = None, None
        for generation, rows in [(0, parsed["initial"])] + [
                (g["generation"], g["queries"]) for g in parsed["generations"]]:
            if generation:
                g = parsed["generations"][generation - 1]
                assert g["generation"] == generation
                for section in ["population", "pkb", "offspring", "elites", "next"]:
                    for row in g[section]:
                        if row["tag"] == "GT":
                            assert abs(valid[row["arch"]] - row["value"] * 100) <= .005001
                for row in g["offspring"]:
                    a = row["arch"]
                    offspring.append({"arch": a, "novel": a not in archive,
                                      "error_pp": row["value"] * 100 - valid[a]})
                if generation == 1:
                    first_offspring = [r["arch"] for r in g["offspring"]]
                    first_preds = [r["value"] for r in g["offspring"]]
            for row in rows:
                a = row["arch"]
                assert a not in archive, (path, "duplicate query", a)
                assert abs(valid[a] - row["value"] * 100) <= .005001
                archive[a] = valid[a]
                record = {"query": len(archive), "generation": generation, "arch": a,
                          "valid_pct": valid[a]}
                if generation:
                    match = re.search(r"reason=(\w+) prediction=([\d.]+)\s+prediction_gen=(\d+)", row["line"])
                    assert match, row["line"]
                    error = float(match[2]) * 100 - valid[a]
                    age = generation - int(match[3])
                    record.update(reason=match[1], error_pp=error, prediction_age=age)
                    errors.append(error)
                    ages.append(age)
                records.append(record)
                curve.append(max(archive.values()))
            if generation:
                assert len(archive) == g["count"]
                assert abs(max(archive.values()) - g["best"] * 100) <= .005001
                assert len(g["next"]) == 20 and len(g["elites"]) == 5
                generation_curve.append(max(archive.values()))
        assert len(archive) == cfg["query_budget"] == 100
        best = max(archive, key=archive.get)  # Stable, earliest-query tie handling.
        first = next(r for r in records if r["arch"] == best)
        text = path.read_text()
        start = datetime.fromisoformat(re.search(r"^started_at=(.*)$", text, re.M)[1])
        end = datetime.fromisoformat(re.findall(r"\[([^\]]+)\] Generation", text)[-1])
        novel = [r for r in offspring if r["novel"]]
        hits = [r for r in records if abs(r["valid_pct"] - vmax) < 1e-9]
        runs.append({"temperature": t, "top_p": p, "rep": rep,
                     "log": str(path.relative_to(ROOT)), "sha256": parsed["hash"],
                     "valid_pct": valid[best], "test_pct": test[best], "best_arch": best,
                     "valid_hit": abs(valid[best] - vmax) < 1e-9,
                     "test_hit": abs(test[best] - tmax) < 1e-9,
                     "first_best_query": first["query"], "first_best_generation": first["generation"],
                     "first_optimum_query": hits[0]["query"] if hits else None,
                     "duration_seconds": (end-start).total_seconds(),
                     "query_mae_pp": st.mean(map(abs, errors)), "query_bias_pp": st.mean(errors),
                     "query_prediction_age": st.mean(ages),
                     "novel_offspring_mae_pp": st.mean(abs(r["error_pp"]) for r in novel),
                     "novel_offspring_bias_pp": st.mean(r["error_pp"] for r in novel),
                     "offspring_unique": len({r["arch"] for r in offspring}),
                     "offspring_count": len(offspring),
                     "mean_regret_queries20_100_pp": st.mean(vmax-v for v in curve[19:]),
                     "best_at_query": {str(q): curve[q-1] for q in [20, 35, 50, 65, 80, 100]},
                     "generation_curve_pct": generation_curve, "query_curve_pct": curve,
                     "first_offspring": first_offspring, "first_predictions": first_preds,
                     "queries": records})
    assert len(initial_sets) == 1
    assert all(c == configs[0] for c in configs)
    temperatures = sorted({r["temperature"] for r in runs})
    top_ps = sorted({r["top_p"] for r in runs})
    assert keys == {(t, p, rep) for t in temperatures for p in top_ps for rep in range(1, 6)}

    def aggregate(rs):
        metrics = ["valid_pct", "test_pct", "duration_seconds", "query_mae_pp", "query_bias_pp",
                   "novel_offspring_mae_pp", "novel_offspring_bias_pp", "offspring_unique",
                   "mean_regret_queries20_100_pp", "query_prediction_age"]
        return {"n": len(rs), **{k: stats([r[k] for r in rs]) for k in metrics},
                "valid_hits": sum(r["valid_hit"] for r in rs),
                "test_hits": sum(r["test_hit"] for r in rs),
                "distinct_best_archs": len({tuple(r["best_arch"]) for r in rs}),
                "first_optimum_query_successes": stats([r["first_optimum_query"] for r in rs
                    if r["valid_hit"]]) if any(r["valid_hit"] for r in rs) else None,
                "best_at_query": {str(q): stats([r["best_at_query"][str(q)] for r in rs])
                                  for q in [20, 35, 50, 65, 80, 100]}}

    cells = [{"temperature": t, "top_p": p, **aggregate([r for r in runs
              if r["temperature"] == t and r["top_p"] == p])}
             for t in temperatures for p in top_ps]
    payload = {"source": "logs/temperature_top_p_grid", "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
               "validation_optimum_pct": vmax, "test_optimum_pct": tmax,
               "common_config": configs[0], "audit": {"complete_runs": len(runs),
               "generations": len(runs)*30, "unique_queries_per_run": 100,
               "identical_initial_population": True, "common_config_except_grid_and_log": True,
               "all_logged_gt_and_generation_best_match_cache": True,
               "identical_first_generation_offspring": len({str(r["first_offspring"]) for r in runs}) == 1},
               "overall": aggregate(runs), "cells": cells,
               "by_temperature": [{"temperature": t, **aggregate([r for r in runs if r["temperature"] == t])} for t in temperatures],
               "by_top_p": [{"top_p": p, **aggregate([r for r in runs if r["top_p"] == p])} for p in top_ps],
               "best_arch_frequency": [{"arch": a, "n": n, "valid_pct": valid[a], "test_pct": test[a]}
                   for a, n in Counter(tuple(r["best_arch"]) for r in runs).most_common()],
               "runs": runs}
    out = ROOT / "docs/temperature_top_p_grid_analysis_20261103.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_report(payload, out.with_suffix(".md"))
    compact = {k: v for k, v in payload.items() if k not in {"runs", "common_config", "cells"}}
    compact["cells"] = [{k: c[k] for k in ["temperature", "top_p", "valid_pct", "test_pct",
             "valid_hits", "test_hits", "distinct_best_archs", "mean_regret_queries20_100_pp",
             "query_mae_pp", "novel_offspring_mae_pp", "duration_seconds"]} for c in cells]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def write_report(x, out):
    lines = [
        "# Temperature × top_p 网格搜索分析（2026-11-03）", "",
        "## 结论", "",
        "若主要目标是在 100 次真实查询内找到最佳架构，当前首选候选是 **temperature=0.1、top_p=0.5**；"
        "**0.8、0.1** 在最终精度上并列。两者均 5/5 命中验证全局最优，验证精度为 91.606667%，对应测试精度为 94.373333%。"
        "这是本次固定 seed 下的经验结果，不构成唯一最优或跨种子优越性的证明。", "",
        "如果优先考虑搜索过程中的查询效率，**0.1、0.9** 值得保留：其平均累计最优差距最小，"
        "虽然最终命中率为 4/5。跨温度看，top_p=0.7–0.9 的结果下界明显更稳。", "",
        "## 范围、口径与核验", "",
        "- 实际网格为 temperature={0.1, 0.3, 0.6, 0.8} × top_p={0.1, 0.3, 0.5, 0.7, 0.9}，"
        "共 20 组 × 5 次 = 100 次。没有 temperature=1.0；启动脚本的帮助文字仍列出 1.0，但实际默认列表只有四档。",
        "- CIFAR-10，DeepSeek-V4-Pro-0813，非 thinking；初始种群/每代候选数配置为 20，30 代，"
        "100 次真实查询（含初始 20 次），architecture_features=both，pkb_error_feedback=false。",
        "- 全部固定 seed=42；100 次初始架构及顺序完全相同，第一代候选也完全相同。除 temperature、top_p 和日志路径外，记录配置一致。",
        "- 核验通过：100 次均有完整的 30 代及最终种群记录；每次恰好查询 100 个不同架构；"
        "逐代累计数、日志 GT 精度、逐代 best-valid 与本地 NAS-Bench-201 缓存一致（容许四位小数舍入）。",
        "- 按初始种群和每代新查询重建有序 archive，以缓存完整精度选验证最佳架构；验证并列时取最早查询者。"
        "测试精度只读取这个验证最佳架构，不在测试集上挑选架构。",
        "- 均值 ± 样本标准差（ddof=1）；准确率单位为 %，标准差、误差及差距单位为百分点。统计单位为一次完整搜索。",
        "- 本分析仅读取日志和本地缓存，没有重新调用 LLM、训练网络或消耗搜索查询预算。", "",
        "## 全部 20 组结果", "",
        "热图见同目录 `temperature_top_p_grid_20261103.png`（另有 PDF、SVG）。颜色仅表示均值，"
        "不表示显著性；完整 SD 及范围如下。", "",
        "命中指验证全局最优命中次数；本次对应测试全局最优命中次数恰好相同。", "",
        "| Temperature | top_p | 验证精度均值 ± SD (%) | 对应测试精度均值 ± SD (%) | 验证范围 (%) | 命中 | 最佳架构种类 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in x["cells"]:
        v, t = c["valid_pct"], c["test_pct"]
        lines.append(f"| {c['temperature']:.1f} | {c['top_p']:.1f} | {v['mean']:.6f} ± {v['sd']:.6f} | "
                     f"{t['mean']:.6f} ± {t['sd']:.6f} | {v['min']:.4f}–{v['max']:.4f} | {c['valid_hits']}/5 | {c['distinct_best_archs']} |")
    lines += ["", "## 参数趋势", "",
              "以下为平衡网格内的边际汇总：每个 temperature 包含五档 top_p，每个 top_p 包含四档 temperature。"
              "SD 同时包含不同参数格之间的差异及格内重复波动，不能解释为单一配置的 SD。", "",
              "| 参数 | 取值 | n | 验证均值 ± SD (%) | 验证最优命中 | 对应测试均值 (%) |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, label in [("by_temperature", "temperature"), ("by_top_p", "top_p")]:
        for c in x[key]:
            v = c["valid_pct"]
            lines.append(f"| {label} | {c[label]:.1f} | {c['n']} | {v['mean']:.6f} ± {v['sd']:.6f} | "
                         f"{c['valid_hits']}/{c['n']} | {c['test_pct']['mean']:.6f} |")
    lines += ["",
        "1. **温度没有单调关系。** 0.6 在本次边际均值上最低，0.8 又恢复；不能概括为温度越低越好，"
        "也不能据此认定某个中间温度有普遍劣势。temperature 与 top_p 的组合需要一起看。",
        "2. **较高 top_p 的主要优势是较好的结果下界。** top_p=0.7 和 0.9 各 20 次，均为 15 次验证最优、5 次次优，"
        "最差验证精度均为 91.553333%。top_p≤0.5 的 60 次中有 12 次低于 91.55%。这是描述性观察，未作显著性宣称。",
        "3. **极低温度和极低 top_p 的组合并不稳。** (0.1, 0.1) 仅 3/5 命中，验证 SD=0.139757 个百分点，"
        "最差为 91.284000%；(0.6, 0.1) 的 SD 更大，为 0.160579。",
        "4. **验证排名接近，不保证测试结果接近。** (0.3, 0.5) 验证均值第三，但其中一次选择了"
        "架构 [4,3,0,1,2,3]，验证 91.568000%、测试 93.810000%，拉低测试均值至 94.260667%。"
        "这是验证与测试排序不一致的实例，不能用测试结果反向挑参数后仍把测试成绩当作独立评估。", "",
        "## 查询效率与收敛", "",
        "效率指标为查询编号 q=20…100 上的平均差距：mean(91.606667 − best_valid_at_q)，"
        "越小表示在同样真实查询预算下越早接近最优。包含全部成功和未命中重复，按日志中的批内查询顺序计算。"
        "它比仅对命中重复计算首次命中查询数更适合比较，后者存在成功者筛选偏差。", "",
        "| Temperature | top_p | 平均差距（百分点） | 最优命中 | 五次首次最优查询编号 |",
        "|---:|---:|---:|---:|---|",
    ]
    for c in sorted(x["cells"], key=lambda c: c["mean_regret_queries20_100_pp"]["mean"]):
        rs = [r for r in x["runs"] if (r["temperature"], r["top_p"]) == (c["temperature"], c["top_p"])]
        q = ", ".join(str(r["first_optimum_query"]) if r["valid_hit"] else "未命中" for r in rs)
        lines.append(f"| {c['temperature']} | {c['top_p']} | {c['mean_regret_queries20_100_pp']['mean']:.6f} | {c['valid_hits']}/5 | {q} |")
    lines += ["",
        "- (0.1, 0.9) 的平均差距为 0.193475，比 (0.1, 0.5) 的 0.258426 小约 25.1%，但最终命中少一次。"
        "因此“最终命中率最好”与“过程查询效率最好”并非同一配置。",
        "- top_p=0.9 的边际平均差距为 0.210855，较 top_p=0.7 的 0.253080 小约 16.7%；两者最终成绩分布相同。",
        "- 初始最优均为 90.814667%。100 次搜索整体在查询 50、80、100 时的平均最佳验证精度分别为 "
        "91.322020%、91.546967%、91.578300%；大部分提升出现在中前段，但部分重复直到第 100 次查询才找到最优。", "",
        "## 预测误差、路径差异与耗时", "",
        "- 每次后续 80 个查询的 MAE 平均为 **1.138575 个百分点**，有符号偏差平均为 **+0.836118 个百分点**，"
        "表现为总体高估。exploit 共 5400 次，MAE=0.951414；explore 共 2600 次，MAE=1.527295。"
        "二者候选分布不同，不能把误差差异当作策略因果效果。",
        "- 用缓存事后核对尚未入 archive 的 offspring：先按运行求均值，再跨运行汇总，MAE=1.631107、"
        "偏差=+0.683403 个百分点。该指标包含同一架构在不同代重复出现；不是独立测试样本的泛化误差。"
        "预测值来自日志四位小数，误差统计也有对应舍入精度限制。",
        "- 查询 MAE 与最终验证精度的运行间 Pearson 描述性相关约为 −0.050。"
        "例如 (0.1, 0.1) 查询 MAE 较小，但最终结果明显更不稳定；不能只按回归 MAE 选择搜索参数。"
        "适应性选择的候选分布也会影响这个相关性，本分析没有据此作因果结论。",
        "- 第一代候选完全一致，但 100 组预测向量中已有 92 种不同结果；(0.1, 0.5) 五次第一代预测也全部不同。"
        "它最终 SD=0 只说明找到了同一最优架构，不说明 LLM 输出、演化过程或查询顺序一致。",
        "- 每次生成的不同 offspring 架构数平均 461.26，范围 420–495；实际真实查询严格固定为 100。",
        "- 以实验开始至第 30 代记录时间计算，运行总计约 5.10 小时；单次中位数 134.915 秒，"
        "最短 110.376 秒、最长 1342.366 秒。(0.8, 0.1) 平均 760.009 秒，而 (0.1, 0.5) 平均 132.782 秒。"
        "异常从 (0.6, 0.9) 第五次延续到后续部分运行，可能涉及服务负载、网络或重试；当前日志无法区分。"
        "参数按顺序执行，时间段与参数混杂，不能认定高温导致变慢，也不能用本次墙钟时间作为稳定的参数排名。", "",
        "## 最终架构分布", "",
        "| 架构编码 | 次数 | 验证精度 (%) | 对应测试精度 (%) |",
        "|---|---:|---:|---:|",
    ]
    for a in x["best_arch_frequency"]:
        lines.append(f"| `{a['arch']}` | {a['n']} | {a['valid_pct']:.6f} | {a['test_pct']:.6f} |")
    lines += ["", "## 解释边界与下一轮实验", "",
        "1. **五次相同 seed 重复，不是五个独立演化种子。** 本次能衡量固定初始条件下线上响应及后续搜索的波动；"
        "不能证明对不同初始种群、不同数据集或不同模型同样有效。",
        "2. **同 seed 不等于全程相同 RNG 轨迹。** 响应改变排序与入选种群，去重和锦标赛的循环次数还可能改变随机数消耗。"
        "启动脚本关于全程 RNG 字节一致、唯一变化来源就是采样的注释过强。客户端未传 API seed，"
        "当前也没有 prompt/response、重试或服务后端标识日志，因此无法将所有变异归为 temperature/top_p 采样本身。",
        "3. **20 组中事后挑冠军存在选择偏差。** 5/5 命中不是稳定 100% 成功的证明；若假定独立同分布伯努利重复，"
        "其双侧 95% 精确二项区间下界也只有约 47.8%。这里仅用于说明样本量限制，实际服务时间相关性可能进一步影响推断。"
        "本报告没有进行 20 组两两显著性检验，也没有把 3000 代或大量架构当作独立重复扩充样本量。",
        "4. **建议下一轮保留三组候选**：(0.1,0.5) 作为最终命中候选，(0.8,0.1) 作为并列候选，"
        "(0.1,0.9) 作为查询效率候选。使用同一组至少 20 个新 seed 对三组配对比较，条件允许时每个 seed 再做 2–3 次线上重复。"
        "随机化或交错运行顺序；记录请求响应、重试/回退、耗时和模型版本信息。",
        "5. 预先指定主指标为预算 100 下的最佳验证精度或全局最优命中率，平均查询差距为次指标。"
        "测试精度只在按验证标准确定配置后报告；本次测试结果应标记为探索性。", "",
        "## 复现与文件", "",
        "在仓库根目录运行 `.venv/bin/python tool/analyze_temperature_top_p_grid.py`。"
        "同名 JSON 保存全部 100 次的原始日志路径及 SHA-256、缓存 SHA-256、查询记录、完整精度和逐查询收敛曲线。"
        "本报告不替代原始实验的完整响应审计；当前目录只有搜索状态日志。", "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
