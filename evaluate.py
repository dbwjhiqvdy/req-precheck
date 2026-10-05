# -*- coding: utf-8 -*-
"""评测：把预审结果与人工标注对比，算出准确率、漏判率、误拦率与分项一致率。

为什么这个脚本比 precheck.py 更重要
----------------------------------
"我做了个 AI 工具"没有说服力，"我知道它错在哪、错多少、为什么错"才有。
本脚本产出的就是后者：不是感觉，是数字。

用法
----
    # 单个版本
    python evaluate.py --results results/precheck_v2.jsonl --labels data/labels.jsonl

    # 两个版本对比（消融实验：只改 prompt，其他都不变）
    python evaluate.py --results results/precheck_v1.jsonl results/precheck_v2.jsonl --labels data/labels.jsonl

指标定义（命名刻意用大白话，避免"精确率/召回率"造成歧义）
--------------------------------------------------------
- **漏判**：人工判「不就绪」，工具判「就绪」→ 放行了坏需求，**最危险的错误**
- **误拦**：人工判「就绪」，工具判「不就绪」→ 拦下了好需求，代价只是多补一次说明
- **准确率**：整体判定与人工一致的比例
- **分项一致率**：8 项规则里，每一项的判定与人工标注的一致比例
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from rubric import RUBRIC

MOCK_MARK = "MOCK"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path} 第 {line_number} 行不是合法 JSON：{error}") from error
    return records


def evaluate_one(results_path: str, labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """对单个结果文件做评测。"""
    results = load_jsonl(results_path)
    if not results:
        raise SystemExit(f"结果文件为空：{results_path}")

    is_mock = any(record.get("mode") == MOCK_MARK for record in results)

    # 输出不可解析的记录数：这类结果的成绩"看起来有数字"，实际不可解读。
    # 必检——否则读者会把"全部强制判失败"当成模型判定质量差。
    def _looks_unparsed(record: dict[str, Any]) -> bool:
        problems = [str(problem) for problem in (record.get("problems") or [])]
        if any(problem.startswith("SCHEMA") or problem.startswith("API_ERROR") for problem in problems):
            return True
        # 兼容早期版本写出的结果文件：解析失败时 8 项理由都会被写成同一句占位文案
        items = record.get("items") or []
        return bool(items) and all("模型未返回该项" in str(item.get("reason", "")) for item in items)

    schema_issue_records = sum(1 for record in results if _looks_unparsed(record))

    paired: list[dict[str, Any]] = []
    unlabeled = 0
    for record in results:
        label = labels.get(record.get("id"))
        if not label or label.get("human_missing") is None:
            unlabeled += 1
            continue
        human_missing = set(label.get("human_missing") or [])
        human_ready = len(human_missing) == 0
        llm_ready = bool(record.get("ready"))
        llm_missing = set(record.get("missing") or [])
        paired.append(
            {
                "id": record.get("id"),
                "prompt_version": record.get("prompt_version"),
                "human_ready": human_ready,
                "llm_ready": llm_ready,
                "human_missing": sorted(human_missing),
                "llm_missing": sorted(llm_missing),
                "both_ready": human_ready and llm_ready,
                "both_not_ready": (not human_ready) and (not llm_ready),
                "missed": (not human_ready) and llm_ready,      # 漏判：放行坏需求
                "false_alarm": human_ready and (not llm_ready),  # 误拦：拦下好需求
                "items": record.get("items") or [],
            }
        )

    if not paired:
        raise SystemExit(
            f"{results_path}：没有任何已标注的数据。\n"
            "请先在 data/labels.jsonl 里填写 human_missing（把该需求缺失的规则 id 列进去，全部满足则填 []）。"
        )

    total = len(paired)
    human_not_ready = [row for row in paired if not row["human_ready"]]
    human_ready_rows = [row for row in paired if row["human_ready"]]

    missed = [row for row in paired if row["missed"]]
    false_alarm = [row for row in paired if row["false_alarm"]]
    correct = [row for row in paired if row["both_ready"] or row["both_not_ready"]]

    # 分项一致率
    item_stats: list[dict[str, Any]] = []
    for rule in RUBRIC:
        agree = 0
        counted = 0
        for row in paired:
            llm_item = next((item for item in row["items"] if item.get("id") == rule["id"]), None)
            if llm_item is None:
                continue
            counted += 1
            human_passed = rule["id"] not in row["human_missing"]
            if bool(llm_item.get("passed")) == human_passed:
                agree += 1
        item_stats.append(
            {
                "id": rule["id"],
                "name": rule["name"],
                "agreement": (agree / counted) if counted else 0.0,
                "counted": counted,
                "agreed": agree,
            }
        )
    item_stats.sort(key=lambda entry: entry["agreement"])

    # 懒惰基线：用来证明工具确实"做了事"，而不是靠数据分布取胜。
    # 就绪率越低，"永远打回"这个什么都不做的策略漏判率就越低（甚至为 0）——
    # 报告里不报基线，低漏判率就是误导。
    baseline_reject = {
        "name": "永远判「不就绪」（什么都不做）",
        "accuracy": len(human_not_ready) / total,
        "missed_rate": 0.0,
        "false_alarm_rate": (len(human_ready_rows) / len(human_ready_rows)) if human_ready_rows else 0.0,
    }
    baseline_pass = {
        "name": "永远判「就绪」（全部放行）",
        "accuracy": len(human_ready_rows) / total,
        "missed_rate": 1.0 if human_not_ready else 0.0,
        "false_alarm_rate": 0.0,
    }

    return {
        "results_path": results_path,
        "prompt_version": paired[0].get("prompt_version"),
        "baselines": [baseline_reject, baseline_pass],
        "is_mock": is_mock,
        "schema_issue_records": schema_issue_records,
        "total": total,
        "unlabeled": unlabeled,
        "accuracy": len(correct) / total,
        "missed_count": len(missed),
        "missed_rate": (len(missed) / len(human_not_ready)) if human_not_ready else 0.0,
        "false_alarm_count": len(false_alarm),
        "false_alarm_rate": (len(false_alarm) / len(human_ready_rows)) if human_ready_rows else 0.0,
        "correct_block": len([row for row in paired if row["both_not_ready"]]),
        "correct_pass": len([row for row in paired if row["both_ready"]]),
        "human_not_ready": len(human_not_ready),
        "human_ready": len(human_ready_rows),
        "missed_rows": missed,
        "false_alarm_rows": false_alarm,
        "item_stats": item_stats,
        "paired": paired,
    }


def render_report(reports: list[dict[str, Any]], labels_path: str) -> str:
    lines: list[str] = []
    any_mock = any(report["is_mock"] for report in reports)

    lines.append("# 需求预审工具 · 评测报告")
    lines.append("")
    lines.append(f"- 标注文件：`{labels_path}`")
    lines.append(f"- 结果文件：{', '.join('`' + report['results_path'] + '`' for report in reports)}")
    if any_mock:
        lines.append("")
        lines.append("> ⚠️ **本报告包含 MOCK 模式的数据。MOCK 由关键词启发式伪造，不代表模型能力，"
                     "不得作为工具效果的证据使用。**")
    lines.append("")

    # 概要对比
    lines.append("## 一、总体指标")
    lines.append("")
    lines.append("| prompt 版本 | 已标注条数 | 准确率 | 漏判率（放行坏需求） | 误拦率（拦下好需求） |")
    lines.append("|---|---|---|---|---|")
    for report in reports:
        version = report["prompt_version"] or os.path.basename(report["results_path"])
        lines.append(
            f"| {version} | {report['total']} | {report['accuracy']:.1%} | "
            f"{report['missed_rate']:.1%}（{report['missed_count']}/{report['human_not_ready']}） | "
            f"{report['false_alarm_rate']:.1%}（{report['false_alarm_count']}/{report['human_ready']}） |"
        )
    lines.append("")

    # 懒惰基线（批判性看数字的前提：先看这张表，再看模型成绩）
    lines.append("### 懒惰基线（必须先看这张表，再看上面）")
    lines.append("")
    lines.append("| 策略 | 准确率 | 漏判率 | 误拦率 |")
    lines.append("|---|---|---|---|")
    for baseline in reports[0]["baselines"]:
        lines.append(
            f"| {baseline['name']} | {baseline['accuracy']:.1%} | "
            f"{baseline['missed_rate']:.1%} | {baseline['false_alarm_rate']:.1%} |"
        )
    lines.append("")
    best_accuracy = max(report["accuracy"] for report in reports)
    best_baseline = max(baseline["accuracy"] for baseline in reports[0]["baselines"])
    if best_accuracy > best_baseline:
        lines.append(
            f"✅ 最好的 prompt 版本准确率 {best_accuracy:.1%}，**高于**最强的懒惰基线 {best_baseline:.1%}"
            f"（高出 {(best_accuracy - best_baseline) * 100:.1f} 个百分点）——"
            "说明工具确实在做判定，而不是靠数据分布取胜。"
        )
    else:
        lines.append(
            f"❌ 最好的 prompt 版本准确率 {best_accuracy:.1%}，**没有超过**懒惰基线 {best_baseline:.1%}——"
            "此时工具的判定价值不成立，结论必须如实写成「未证明有效」，不得包装。"
        )
    lines.append("")

    if len(reports) > 1:
        first, last = reports[0], reports[-1]
        lines.append("### 消融对比（同数据、同模型，只改 prompt）")
        lines.append("")
        lines.append(f"- 准确率：{first['accuracy']:.1%} → {last['accuracy']:.1%}"
                     f"（{(last['accuracy'] - first['accuracy']) * 100:+.1f} 个百分点）")
        lines.append(f"- 漏判率：{first['missed_rate']:.1%} → {last['missed_rate']:.1%}"
                     f"（{(last['missed_rate'] - first['missed_rate']) * 100:+.1f} 个百分点）")
        lines.append(f"- 误拦率：{first['false_alarm_rate']:.1%} → {last['false_alarm_rate']:.1%}"
                     f"（{(last['false_alarm_rate'] - first['false_alarm_rate']) * 100:+.1f} 个百分点）")
        lines.append("")

    # 每个版本的明细（作为二级标题下的子节，避免多版本时出现重复的二级标题）
    lines.append("## 二、各版本明细")
    lines.append("")
    for report in reports:
        version = report["prompt_version"] or os.path.basename(report["results_path"])
        lines.append(f"### {version}")
        lines.append("")
        if report["schema_issue_records"]:
            lines.append(
                f"> ⚠️ **本版本有 {report['schema_issue_records']}/{report['total']} 条输出无法按规则 id 解析**"
                "（SCHEMA 类问题）。这意味着："
            )
            lines.append("> - 这些记录的判定被**强制记为「全部缺失」**，等价于「一律打回」；")
            lines.append("> - 因此本节的**分项一致率与混淆矩阵不可解读**——它们反映的是解析失败，而不是模型的判定质量；")
            lines.append("> - 正确读法是：**该版本没有产出可用信息**，其准确率应当与「永远打回」基线比较，而不是与其他 prompt 版本比较。")
            lines.append("")
        lines.append("**混淆矩阵**（行=人工标注，列=工具判定）")
        lines.append("")
        lines.append("| | 工具判就绪 | 工具判不就绪 |")
        lines.append("|---|---|---|")
        lines.append(f"| 人工判就绪 | {report['correct_pass']}（正确放行） | {report['false_alarm_count']}（误拦） |")
        lines.append(f"| 人工判不就绪 | {report['missed_count']}（**漏判**） | {report['correct_block']}（正确拦截） |")
        lines.append("")

        lines.append("**分项一致率**（从最差到最好——最差的那几项就是下一步要改的地方）")
        lines.append("")
        lines.append("| 规则项 | 一致率 | 一致/总数 |")
        lines.append("|---|---|---|")
        for item in report["item_stats"]:
            lines.append(f"| {item['name']}（`{item['id']}`） | {item['agreement']:.1%} | {item['agreed']}/{item['counted']} |")
        lines.append("")

        if report["missed_rows"]:
            lines.append("**漏判明细（最需要分析的部分）**")
            lines.append("")
            lines.append("| 需求 | 人工认为缺 | 工具认为缺 | 差额（工具漏掉的项） |")
            lines.append("|---|---|---|---|")
            for row in report["missed_rows"]:
                missed_items = sorted(set(row["human_missing"]) - set(row["llm_missing"]))
                lines.append(
                    f"| {row['id']} | {', '.join(row['human_missing']) or '—'} | "
                    f"{', '.join(row['llm_missing']) or '（无）'} | **{', '.join(missed_items)}** |"
                )
            lines.append("")

        if report["false_alarm_rows"]:
            lines.append("**误拦明细**")
            lines.append("")
            lines.append("| 需求 | 工具认为缺 | 人工认为 |")
            lines.append("|---|---|---|")
            for row in report["false_alarm_rows"]:
                over = sorted(set(row["llm_missing"]) - set(row["human_missing"]))
                lines.append(f"| {row['id']} | {', '.join(over)} | 全部满足 |")
            lines.append("")

    lines.append("## 三、局限（必须如实写进结论）")
    lines.append("")
    lines.append(f"- 样本量仅 {reports[0]['total']} 条，**且需求文本全部由助手起草**（可能比真实需求更贴合规则结构，缺少真实需求的混乱度）→ **结果偏乐观，不能推断到真实需求分布**")
    lines.append("- 人工标注**不是独立标注**：判定由作者主导，其中一部分是在助手的冲突提示下修订的；新增样本为作者盲判，但正文由助手起草。**没有第二人独立复核**，主观性未被消除")
    lines.append("- 判定标准里有些项（如「依赖方已确认」）在文本层面难以完全判定，天然存在上限")
    lines.append("- 就绪率不高（就绪样本数见上），误拦率的分母偏小，个别样本的翻转就会显著改变百分比")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测需求预审工具的效果")
    parser.add_argument("--results", nargs="+", required=True, help="一个或多个结果文件（按对比顺序传入）")
    parser.add_argument("--labels", default="data/labels.jsonl", help="人工标注文件")
    parser.add_argument("--out", default=None, help="报告输出路径")
    args = parser.parse_args()

    labels = {record.get("id"): record for record in load_jsonl(args.labels)}
    reports = [evaluate_one(path, labels) for path in args.results]

    any_mock = any(report["is_mock"] for report in reports)
    default_name = "eval_comparison.md" if len(reports) > 1 else f"eval_{reports[0]['prompt_version'] or 'result'}.md"
    if any_mock:
        default_name = f"_MOCK_{default_name}"
    out_path = args.out or os.path.join("reports", default_name)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(reports, args.labels))

    for report in reports:
        version = report["prompt_version"] or os.path.basename(report["results_path"])
        print(
            f"[{version}] 已标注 {report['total']} 条（未标注 {report['unlabeled']} 条）"
            f"｜准确率 {report['accuracy']:.1%}"
            f"｜漏判率 {report['missed_rate']:.1%}"
            f"｜误拦率 {report['false_alarm_rate']:.1%}"
        )
        worst = report["item_stats"][:3]
        if worst:
            print("   最差的 3 项：" + "、".join(f"{item['name']}（{item['agreement']:.0%}）" for item in worst))
    if any_mock:
        print("\n⚠️ 检测到 MOCK 数据：报告已加 _MOCK_ 前缀，**不可作为工具效果的证据**")
    print(f"\n报告已写入：{out_path}")


if __name__ == "__main__":
    main()
