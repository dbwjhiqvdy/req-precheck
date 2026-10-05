# -*- coding: utf-8 -*-
"""结果查看器：把 results/*.jsonl 变成人能读的东西。

为什么需要它
------------
预审结果是 JSONL（每行一个 JSON），一行几千字符，直接用记事本或 Excel 打开几乎没法看。
更重要的问题是：**只看"就绪/打回"没有意义，要看它为什么这么判**——
而 `raw_response` 和 notes 里恰好有模型的推理过程，那才是改进 prompt 的依据。

用法
----
    python show_results.py                     # 一览表：每条需求 × 各版本 + 人工标注
    python show_results.py R01                 # 展开某条：人工 vs 各版本逐项对比 + 理由
    python show_results.py R01 -v v2           # 只看某个版本（显示完整理由）
    python show_results.py --diff v1b v2       # 列出两个版本判定不同的需求

命名说明
--------
本文件原名 `inspect.py`，结果遮蔽了标准库的 `inspect` 模块，导致同目录下**所有**脚本
在 `argparse` 初始化时报 `AttributeError: module 'inspect' has no attribute 'signature'`。
教训：**模块名绝不与标准库同名**——否则症状会出现在完全无关的第三方代码里，极难定位。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

from rubric import RUBRIC

NAME = {rule["id"]: rule["name"] for rule in RUBRIC}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            rows.append(json.loads(text))
    return rows


def discover_results() -> dict[str, str]:
    """找到所有非 MOCK 的结果文件，键为版本名。"""
    found: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join("results", "precheck_*.jsonl"))):
        base = os.path.basename(path)
        if "MOCK" in base:
            continue
        version = base.replace("precheck_", "").replace(".jsonl", "")
        found[version] = path
    return found


def load_labels() -> dict[str, list[str]]:
    path = os.path.join("data", "labels.jsonl")
    if not os.path.isfile(path):
        return {}
    return {row["id"]: (row.get("human_missing") or []) for row in load_jsonl(path)}


def print_overview(results: dict[str, dict[str, dict[str, Any]]], labels: dict[str, list[str]]) -> None:
    versions = list(results)
    header = f"{'ID':<6}{'人工':<8}" + "".join(f"{v:<10}" for v in versions) + "一致?"
    print(header)
    print("-" * len(header))
    for rid in sorted(labels or results[versions[0]]):
        human_missing = labels.get(rid)
        human = "—" if human_missing is None else ("就绪" if not human_missing else f"打回({len(human_missing)})")
        cells = ""
        agree = 0
        counted = 0
        for version in versions:
            record = results[version].get(rid)
            if record is None:
                cells += f"{'?':<10}"
                continue
            cell = "就绪" if record["ready"] else f"打回({len(record['missing'])})"
            cells += f"{cell:<10}"
            if human_missing is not None:
                counted += 1
                if bool(record["ready"]) == (len(human_missing) == 0):
                    agree += 1
        tail = f"{agree}/{counted}" if counted else "—"
        print(f"{rid:<6}{human:<8}{cells}{tail}")

    print()
    print("各版本汇总：")
    for version in versions:
        rows = list(results[version].values())
        ready = sum(1 for row in rows if row["ready"])
        schema = sum(1 for row in rows if any(str(p).startswith("SCHEMA") for p in (row.get("problems") or [])))
        note = f"｜输出不可解析 {schema}/{len(rows)}（该版本数字不可解读）" if schema else ""
        print(f"  {version:<6} 判定就绪 {ready}/{len(rows)}{note}")


def print_detail(
    rid: str,
    results: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, list[str]],
    only_version: str | None,
    full_reason: bool,
) -> None:
    versions = [only_version] if only_version else list(results)
    human_missing = labels.get(rid)
    if human_missing is not None:
        print(f"人工标注：{'就绪（8 项全通过）' if not human_missing else '打回，缺 ' + ', '.join(human_missing)}")
    else:
        print("人工标注：未标注")
    print()

    for version in versions:
        record = results[version].get(rid)
        if record is None:
            print(f"[{version}] 无该条记录")
            continue
        print(f"=== {version}（{'就绪' if record['ready'] else '打回'}，缺 {len(record['missing'])} 项）===")
        if record.get("problems"):
            for problem in record["problems"]:
                print(f"  ⚠ {problem}")
        if record.get("model_criteria"):
            print(f"  模型自用标准：{'、'.join(record['model_criteria'][:8])}")
        for item in record["items"]:
            mark = "✅" if item["passed"] else "❌"
            human_mark = ""
            if human_missing is not None:
                human_passed = item["id"] not in human_missing
                human_mark = "  ← 与人工不一致" if human_passed != item["passed"] else ""
            reason = item.get("reason", "")
            if not full_reason and len(reason) > 90:
                reason = reason[:90] + "…"
            print(f"  {mark} {NAME[item['id']]:<12}{reason}{human_mark}")
        print()


def print_diff(
    results: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, list[str]],
    version_a: str,
    version_b: str,
) -> None:
    print(f"=== {version_a} 与 {version_b} 判定不同的需求 ===")
    count = 0
    for rid in sorted(results.get(version_a, {})):
        a = results[version_a].get(rid)
        b = results[version_b].get(rid)
        if a is None or b is None or a["ready"] == b["ready"]:
            continue
        count += 1
        human_missing = labels.get(rid)
        human = "—" if human_missing is None else ("就绪" if not human_missing else "打回")
        diff_a = set(a["missing"]) - set(b["missing"])
        diff_b = set(b["missing"]) - set(a["missing"])
        print(f"{rid}  人工={human}  {version_a}={'就绪' if a['ready'] else '打回'}  {version_b}={'就绪' if b['ready'] else '打回'}")
        if diff_a:
            print(f"    仅 {version_a} 判缺失：{', '.join(sorted(diff_a))}")
        if diff_b:
            print(f"    仅 {version_b} 判缺失：{', '.join(sorted(diff_b))}")
    if not count:
        print("两个版本的整体判定完全一致。")


def main() -> None:
    parser = argparse.ArgumentParser(description="查看预审结果")
    parser.add_argument("id", nargs="?", help="需求 ID（如 R01）；省略则显示一览表")
    parser.add_argument("-v", "--version", help="只看某个 prompt 版本")
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="列出两版判定不同的需求")
    parser.add_argument("--full", action="store_true", help="显示完整理由，不截断")
    args = parser.parse_args()

    paths = discover_results()
    if not paths:
        raise SystemExit("results/ 下没有找到结果文件。先运行：python precheck.py --prompt v2")
    results = {version: {row["id"]: row for row in load_jsonl(path)} for version, path in paths.items()}
    labels = load_labels()

    if args.diff:
        print_diff(results, labels, args.diff[0], args.diff[1])
    elif args.id:
        print_detail(args.id, results, labels, args.version, args.full)
    else:
        print(f"结果文件：{', '.join(paths.values())}")
        print()
        print_overview(results, labels)


if __name__ == "__main__":
    main()
