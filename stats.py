# -*- coding: utf-8 -*-
"""标注统计：一眼看清标注进度、判定分布、各规则项被判定缺失的频次。

用法：
    python stats.py

为什么需要它
------------
标注数据集最容易出的问题是"分布失衡"——例如某个规则项从来没被判过缺失，
那评测里这一项的一致率就没有意义（分母是 0 或全通过）。
这个脚本在跑评测前先暴露这类问题。
"""

from __future__ import annotations

import json
from collections import Counter

from rubric import RUBRIC

NAME = {rule["id"]: rule["name"] for rule in RUBRIC}


def load_rows(path: str = "data/labels.jsonl") -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("//"):
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path} 第 {line_number} 行不是合法 JSON：{error}") from error
    return rows


def main() -> None:
    rows = load_rows()
    labeled = [row for row in rows if row["human_missing"] is not None]
    unlabeled = [row for row in rows if row["human_missing"] is None]
    ready = [row for row in labeled if not row["human_missing"]]

    print("=== 逐条状态 ===")
    print(f"{'ID':<5}{'状态':<8}{'缺项数':<8}缺失项")
    for row in rows:
        if row["human_missing"] is None:
            print(f"{row['id']:<5}{'未标注':<8}{'-':<8}-")
        else:
            missing = row["human_missing"]
            detail = ", ".join(missing) if missing else "（8 项全通过）"
            print(f"{row['id']:<5}{'已标注':<8}{len(missing):<8}{detail}")

    print()
    print("=== 概要 ===")
    print(f"总数 {len(rows)}｜已标注 {len(labeled)}｜未标注 {len(unlabeled)}")
    if ready:
        print(f"判定就绪 {len(ready)} 条：" + ", ".join(row["id"] for row in ready))
    print(f"判定不就绪 {len(labeled) - len(ready)} 条")

    print()
    print("=== 各规则项被判定缺失的频次（降序；频次为 0 的项要警惕） ===")
    counter: Counter[str] = Counter()
    for row in labeled:
        for item in row["human_missing"]:
            counter[item] += 1
    total = len(labeled) or 1
    for item, count in counter.most_common():
        print(f"{NAME[item]:<14}{count:>3}/{total}  {'█' * count}")
    for rule in RUBRIC:
        if rule["id"] not in counter:
            print(f"{rule['name']:<14}{0:>3}/{total}  （从未被判缺失——评测中该项无区分度）")

    print()
    notes_filled = [row for row in labeled if row["note"].strip()]
    print(f"=== note 填写情况：{len(notes_filled)}/{len(labeled)} 条有理由 ===")
    if len(notes_filled) < len(labeled):
        print("提示：没有理由的标注无法回溯，建议补齐。")


if __name__ == "__main__":
    main()
