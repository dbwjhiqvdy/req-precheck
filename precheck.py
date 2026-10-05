# -*- coding: utf-8 -*-
"""需求预审：按 DoR 8 项逐条判定一条游戏需求是否「就绪」。

用法
----
    python precheck.py --input data/requirements.jsonl --prompt v2 --mock
    python precheck.py --input data/requirements.jsonl --prompt v2          # 真实调用

产出
----
    results/precheck_<prompt>[_MOCK].jsonl   每条需求一行，含 8 项判定与理由

设计要点
--------
1. **两个 prompt 版本**（v1 朴素 / v2 工程化），用来做消融对比：
   同一份数据、同一个模型，只改 prompt —— 差异就是 prompt 工程的价值。
   这正是评测能证明的东西。
2. **判定缺项一律视为未通过**：宁可打回，不可漏放（DoR 的纪律）。
3. **每条判定必须带理由**，且理由要引用需求原文片段，便于需求方反驳。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from llm import MOCK_MARK, LLMConfigError, chat_json, load_dotenv, mock_chat_json
from rubric import RUBRIC, missing_items, overall_ready, render_rubric

SYSTEM_PROMPT = "你是一位资深游戏项目管理者的助手，负责在需求评审前做预审。你的输出必须是严格的 JSON，不能包含任何解释性文字。"

# ---------------------------------------------------------------------------
# prompt v1：朴素写法（对照组）
# ---------------------------------------------------------------------------
PROMPT_V1 = """请判断下面这条游戏需求是否满足项目管理中的需求就绪标准，逐项给出判定。

需求：
{requirement}
"""

# ---------------------------------------------------------------------------
# prompt v1b：只给输出格式的对照组（**实验设计的关键补充**）
#
# 为什么需要它：v1 的失败主要来自"输出不可解析"，而不是"判定质量差"。
# 如果只比 v1 和 v2，两个效果被混在一起：
#   效果①：说了输出格式 → 结果能被程序用（可用性）
#   效果②：给了规则细节 + 判定纪律 → 判定更准（准确性）
# v1b 只给①不给②，于是：
#   v1 → v1b 的差异 = 效果①（输出格式的价值）
#   v1b → v2 的差异 = 效果②（规则与纪律的价值）
# 这样结论才能落到具体做法上，而不是笼统的"prompt 很重要"。
# ---------------------------------------------------------------------------
PROMPT_V1B = """请对下面的游戏研发需求做逐项判定，判断它是否满足「需求就绪」标准。

## 输出格式（严格遵守，只输出 JSON）

{{
  "items": [
    {{"id": "one_liner", "passed": true, "reason": "…"}},
    {{"id": "core_loop", "passed": true, "reason": "…"}},
    {{"id": "acceptance", "passed": true, "reason": "…"}},
    {{"id": "edge_cases", "passed": true, "reason": "…"}},
    {{"id": "dependencies", "passed": true, "reason": "…"}},
    {{"id": "assets", "passed": true, "reason": "…"}},
    {{"id": "estimation", "passed": true, "reason": "…"}},
    {{"id": "risk_and_cut", "passed": true, "reason": "…"}}
  ]
}}

8 个 id 的含义：
one_liner=一句话说得清 / core_loop=核心循环明确 / acceptance=验收标准可度量 /
edge_cases=边界与异常已定义 / dependencies=依赖方已确认 / assets=素材到位或已有排期 /
estimation=工作量已估算且拆到 ≤3 天 / risk_and_cut=风险与砍法已写明

## 待预审的需求

{requirement}
"""

# ---------------------------------------------------------------------------
# prompt v2：工程化写法（实验组）
# 改动点：① 明确角色与任务边界 ② 给出完整、结构化的判定规则
#        ③ 明确输出 schema ④ 规定缺项时如何处理 ⑤ 规定不确定时的倾向
#        ⑥ 要求引用原文片段作为证据
# ---------------------------------------------------------------------------
PROMPT_V2 = """你要对一条游戏研发需求做**评审前预审**：逐项判定它是否满足「需求就绪定义（DoR）」。

## 判定规则（共 8 项，必须逐项判定）

{rubric}

## 判定纪律（必须遵守）

1. 只依据需求文本本身判定。**不要**依据"一般项目都会这么做"来推断——文本里没写，就是没写。
2. **不确定时判为未通过**。DoR 的代价是不对称的：漏放一条未就绪的需求，代价是开发中返工；误拦一条已就绪的需求，代价只是补一次说明。
3. 如果需求文本中存在与开发者约定俗成的冲突（例如验收标准写了"体验更好"这类不可度量的表述），判为未通过。
4. 每一项的 `reason` 必须**引用需求原文中的片段**（可以是"原文未提及"），不能只给结论。
5. 需求里若出现"参照竞品""后面再定""简单做一下"这类表述，确认其是否真的缺少对应信息；缺了就判未通过。

## 输出格式（严格遵守，只输出 JSON）

{{
  "items": [
    {{"id": "one_liner", "passed": true, "reason": "原文：<引用片段>；判定：…"}},
    ... 共 8 项，id 必须与上方规则一一对应，顺序一致 ...
  ]
}}

## 待预审的需求

{requirement}
"""

PROMPTS = {"v1": PROMPT_V1, "v1b": PROMPT_V1B, "v2": PROMPT_V2}


def build_prompt(version: str, requirement: dict[str, Any]) -> str:
    """按版本拼装 prompt。需求文本 = 标题 + 正文。"""
    template = PROMPTS[version]
    text = requirement.get("text", "").strip()
    title = requirement.get("title", "").strip()
    requirement_text = f"【标题】{title}\n【正文】{text}" if title else text
    if version == "v2":
        return template.format(rubric=render_rubric(), requirement=requirement_text)
    # v1 与 v1b 都不内嵌规则文本
    return template.format(requirement=requirement_text)


# 模型可能自造的字段别名。实测 v1 用的是 criterion / pass / detail，
# 因此这里的容忍不是"以防万一"，而是"已经发生过"。
_ID_KEYS = ("id", "criterion_id", "key")
_PASSED_KEYS = ("passed", "pass", "ok", "satisfied", "is_satisfied")
_REASON_KEYS = ("reason", "detail", "explanation", "comment")
_LABEL_KEYS = ("criterion", "name", "标准", "criterion_name")


def _first_present(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def normalize_items(raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """把模型输出规整成 8 项的固定结构。

    返回 (规整后的判定列表, 异常说明列表, 模型自用的标准名列表)。

    三条规则：
    1. 缺失项 → passed=False（宁可打回，不可漏放）；
    2. 字段名容忍别名（实测模型会用 pass/detail/criterion）；
    3. 问题必须分类记录——"格式异常"这种笼统描述无法定位原因。
    """
    problems: list[str] = []
    model_criteria: list[str] = []

    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        # 容忍模型把判定数组放在别的键下（实测出现过 judgments 这种拼写）
        for key, value in raw.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                raw_items = value
                problems.append(f"SCHEMA_ITEMS_KEY：无 items 字段，改用键 {key!r}")
                break
    if not isinstance(raw_items, list):
        problems.append("SCHEMA_NO_ITEMS：输出中找不到任何可用的判定数组")
        raw_items = []

    by_id: dict[str, dict[str, Any]] = {}
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        label = _first_present(entry, _LABEL_KEYS)
        if label:
            model_criteria.append(str(label).strip()[:40])
        entry_id = _first_present(entry, _ID_KEYS)
        if isinstance(entry_id, str):
            by_id[entry_id.strip()] = entry

    matched = sum(1 for rule in RUBRIC if rule["id"] in by_id)
    if raw_items and matched == 0:
        preview = "、".join(model_criteria[:4]) or "（未提供标准名）"
        problems.append(
            f"SCHEMA_ID_MISMATCH：模型返回了 {len(raw_items)} 项判定，但没有任何一项使用规则 id"
            f"（模型自用标准：{preview}）"
        )

    normalized: list[dict[str, Any]] = []
    for rule in RUBRIC:
        entry = by_id.get(rule["id"])
        if entry is None:
            problems.append(f"MISSING_ITEM：{rule['id']}")
            normalized.append(
                {"id": rule["id"], "name": rule["name"], "passed": False, "reason": "模型未返回该项，按未通过处理"}
            )
            continue
        passed = _first_present(entry, _PASSED_KEYS)
        if isinstance(passed, str):
            passed = passed.strip().lower() in ("true", "yes", "1", "是", "通过")
        if not isinstance(passed, bool):
            problems.append(f"BAD_TYPE：{rule['id']} 的判定值不是布尔：{passed!r}")
            passed = False
        reason = _first_present(entry, _REASON_KEYS)
        normalized.append(
            {
                "id": rule["id"],
                "name": rule["name"],
                "passed": passed,
                "reason": str(reason or "").strip(),
            }
        )
    return normalized, problems, model_criteria


def run(
    input_path: str,
    version: str,
    out_path: str,
    use_mock: bool,
    limit: int | None,
) -> str:
    if version not in PROMPTS:
        raise SystemExit(f"未知的 prompt 版本：{version}（可选：{', '.join(PROMPTS)}）")

    requirements = load_jsonl(input_path)
    if limit:
        requirements = requirements[:limit]
    if not requirements:
        raise SystemExit(f"输入文件没有数据：{input_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results: list[dict[str, Any]] = []

    failures = 0
    for index, requirement in enumerate(requirements, start=1):
        prompt = build_prompt(version, requirement)
        rid = requirement.get("id")
        try:
            raw = mock_chat_json(SYSTEM_PROMPT, prompt) if use_mock else chat_json(SYSTEM_PROMPT, prompt)
        except (RuntimeError, ValueError) as error:
            # 单条失败不能拖垮整轮：30 条的结果不该因为第 21 条调用失败而全部丢掉。
            # 失败按"未通过"记录（宁可打回，不可漏放），但标明它不反映模型能力。
            failures += 1
            results.append(
                {
                    "id": rid,
                    "title": requirement.get("title", ""),
                    "prompt_version": version,
                    "mode": MOCK_MARK if use_mock else "REAL",
                    "model_criteria": [],
                    "ready": False,
                    "missing": [rule["id"] for rule in RUBRIC],
                    "items": [
                        {
                            "id": rule["id"],
                            "name": rule["name"],
                            "passed": False,
                            "reason": "调用或解析失败，按未通过处理（宁可不放行）",
                        }
                        for rule in RUBRIC
                    ],
                    "problems": [f"API_ERROR：{type(error).__name__}: {error}"],
                    "raw_response": None,
                }
            )
            print(f"[{index}/{len(requirements)}] {rid} → ⚠ 调用失败，已记录并继续（{type(error).__name__}）")
            continue

        items, problems, model_criteria = normalize_items(raw)
        record = {
            "id": rid,
            "title": requirement.get("title", ""),
            "prompt_version": version,
            "mode": MOCK_MARK if use_mock else "REAL",
            "model_criteria": model_criteria,
            "ready": overall_ready(items),
            "missing": missing_items(items),
            "items": items,
            "problems": problems,
            "raw_response": raw,
        }
        results.append(record)
        flag = "就绪" if record["ready"] else f"打回（缺 {len(record['missing'])} 项）"
        print(f"[{index}/{len(requirements)}] {record['id']} → {flag}")

    with open(out_path, "w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    ready_count = sum(1 for record in results if record["ready"])
    problem_count = sum(1 for record in results if record["problems"])
    print(f"\n完成：{len(results)} 条，其中判定就绪 {ready_count} 条、打回 {len(results) - ready_count} 条")
    if problem_count:
        # 按问题类型分类统计——"格式异常"这种笼统说法无法定位原因
        kinds: dict[str, int] = {}
        for record in results:
            for problem in record["problems"]:
                kind = problem.split("：", 1)[0]
                kinds[kind] = kinds.get(kind, 0) + 1
        detail = "、".join(f"{kind}×{count}" for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]))
        print(f"⚠️ 有 {problem_count} 条存在解析问题：{detail}")
        if any(problem.startswith("SCHEMA_ID_MISMATCH") for record in results for problem in record["problems"]):
            print("   提示：SCHEMA_ID_MISMATCH 表示模型没按规则 id 输出——这是 prompt 未指定输出格式的典型后果，")
            print("        不是模型判定能力的问题。建议对照 v1b（只给格式）与 v2（给格式+规则+纪律）。")
    if failures:
        print(f"⚠️ 其中 {failures} 条调用/解析失败，已按「未通过」记录（宁可不放行）——")
        print("   这些记录**不反映模型能力**，评测报告的「不可解读」提示会覆盖它们。建议重跑该版本。")
    if use_mock:
        print(f"⚠️ 当前为 {MOCK_MARK} 模式：结果由关键词启发式伪造，**不可用于评测或写进作品集**")
    print(f"结果已写入：{out_path}")
    return out_path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="按 DoR 规则预审游戏需求")
    parser.add_argument("--input", default="data/requirements.jsonl", help="需求数据（jsonl）")
    parser.add_argument("--prompt", default="v2", choices=sorted(PROMPTS), help="prompt 版本")
    parser.add_argument("--out", default=None, help="输出文件路径")
    parser.add_argument("--mock", action="store_true", help="不联网，用伪造输出跑通流程")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（省 token）")
    args = parser.parse_args()

    load_dotenv()  # 允许通过 .env 提供 DEEPSEEK_API_KEY
    suffix = "_MOCK" if args.mock else ""
    out_path = args.out or f"results/precheck_{args.prompt}{suffix}.jsonl"

    # 把「可预期的失败」变成一行友好提示，而不是一大段 traceback。
    # 对使用者来说，「还没配密钥」和「程序有 bug」是两件事，不该长得一样。
    try:
        run(args.input, args.prompt, out_path, args.mock, args.limit)
    except LLMConfigError as error:
        print(f"\n❌ 配置问题：\n{error}\n", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as error:
        print(f"\n❌ 调用失败：{error}\n", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    sys.exit(main())
