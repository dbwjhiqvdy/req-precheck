# -*- coding: utf-8 -*-
"""DoR（需求就绪定义）结构化规则。

设计意图
--------
本文件把《03-检查清单/DoR-需求就绪定义.md》里**给人读的 8 条清单**，
翻译成**机器可判定的结构化规则**。这是"需求预审工具"的地基：
规则写在代码里，评审标准才不会被 prompt 随意漂移。

关键纪律（来自 DoR 原文）
------------------------
1. 8 项全部通过才算「就绪」，勾不满即打回。
2. 「打回」不是否决，而是"材料不齐"，要指明缺哪几项。
3. 判定必须给出理由，理由要能被需求方反驳——否则评审变成黑箱。
"""

from typing import Any

# 每一项包含：
#   id        机器用标识（评测时按 id 统计一致性）
#   name      人读名称（与 DoR 清单一一对应）
#   question  判定问题
#   pass_when 通过标准（给 LLM 的正向判据）
#   fail_hint 典型不通过的样子（给 LLM 的反例，显著提升判定稳定性）
RUBRIC: list[dict[str, str]] = [
    {
        "id": "one_liner",
        "name": "一句话说得清",
        "question": "能否用一句话描述玩家会体验到什么？",
        "pass_when": "需求正文本身包含一句可概括玩家体验的话，读的人不需要额外背景就能复述。",
        "fail_hint": "只有模块名或功能名（如「优化背包」「新增大厅」），没有体验层面的描述。",
    },
    {
        "id": "core_loop",
        "name": "核心循环明确",
        "question": "这条需求接在核心循环的哪一环？玩家做什么 → 得到什么反馈 → 为什么再来一次。",
        "pass_when": "写明了玩家动作、系统反馈，以及它与既有玩法循环的关系。**工程/工具类需求**（不面向玩家体验，例如内部校验脚本）可判为不适用，不计缺失。",
        "fail_hint": "只说明「要做一个系统」，没说玩家为什么要用它、用完得到什么；或面向玩家的需求里完全没有循环描述。",
    },
    {
        "id": "acceptance",
        "name": "验收标准可度量",
        "question": "做完能拿什么去验？必须是可观察的行为，不是「体验更好」。",
        "pass_when": "给出了可观察、可复现的判定条件（含具体数值、状态、或可演示的操作路径）。",
        "fail_hint": "出现「体验更好」「更加流畅」「参照竞品」这类不可验收的表述。",
    },
    {
        "id": "edge_cases",
        "name": "边界与异常已定义",
        "question": "断线、背包满、数值为 0、重复点击、中途退出，会发生什么？",
        "pass_when": "至少写出**一个**具体异常或边界情形的**期望行为**（例如断线、重复操作、物品被他人取走、跨天瞬间提交时的表现）。",
        "fail_hint": "只描述正常路径（happy path），对异常情况一字未提。",
    },
    {
        "id": "dependencies",
        "name": "依赖方已确认",
        "question": "需要谁配合（美术/音频/后端/本地化/运营）？对方是否已确认工期？",
        "pass_when": "明确列出依赖方，且对方**已确认可提供**（出现「已确认」「已答复」「已同意」等表述）。**若本需求无外部依赖，需明确写出**（例如「不需要服务端改动」）。",
        "fail_hint": "只写「需要某方支持」「等美术出图」「尚未给出排期」这类没有确认的说法；或既未列依赖方、也未说明无依赖。",
    },
    {
        "id": "assets",
        "name": "素材到位或已有排期",
        "question": "图标、动效、音效、文案是谁出、什么时候出？",
        "pass_when": "涉及素材时，说明了来源与时间；或明确本需求不需要新素材。",
        "fail_hint": "需要新素材但完全没提由谁产出、何时产出。",
    },
    {
        "id": "estimation",
        "name": "工作量已估算且拆到 ≤3 天",
        "question": "有没有估算？是否拆到 3 天以内的粒度？",
        "pass_when": "给出了工作量估计（人日或故事点），且单块不超过 3 天。",
        "fail_hint": "完全没有估算，或明显是一个跨越多周的大块工作。",
    },
    {
        "id": "risk_and_cut",
        "name": "风险与砍法已写明",
        "question": "已知风险是什么？如果延期，第一个砍掉的是哪部分？",
        "pass_when": "指出了至少一个风险，并说明延期时的裁剪优先级。",
        "fail_hint": "通篇只讲目标与收益，对风险与裁剪只字未提。",
    },
]

# 便于按 id 取用
RUBRIC_BY_ID: dict[str, dict[str, str]] = {item["id"]: item for item in RUBRIC}


def render_rubric() -> str:
    """把规则渲染成一段适合放进 prompt 的文本。"""
    lines: list[str] = []
    for index, item in enumerate(RUBRIC, start=1):
        lines.append(f"{index}. [{item['id']}] {item['name']}")
        lines.append(f"   - 判定问题：{item['question']}")
        lines.append(f"   - 通过标准：{item['pass_when']}")
        lines.append(f"   - 典型不通过：{item['fail_hint']}")
    return "\n".join(lines)


def overall_ready(item_verdicts: list[dict[str, Any]]) -> bool:
    """按 DoR 纪律判定整体是否就绪：8 项必须全部通过。

    item_verdicts 形如 [{"id": "one_liner", "passed": True, "reason": "..."}, ...]
    """
    if len(item_verdicts) != len(RUBRIC):
        # 缺项一律视为未通过——宁可打回，不可漏放
        return False
    return all(bool(item.get("passed")) for item in item_verdicts)


def missing_items(item_verdicts: list[dict[str, Any]]) -> list[str]:
    """返回未通过的项 id 列表，供「打回原因」使用。"""
    return [item["id"] for item in item_verdicts if not item.get("passed")]


if __name__ == "__main__":
    # 直接运行本文件可自检规则完整性
    print(f"规则项数：{len(RUBRIC)}（DoR 要求 8 项）")
    assert len(RUBRIC) == 8, "规则项数必须与 DoR 清单一致"
    ids = [item["id"] for item in RUBRIC]
    assert len(set(ids)) == len(ids), "存在重复的规则 id"
    print("\n" + render_rubric())
    # 自检：只要有任意一项未通过，整体就不就绪
    all_pass = [{"id": item["id"], "passed": True} for item in RUBRIC]
    one_fail = [{"id": item["id"], "passed": item["id"] != "edge_cases"} for item in RUBRIC]
    assert overall_ready(all_pass) is True
    assert overall_ready(one_fail) is False
    assert missing_items(one_fail) == ["edge_cases"]
    print("\n自检通过：规则完整、判定逻辑符合「8 项全通过才算就绪」。")
