# -*- coding: utf-8 -*-
"""极简 LLM 调用封装。

设计取舍：**只用标准库，零第三方依赖**。
理由：这是要给别人跑的工具，"clone 下来就能跑"比"功能多"更重要。
若日后需要换 SDK，只需替换本文件的 chat_json()。

两种模式
--------
1. 真实调用：读取环境变量 DEEPSEEK_API_KEY，走 DeepSeek 的 OpenAI 兼容接口。
2. mock 模式（--mock）：不联网，用启发式规则伪造输出，仅用于跑通流程与开发调试。

⚠️ 重要：mock 的输出**不是**模型能力，绝不能当作评测结果写进作品集。
本文件会在 mock 输出里强制打上 "MOCK" 标记，评测脚本也会据此拒绝生成正式报告。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

MOCK_MARK = "MOCK"


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def load_dotenv(path: str = ".env") -> None:
    """极简 .env 读取（不引入 python-dotenv）。

    只支持 KEY=VALUE 形式，已存在的环境变量不覆盖。
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class LLMConfigError(RuntimeError):
    """配置缺失或错误（例如没有 API Key）。"""


def get_config(require_key: bool = True) -> dict[str, str]:
    """读取调用配置。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if require_key and not api_key:
        raise LLMConfigError(
            "未找到 DEEPSEEK_API_KEY。\n"
            "解决办法（任选其一）：\n"
            "  1. 在项目根目录新建 .env 文件，写入一行：DEEPSEEK_API_KEY=你的key\n"
            "  2. 在 PowerShell 里临时设置：$env:DEEPSEEK_API_KEY=\"你的key\"\n"
            "  3. 先用 --mock 跑通流程（结果不可用于评测）"
        )
    return {
        "api_key": api_key,
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
    }


# --------------------------------------------------------------------------
# 真实调用
# --------------------------------------------------------------------------
def chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 2,
    timeout: int = 90,
) -> dict[str, Any]:
    """调用模型并要求返回 JSON 对象。

    temperature 默认 0：判定类任务要的是稳定复现，不是创造力。
    """
    config = get_config(require_key=True)
    url = f"{config['base_url']}/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        # 解析失败时追加纠错提示再试：同一个 prompt 重试通常只会拿到同样的坏 JSON
        payload["messages"][1]["content"] = (
            user_prompt if attempt == 0 else user_prompt + _PARSE_HINT
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return _parse_json_loose(content)
        except urllib.error.HTTPError as error:  # 4xx / 5xx
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code in (401, 403):
                raise LLMConfigError(
                    f"API Key 被拒绝（HTTP {error.code}）。请检查 DEEPSEEK_API_KEY 是否正确。\n详情：{detail}"
                ) from error
            if error.code == 429 or error.code >= 500:
                last_error = RuntimeError(f"HTTP {error.code}: {detail}")
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"请求失败 HTTP {error.code}：{detail}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"重试 {max_retries} 次后仍失败：{last_error}")


# 解析失败后的纠错提示（实测：模型偶尔会在字符串里写出未转义的引号）
_PARSE_HINT = (
    "\n\n注意：你上一次的输出不是合法 JSON，解析失败。请只输出**严格合法**的 JSON："
    "所有键与字符串用双引号；字符串内部的引号必须转义（\\\"）；不要有尾随逗号；"
    "不要输出任何解释文字或代码块标记。"
)


def _parse_json_loose(content: str) -> dict[str, Any]:
    """容错解析模型输出。

    三层容错，每一层都对应一类真实出现过的失败：
    1. 直接解析（正常路径）；
    2. 去掉 ```json 代码块包裹；
    3. 用**括号配平扫描**截出第一个完整对象（比贪婪正则可靠），并修掉尾随逗号；
       最后用 raw_decode 兜底，允许对象后面还跟着解释文字。
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    extracted = _extract_balanced_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)

    last_error: Exception | None = None
    for candidate in candidates:
        for fixed in (candidate, _strip_trailing_commas(candidate)):
            try:
                value = json.loads(fixed)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as error:
                last_error = error
                try:  # 对象后面还跟着解释文字时也能救回来
                    value, _ = json.JSONDecoder().raw_decode(fixed.lstrip())
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
    raise last_error or ValueError("无法从模型输出中解析出 JSON 对象")


def _extract_balanced_object(text: str) -> str | None:
    """括号配平扫描：截出第一个完整的 {...}，并正确跳过字符串内部的引号与花括号。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    """去掉对象/数组末尾的多余逗号——最常见的 JSON 语法错误。"""
    return re.sub(r",(\s*[}\]])", r"\1", text)


# --------------------------------------------------------------------------
# mock 调用（仅用于跑通流程）
# --------------------------------------------------------------------------
_MOCK_SIGNALS: dict[str, list[str]] = {
    "acceptance": ["验收", "毫秒", "300ms", "内完成", "可复现", "断言"],
    "edge_cases": ["断线", "重连", "异常", "失败", "为空", "满了", "重复点击", "中途退出"],
    "dependencies": ["依赖", "对接", "后端", "服务端", "美术", "音频", "已确认", "已排期"],
    "assets": ["图标", "动效", "音效", "文案", "素材"],
    "estimation": ["人日", "故事点", "工作量", "天", "估算"],
    "risk_and_cut": ["风险", "延期", "砍", "降级", "兜底"],
}


def mock_chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """伪造一个结构化判定结果。

    规则：命中关键词即视为该项通过。这种"关键词启发式"的准确率必然是假的，
    它的唯一用途是让下游代码（解析、统计、报告）能在没有网络/没有 Key 时跑起来。

    实现细节：只在**需求正文**里找关键词，不能扫整段 prompt——
    否则 v2 的 prompt 里嵌入的规则文本本身会命中所有关键词，导致"全部就绪"的荒谬结果。
    """
    marker = "## 待预审的需求"
    text = user_prompt.split(marker, 1)[1] if marker in user_prompt else user_prompt
    item_verdicts: list[dict[str, Any]] = []
    for item_id, keywords in _MOCK_SIGNALS.items():
        passed = any(keyword in text for keyword in keywords)
        item_verdicts.append(
            {
                "id": item_id,
                "passed": passed,
                "reason": f"[{MOCK_MARK}] 关键词{'命中' if passed else '未命中'}，非真实模型判定",
            }
        )
    # 两项目前无法用关键词伪造，固定为未通过，避免 mock 结果过于乐观
    for item_id in ("one_liner", "core_loop"):
        item_verdicts.append(
            {
                "id": item_id,
                "passed": len(text) > 40,
                "reason": f"[{MOCK_MARK}] 按文本长度粗暴估计，非真实模型判定",
            }
        )
    # 保持与 RUBRIC 的顺序一致
    from rubric import RUBRIC

    order = {item["id"]: index for index, item in enumerate(RUBRIC)}
    item_verdicts.sort(key=lambda entry: order.get(entry["id"], 99))
    return {"items": item_verdicts, "mode": MOCK_MARK}
