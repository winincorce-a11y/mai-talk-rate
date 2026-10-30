"""MaiBot 插件：发言频率控制（talk-rate）。

把 MaiBot 原版「只能 0.01 起步」的群聊回复频率，换成可以精确到 0 的逐会话规则：

- **@ 才算说话**：某个群只有被 @ / 被提到时才回复，平时聊天内容完全不会触发回复。
- **任意频率**：0 / 0.0x / 百分比 / 关 / 开，都能精确设置，不再被 0.01 兜底卡住。
- **按群下规则**：每个群一套规则，互不影响，支持 ``*`` 通配与私聊、白名单。
- **立即生效**：不需要重启，规则热更新。

原版为什么做不到
----------------
``src/maisaka/runtime.py::_get_effective_reply_frequency`` 一旦返回 0，调度器
(``turn_trigger/scheduler.py``) 会先命中「静默消费」分支，而
``reasoning_engine.py::_handle_silent_turn`` 里第一件事就是
``_clear_forced_turn_state()``——@ 触发在更后面才判断，于是连 @ 都被吞掉。
所以「频率 0」在原版等于彻底闭嘴。

本插件的做法
------------
不在频率数值上硬碰，而是让**规则自带的放行条件**绕开频率系统，并在两处兜底：

1. ``chat.receive.after_process``：会话不允许回复时直接 abort，消息不进上下文、不调模型。
2. ``maisaka.planner.before_request``：兜底把 ``reply`` 工具从本轮工具表里摘掉，
   即使消息因为 @ 之类的原因进了规划器，也不会真的发出回复。

只使用 ``@HookHandler`` / ``@Command`` / ``@Tool``，不修改 MaiBot 主程序。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import contextlib
import json
import logging
import os
import random
import re
import time

from maibot_sdk import (
    Command,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import (
    ErrorPolicy,
    HookMode,
    HookOrder,
    ToolParameterInfo,
    ToolParamType,
)

__all__ = [
    "ChatRef",
    "ChatRule",
    "RateState",
    "TalkRatePlugin",
    "create_plugin",
    "parse_group_rules",
    "parse_rate",
]

PLUGIN_ID = "github.winincorce-a11y.talk-rate"

#: 规划器里负责真正发出回复的工具名。摘掉它，本轮就不会说话。
_REPLY_TOOL_NAME = "reply"

#: 插件自己的控制命令，**永远**穿透所有限制，否则设置了 0 之后就再也调不回来。
_CONTROL_COMMAND_RE = re.compile(
    r"^\s*[/／]\s*(?:tr|talk[-_ ]?rate|发言频率|说话频率|频率)(?=\s|$)",
    re.IGNORECASE,
)

#: 以斜杠开头、且斜杠后面紧跟「命令名首字符」的写法。
#: 之所以要求紧跟字符而不是 ``\S``，是为了排除 ``//哈哈`` 这类闲聊——
#: 旧写法会让任意 ``/`` 开头的闲聊都打开命令豁免窗口。
_SLASH_COMMAND_RE = re.compile(r"^\s*[/／]\s*[A-Za-z0-9_\-\u4e00-\u9fff]")

#: 常见「不带斜杠」的命令触发词默认值。第三方插件里有大量这种写法
#: （例如随机图片插件的 ``随机图片`` / ``来张图``），不能只认斜杠。
#: 用户可以在配置文件里追加，避免硬编码漏判。
_DEFAULT_COMMAND_KEYWORDS = (
    "随机图片",
    "来张图",
    "来点图",
    "来张图片",
    "来点图片",
    "涩图",
    "随机涩图",
    "setu",
    "点歌",
    "搜图",
    "签到",
    "今日运势",
    "画图",
)


def _looks_like_command(text: str, keywords: "Iterable[str]" = ()) -> bool:
    """判断这条消息看起来像不像是某个插件的命令。

    宁可**宽进**：把普通聊天误判成命令，最坏结果是这一条不静默；
    把命令漏判成聊天，会导致别的插件功能整体失效——后者严重得多。
    """

    stripped = text.strip()
    if not stripped:
        return False
    if _SLASH_COMMAND_RE.match(stripped):
        return True
    lowered = stripped.lower()
    for word in keywords:
        word = str(word).strip()
        if word and lowered.startswith(word.lower()):
            return True
    return False


#: 这些消息组件说明「这条出站不是普通聊天文本」。
_MEDIA_SEGMENT_TYPES = frozenset({"image", "emoji", "voice", "file", "forward", "card", "json", "xml"})


def _has_media_segment(message: Any) -> bool:
    """判断一条出站消息里是否包含图片/表情/转发等非文本内容。"""

    if not isinstance(message, dict):
        return False
    raw = message.get("raw_message")
    if not isinstance(raw, list):
        return False
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() in _MEDIA_SEGMENT_TYPES:
            return True
    return False


def _preview_text(message: Any, limit: int = 60) -> str:
    """截取一段出站文本用于日志，避免刷屏或泄漏过长内容。"""

    if not isinstance(message, dict):
        return ""
    text = _as_str(message.get("processed_plain_text"))
    if not text:
        raw = message.get("raw_message")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and str(item.get("type") or "") == "text":
                    text = _as_str(item.get("data"))
                    if text:
                        break
    text = text.replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _as_str(value: Any) -> str:
    """把任意值压成干净的字符串；容器和 None 一律返回空串。"""

    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if float(value).is_integer() else str(value)
    return ""


def _as_float(value: Any) -> Optional[float]:
    """尽量把值转成 float，失败返回 None。"""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _as_str(value)
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    """把配置里的「像布尔值的东西」统一成 bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _as_str(value).lower()
    if text in {"1", "true", "yes", "on", "open", "enable", "enabled", "开", "是"}:
        return True
    if text in {"0", "false", "no", "off", "close", "disable", "disabled", "关", "否"}:
        return False
    return default


def _as_list(value: Any) -> list[Any]:
    """把配置里的列表/逗号串统一成 list。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，\s]+", value) if item.strip()]
    return [value]


def _cfg_get(section: Any, key: str, default: Any = None) -> Any:
    """同时兼容 ``PluginConfigBase`` 对象和 dict 的字段读取。"""

    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _fmt_rate(rate: float) -> str:
    """把频率格式化成人看的样子：整数就不带小数点。"""

    if rate <= 0:
        return "0（只在允许的触发条件时才回复）"
    if rate >= 1:
        return "1（每条都考虑回复）"
    return f"{rate:g}（约 {rate * 100:.1f}%）"


# ---------------------------------------------------------------------------
# 频率解析
# ---------------------------------------------------------------------------


def parse_rate(raw: Any) -> Optional[float]:
    """解析一条频率写法。

    支持：``0``、``0.05``、``5%``、``off``、``关``、``on``、``开``、``default``…
    返回 ``0.0``~``1.0`` 的浮点数；无法识别时返回 ``None``。
    """

    text = _as_str(raw).lower()
    if not text:
        return None
    if text in {"off", "close", "disable", "disabled", "never", "mute", "闭嘴", "关", "关闭", "不说"}:
        return 0.0
    if text in {"on", "open", "enable", "enabled", "normal", "开", "开启", "正常"}:
        return 1.0
    percent = text.endswith("%") or text.endswith("％")
    if percent:
        text = text[:-1].strip()
    value = _as_float(text)
    if value is None:
        return None
    if percent:
        value /= 100.0
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 会话标识
# ---------------------------------------------------------------------------

_CHAT_KEYS = ("session_id", "sessionId", "chat_id", "chatId", "stream_id", "streamId")
_GROUP_KEYS = ("group_id", "groupId", "group")
_PLATFORM_KEYS = ("platform", "platform_id")
_USER_KEYS = ("user_id", "userId")


def _extract_sources(payload: Any) -> list[dict[str, Any]]:
    """从 Hook/命令参数里收集所有可能承载会话信息的 dict。"""

    if not isinstance(payload, dict):
        return []
    sources: list[dict[str, Any]] = [payload]
    message = payload.get("message")
    if isinstance(message, dict):
        sources.append(message)
        message_info = message.get("message_info")
        if isinstance(message_info, dict):
            group_info = message_info.get("group_info")
            if isinstance(group_info, dict):
                sources.append(group_info)
            user_info = message_info.get("user_info")
            if isinstance(user_info, dict):
                sources.append(user_info)
    return sources


def _first(sources: Iterable[dict[str, Any]], keys: Iterable[str]) -> str:
    for source in sources:
        for key in keys:
            value = _as_str(source.get(key))
            if value and value.lower() not in {"0", "none", "null"}:
                return value
    return ""


@dataclass(frozen=True)
class ChatRef:
    """一个聊天流的身份，以及从消息里看到的内容。"""

    chat_id: str
    group_id: str
    platform: str
    user_id: str
    text: str
    is_at: bool
    is_mentioned: bool
    is_private: bool

    @property
    def keys(self) -> tuple[str, ...]:
        """所有可以用来匹配规则的会话标识，按优先级排序。"""

        ordered: list[str] = []
        for value in (self.group_id, self.chat_id, self.user_id):
            if value and value not in ordered:
                ordered.append(value)
        return tuple(ordered)

    @property
    def primary_key(self) -> str:
        return self.keys[0] if self.keys else ""

    @classmethod
    def from_payload(cls, payload: Any) -> "ChatRef":
        sources = _extract_sources(payload)
        chat_id = _first(sources, _CHAT_KEYS)
        group_id = _first(sources, _GROUP_KEYS)
        platform = _first(sources, _PLATFORM_KEYS)
        user_id = _first(sources, _USER_KEYS)

        is_at = False
        is_mentioned = False
        text = ""
        for source in sources:
            if source.get("is_at"):
                is_at = True
            if source.get("is_mentioned"):
                is_mentioned = True
            if not text:
                candidate = _as_str(source.get("processed_plain_text")) or _as_str(source.get("plain_text"))
                if candidate:
                    text = candidate

        # 没有任何会话标识时，退化成「只按规则里的全局配置处理」，不要抛错。
        is_private = not group_id and bool(user_id)
        return cls(
            chat_id=chat_id,
            group_id=group_id,
            platform=platform,
            user_id=user_id,
            text=text,
            is_at=is_at,
            is_mentioned=is_mentioned,
            is_private=is_private,
        )


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


@dataclass
class ChatRule:
    """一个会话生效的规则。"""

    rate: float = 1.0
    source: str = "默认"

    @property
    def at_only(self) -> bool:
        return self.rate <= 0.0


@dataclass
class RateState:
    """插件的持久化状态：逐会话的临时频率覆盖与放行计数。"""

    overrides: dict[str, float] = field(default_factory=dict)
    allowed_at: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"overrides": dict(self.overrides), "allowed_at": dict(self.allowed_at)}

    @classmethod
    def from_dict(cls, raw: Any) -> "RateState":
        state = cls()
        if not isinstance(raw, dict):
            return state
        for key, value in (raw.get("overrides") or {}).items():
            parsed = parse_rate(value)
            if isinstance(key, str) and parsed is not None:
                state.overrides[key] = parsed
        for key, value in (raw.get("allowed_at") or {}).items():
            moment = _as_float(value)
            if isinstance(key, str) and moment is not None:
                state.allowed_at[key] = moment
        return state


#: 入站判定结果的保鲜期。规划器只可能在本批消息到达后很快运行，
#: 超过这个时间就认为记忆过期，按「没有 @」处理，避免把很久以前的 @ 一直算数。
_INBOUND_MEMORY_TTL_SECONDS = 180.0

#: 每个会话最多缓存多少条入站判定，避免无界增长。
_INBOUND_MEMORY_MAX_SESSIONS = 2000

#: 命令豁免窗口的默认值（秒）。命令可能分多条发送（先回执、再结果、最后总结），
#: 每次命中都会顺延窗口，避免第二条消息被出站闸门吞掉。
#: 实际取值来自配置 ``outbound_guard.command_window_seconds``。
_DEFAULT_COMMAND_WINDOW_SECONDS = 60.0


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class PluginSectionConfig(PluginConfigBase):
    """插件开关。"""

    __ui_label__ = "插件"
    __ui_icon__ = "sliders"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="关掉后插件完全不生效，麦麦按 MaiBot 原版频率行为说话。",
        json_schema_extra={"label": "启用插件"},
    )
    config_version: str = Field(
        default="1.1.1",
        description="配置结构版本号，由插件自动维护，升级插件时会自动迁移，请不要手动修改。",
        json_schema_extra={"label": "配置版本", "disabled": True, "advanced": True},
    )


class DefaultRuleConfig(PluginConfigBase):
    """没有命中任何规则时使用的默认行为。"""

    __ui_label__ = "默认规则"
    __ui_icon__ = "message-circle"
    __ui_order__ = 1

    group_rate: str = Field(
        default="default",
        description=(
            "没有写进「群规则」的群，默认套用这个频率。\n"
            "default = 沿用 MaiBot 原本的频率设置（插件不干预）；\n"
            "0 = 没人 @、也没命中关键词时就不说话；\n"
            "也可以填 0.05 或 5% 表示约 5% 概率回复。"
        ),
        json_schema_extra={
            "label": "群聊默认频率",
            "placeholder": "default",
            "hint": "可选：default / 0 / 0.05 / 5% / on / off",
        },
    )
    private_rate: str = Field(
        default="default",
        description=(
            "私聊时使用的频率。写法与「群聊默认频率」相同。\n"
            "一般保持 default 即可，私聊本就该有问必答。"
        ),
        json_schema_extra={
            "label": "私聊默认频率",
            "placeholder": "default",
            "hint": "可选：default / 0 / 0.05 / 5% / on / off",
        },
    )
    blocked_whitelist: str = Field(
        default="",
        description=(
            "这里的群或个人永远不受频率限制，无论规则怎么设都会正常回复。\n"
            "填群号或 QQ 号，多个用逗号分隔，例如：123456789, 987654321。\n"
            "留空表示不启用白名单。"
        ),
        json_schema_extra={
            "label": "白名单",
            "placeholder": "123456789, 987654321",
            "hint": "群号 / QQ 号，逗号分隔；留空表示不启用",
        },
    )


class GroupRuleConfig(PluginConfigBase):
    """逐群规则。"""

    __ui_label__ = "群规则"
    __ui_icon__ = "users"
    __ui_order__ = 2

    rules: str = Field(
        default="",
        description=(
            "给指定群单独设置频率，一行一个，格式「群号=频率」。\n"
            "从下往上匹配，写在前面的优先；没匹配到的群走「默认规则」。\n"
            "示例：\n"
            "123456789=0          ← 这个群只有被 @ 或命中关键词才回复\n"
            "987654321=0.05       ← 这个群约 5% 概率回复\n"
            "*=default            ← 其余所有群都按原版频率\n"
            "支持 # 或 // 开头的注释。"
        ),
        json_schema_extra={
            "label": "逐群频率规则",
            "x-widget": "textarea",
            "rows": 6,
            "placeholder": "123456789=0\n987654321=0.05\n*=default",
            "hint": "格式：群号=频率；*=default 表示其余群不干预",
        },
    )
    at_bypass: bool = Field(
        default=True,
        description=(
            "频率为 0 的群里，有人 @ 麦麦或叫它的名字时，是否仍然回复。\n"
            "建议保持开启，否则被点名也完全不理人。"
        ),
        json_schema_extra={"label": "被 @ 时仍然回复"},
    )
    keywords: str = Field(
        default="",
        description=(
            "关键词放行：消息里出现这些词时，即使该群频率为 0 也会回复。\n"
            "一行一个，或用逗号分隔。示例：\n"
            "麦麦\n"
            "你好，在吗，帮我看看\n"
            "支持 # 或 // 开头的注释。留空表示不启用关键词放行。"
        ),
        json_schema_extra={
            "label": "关键词放行",
            "x-widget": "textarea",
            "rows": 5,
            "placeholder": "麦麦\n你好，在吗",
            "hint": "每行一个或用逗号分隔；命中即回复",
        },
    )
    keyword_match: Literal["contains", "prefix", "exact", "regex"] = Field(
        default="contains",
        description=(
            "关键词的匹配方式：\n"
            "contains = 消息里包含关键词就算命中（推荐）；\n"
            "prefix = 消息必须以关键词开头；\n"
            "exact = 消息必须和关键词完全一样；\n"
            "regex = 关键词按正则表达式解释，适合复杂匹配。"
        ),
        json_schema_extra={"label": "关键词匹配方式", "hint": "contains 最宽松，regex 最灵活"},
    )
    keyword_case_sensitive: bool = Field(
        default=False,
        description="开启后英文关键词区分大小写（Hello 与 hello 视为不同）。中文不受影响。",
        json_schema_extra={"label": "关键词区分大小写"},
    )


class OutboundGuardConfig(PluginConfigBase):
    """出站闸门：真正拦住「不该说的闲聊回复」。"""

    __ui_label__ = "出站闸门"
    __ui_icon__ = "shield"
    __ui_order__ = 3

    enabled: bool = Field(
        default=True,
        description=(
            "总开关。关闭后插件只做入站判定与状态记录，不再拦截任何出站消息。\n"
            "如果你发现某个插件的结果被误拦，可以先关掉它来确认是不是这里的问题。"
        ),
        json_schema_extra={"label": "启用出站拦截"},
    )
    command_keywords: str = Field(
        default="",
        description=(
            "额外把哪些开头词视为「其他插件的命令」，从而放行它们的输出。\n"
            "以斜杠开头的命令（/xxx）总是自动放行，无需在此登记。\n"
            "这里用于补充**不带斜杠**的触发词，例如随机图片插件的「随机图片」「来张图」。\n"
            "一行一个或用逗号分隔；留空则只使用插件内置的常见触发词。"
        ),
        json_schema_extra={
            "label": "额外命令触发词",
            "x-widget": "textarea",
            "rows": 4,
            "placeholder": "随机图片\n来张图\n点歌",
            "hint": "补上你自己装的插件的触发词，避免它们的输出被误拦",
        },
    )
    command_window_seconds: int = Field(
        default=60,
        description=(
            "识别到命令后，多长时间内的出站消息视为该命令的结果并放行。\n"
            "命令常常先回一条「正在查询」再发结果，下载类还要补总结，所以需要一段窗口。\n"
            "调小可以减少误放行，调大可以避免长任务的结果被拦。"
        ),
        json_schema_extra={
            "label": "命令豁免窗口（秒）",
            "min": 5,
            "max": 600,
            "step": 5,
            "hint": "建议 30 ～ 120 秒",
        },
    )


class TalkRateConfig(PluginConfigBase):
    """发言频率控制插件的完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    default_rule: DefaultRuleConfig = Field(default_factory=DefaultRuleConfig)
    group_rules: GroupRuleConfig = Field(default_factory=GroupRuleConfig)
    outbound_guard: OutboundGuardConfig = Field(default_factory=OutboundGuardConfig)


# ---------------------------------------------------------------------------
# 规则解析
# ---------------------------------------------------------------------------


def parse_group_rules(raw: Any) -> list[tuple[str, float]]:
    """解析「群号=频率」规则文本，返回按书写顺序排列的 ``(模式, 频率)``。"""

    parsed: list[tuple[str, float]] = []
    for line in re.split(r"[\r\n;；]+", _as_str(raw)):
        entry = line.strip()
        if not entry or entry.startswith(("#", "//")):
            continue
        # 去掉行尾注释
        entry = re.split(r"\s*(?:#|//)\s*", entry, maxsplit=1)[0].strip()
        if not entry:
            continue
        for separator in ("=", "：", ":"):
            if separator in entry:
                pattern, _, value = entry.partition(separator)
                break
        else:
            parts = entry.split()
            if len(parts) < 2:
                continue
            pattern, value = parts[0], parts[1]
        pattern = pattern.strip().strip("'\"")
        rate = parse_rate(value)
        if not pattern or rate is None:
            continue
        parsed.append((pattern, rate))
    return parsed


def _match_pattern(pattern: str, keys: Iterable[str]) -> bool:
    """精确匹配或 ``*`` 通配匹配。"""

    keys = [key for key in keys if key]
    if not keys:
        return False
    for key in keys:
        if pattern == "*":
            return True
        if pattern == key:
            return True
        if "*" in pattern or "?" in pattern:
            regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
            if re.match(regex, key, re.IGNORECASE):
                return True
    return False


#: 支持的关键词匹配方式。
_KEYWORD_MODES = frozenset({"contains", "prefix", "exact", "regex"})


def parse_keywords(raw: Any) -> list[str]:
    """解析关键词配置：一行一个或逗号分隔，支持 ``#`` / ``//`` 注释。"""

    keywords: list[str] = []
    for line in re.split(r"[\r\n]+", _as_str(raw)):
        entry = re.split(r"\s*(?:#|//)\s*", line, maxsplit=1)[0].strip()
        if not entry:
            continue
        for item in re.split(r"[,，;；]", entry):
            word = item.strip().strip("'\"")
            if word and word not in keywords:
                keywords.append(word)
    return keywords


def normalize_keyword_mode(raw: Any) -> str:
    """把匹配方式规范成四种之一，非法值退回 ``contains``。"""

    mode = _as_str(raw).strip().lower() or "contains"
    return mode if mode in _KEYWORD_MODES else "contains"


def match_keyword(text: str, keywords: Iterable[str], mode: str, case_sensitive: bool) -> Optional[str]:
    """返回命中的第一个关键词；没有命中返回 ``None``。"""

    haystack = text if case_sensitive else text.lower()
    flags = 0 if case_sensitive else re.IGNORECASE
    for word in keywords:
        needle = word if case_sensitive else word.lower()
        if not needle:
            continue
        if mode == "exact":
            if haystack.strip() == needle:
                return word
        elif mode == "prefix":
            if haystack.strip().startswith(needle):
                return word
        elif mode == "regex":
            try:
                if re.search(word, text, flags):
                    return word
            except re.error:
                # 正则写错不能拖垮整条消息判定，跳过即可。
                continue
        else:  # contains
            if needle in haystack:
                return word
    return None


# ---------------------------------------------------------------------------
# 插件主体
# ---------------------------------------------------------------------------


class TalkRatePlugin(MaiBotPlugin):
    """发言频率控制插件。"""

    config_model = TalkRateConfig

    def __init__(self) -> None:
        super().__init__()
        self._state = RateState()
        self._state_loaded = False
        self._last_reason: str = "尚未判定"
        self._stats: dict[str, int] = {"blocked": 0, "allowed": 0, "silenced": 0}
        #: session_id -> (是否允许回复, 判定原因, 记录时刻)
        self._inbound_memory: dict[str, tuple[bool, str, float]] = {}
        #: session_id -> 豁免截止时刻；本插件自己发出的命令回复不走出站拦截
        self._control_reply_until: dict[str, float] = {}

    # -- 基础设施 -----------------------------------------------------------

    def _logger(self) -> logging.Logger:
        try:
            return self.ctx.logger
        except Exception:
            return logging.getLogger(f"plugin.{PLUGIN_ID}")

    def _enabled(self) -> bool:
        return _as_bool(_cfg_get(self.config.plugin, "enabled", True), True)

    # -- 规则计算 -----------------------------------------------------------

    def _default_rate(self, chat: ChatRef) -> ChatRule:
        section = self.config.default_rule
        if chat.is_private:
            raw = _cfg_get(section, "private_rate", "default")
            rate = parse_rate(raw)
            if rate is None:
                return ChatRule(rate=1.0, source="私聊默认（原版）")
            return ChatRule(rate=rate, source="私聊默认")
        raw = _cfg_get(section, "group_rate", "default")
        rate = parse_rate(raw)
        if rate is None:
            return ChatRule(rate=1.0, source="群聊默认（原版）")
        return ChatRule(rate=rate, source="群聊默认")

    def _whitelisted(self, chat: ChatRef) -> bool:
        raw = _cfg_get(self.config.default_rule, "blocked_whitelist", "")
        patterns = [_as_str(item) for item in _as_list(raw)]
        patterns = [item for item in patterns if item]
        if not patterns:
            return False
        return any(_match_pattern(pattern, chat.keys) for pattern in patterns)

    def _override_keys(self, chat: ChatRef) -> list[str]:
        """返回当前会话所有可能被用作临时设置的键。

        同一个群在不同链路里的标识并不统一：
        - 命令 / 入站消息里能拿到 ``group_id``（群号）；
        - 规划器 Hook 只给 ``session_id``（主程序算出来的哈希）。
        两个都要认，否则 ``/tr 0`` 写进去的键永远查不到。

        注意**不要把群成员 ``user_id`` 当成会话键**：群里的发言人不是会话标识，
        把它写进 override 会连带把这个人的私聊一起关掉。
        """

        candidates: list[str] = []
        for value in (chat.chat_id, chat.group_id):
            if value and value not in candidates:
                candidates.append(value)
        # 私聊没有 group_id，此时才用 user_id 兜底。
        if chat.is_private and chat.user_id and chat.user_id not in candidates:
            candidates.append(chat.user_id)
        return candidates

    def resolve_rule(self, chat: ChatRef) -> ChatRule:
        """按「临时覆盖 → 白名单 → 逐群规则 → 默认规则」的顺序给出最终规则。"""

        for key in self._override_keys(chat):
            if key in self._state.overrides:
                return ChatRule(rate=self._state.overrides[key], source=f"临时设置（{key}）")

        if self._whitelisted(chat):
            return ChatRule(rate=1.0, source="白名单")

        # 群规则只作用于群聊：私聊由 private_rate 单独控制，
        # 避免一条 `*=0` 把私聊也一起关掉。
        if not chat.is_private:
            for pattern, rate in parse_group_rules(_cfg_get(self.config.group_rules, "rules", "")):
                if _match_pattern(pattern, chat.keys):
                    return ChatRule(rate=rate, source=f"群规则（{pattern}）")

        return self._default_rate(chat)

    # -- 状态持久化 ---------------------------------------------------------

    def _state_path(self) -> Optional[Path]:
        try:
            data_dir = getattr(self.ctx.paths, "data_dir", None)
            if not data_dir:
                return None
            return Path(str(data_dir)) / "state.json"
        except Exception:
            return None

    def _load_state(self) -> None:
        if self._state_loaded:
            return
        self._state_loaded = True
        path = self._state_path()
        if path is None or not path.exists():
            return
        try:
            self._state = RateState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            # 不要静默丢掉用户设置：把坏文件留档，并明确告知恢复方式。
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            moved = False
            try:
                path.replace(backup)
                moved = True
            except Exception:
                pass
            if moved:
                self._logger().error(
                    "发言频率控制：状态文件无法解析，已备份为 %s，本次以空状态启动；"
                    "修复该文件并重命名回 %s 即可恢复之前的临时设置",
                    backup,
                    path.name,
                )
            else:
                self._logger().error(
                    "发言频率控制：状态文件无法解析且无法备份（%s），本次以空状态启动", path
                )

    def _save_state(self) -> None:
        """原子写入状态文件：先写临时文件再 ``os.replace``。

        直接 ``write_text`` 覆盖时，进程在写到一半崩溃会留下半截 JSON，
        下次启动就读不回来，用户视角是「我设的规则莫名没了」。
        """

        path = self._state_path()
        if path is None:
            return
        tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_path, path)
        except Exception:
            self._logger().error("发言频率控制：状态保存失败 %s", path)
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)

    # -- 核心判定 -----------------------------------------------------------

    def _match_keywords(self, text: str) -> Optional[str]:
        """按配置匹配关键词，返回命中的词；未命中返回 None。"""

        if not text:
            return None
        section = self.config.group_rules
        keywords = parse_keywords(_cfg_get(section, "keywords", ""))
        if not keywords:
            return None
        mode = normalize_keyword_mode(_cfg_get(section, "keyword_match", "contains"))
        case_sensitive = _as_bool(_cfg_get(section, "keyword_case_sensitive", False), False)
        return match_keyword(text, keywords, mode, case_sensitive)

    def decide(self, chat: ChatRef) -> tuple[bool, str]:
        """返回 ``(是否允许本轮回复, 原因说明)``。"""

        if not self._enabled():
            return True, "插件已停用"

        if _CONTROL_COMMAND_RE.match(chat.text):
            return True, "控制命令，永远放行"

        rule = self.resolve_rule(chat)
        if rule.rate >= 1.0:
            return True, f"{rule.source}：频率={_fmt_rate(rule.rate)}"

        # 关键词放行：优先级放在 @ 之上——
        # 有人喊了关键词就说明「在叫麦麦」，无论频率多低都应该回应。
        hit = self._match_keywords(chat.text)
        if hit is not None:
            return True, f"{rule.source}：频率={_fmt_rate(rule.rate)}，命中关键词「{hit}」，放行"

        if rule.rate <= 0.0:
            at_bypass = _as_bool(_cfg_get(self.config.group_rules, "at_bypass", True), True)
            if at_bypass and (chat.is_at or chat.is_mentioned):
                trigger = "@" if chat.is_at else "被提及"
                return True, f"{rule.source}：频率=0，但本轮命中{trigger}，放行"
            return False, f"{rule.source}：频率=0，且未命中 @/提及/关键词，不回复"

        # 0 < rate < 1：按概率放行，但 @/提及永远放行（与原版 inevitable_at_reply 一致）。
        if chat.is_at or chat.is_mentioned:
            trigger = "@" if chat.is_at else "被提及"
            return True, f"{rule.source}：频率={_fmt_rate(rule.rate)}，本轮命中{trigger}，放行"

        if random.random() < rule.rate:
            return True, f"{rule.source}：频率={_fmt_rate(rule.rate)}，随机放行"
        return False, f"{rule.source}：频率={_fmt_rate(rule.rate)}，随机未命中"

    # -- 入站判定记忆 -------------------------------------------------------

    def _remember_inbound(self, session_id: str, allowed: bool, reason: str) -> None:
        """记录本轮入站判定，供规划器阶段复用。"""

        if not session_id:
            return
        now = time.time()
        self._inbound_memory[session_id] = (allowed, reason, now)
        # 顺手清理过期项，顺便把字典规模压住。
        if len(self._inbound_memory) > _INBOUND_MEMORY_MAX_SESSIONS:
            expire_before = now - _INBOUND_MEMORY_TTL_SECONDS
            for key in [k for k, v in self._inbound_memory.items() if v[2] < expire_before]:
                self._inbound_memory.pop(key, None)
            while len(self._inbound_memory) > _INBOUND_MEMORY_MAX_SESSIONS:
                self._inbound_memory.pop(next(iter(self._inbound_memory)), None)

    def _recall_inbound(self, session_id: str) -> Optional[tuple[bool, str]]:
        """取回本轮的入站判定；过期或不存在时返回 None。"""

        if not session_id:
            return None
        entry = self._inbound_memory.get(session_id)
        if entry is None:
            return None
        allowed, reason, recorded_at = entry
        if time.time() - recorded_at > _INBOUND_MEMORY_TTL_SECONDS:
            self._inbound_memory.pop(session_id, None)
            return None
        return allowed, reason

    # -- 控制回复豁免 -------------------------------------------------------

    def _command_window_seconds(self) -> float:
        """命令豁免窗口长度，取自配置并做上下限保护。"""

        raw = _cfg_get(self.config.outbound_guard, "command_window_seconds", 60)
        value = _as_float(raw)
        if value is None:
            return _DEFAULT_COMMAND_WINDOW_SECONDS
        return max(5.0, min(600.0, value))

    def _command_keywords(self) -> list[str]:
        """内置常见触发词 + 用户追加的触发词。"""

        extra = parse_keywords(_cfg_get(self.config.outbound_guard, "command_keywords", ""))
        merged: list[str] = []
        for word in (*_DEFAULT_COMMAND_KEYWORDS, *extra):
            if word and word not in merged:
                merged.append(word)
        return merged

    def _outbound_guard_enabled(self) -> bool:
        return _as_bool(_cfg_get(self.config.outbound_guard, "enabled", True), True)

    def _mark_control_reply(self, session_id: str, seconds: Optional[float] = None) -> None:
        """登记「接下来这段时间的出站属于命令/本插件的回复」。"""

        if not session_id:
            return
        window = self._command_window_seconds() if seconds is None else seconds
        self._control_reply_until[session_id] = time.time() + window
        if len(self._control_reply_until) > _INBOUND_MEMORY_MAX_SESSIONS:
            now = time.time()
            for key in [k for k, v in self._control_reply_until.items() if v < now]:
                self._control_reply_until.pop(key, None)

    def _is_control_reply(self, session_id: str) -> bool:
        """判断这条出站是否属于命令回复；命中后不消费，而是顺延豁免窗口。

        命令/插件常常先发一条「正在查询」，几秒后再发结果，下载任务还会在
        图片之后补一条总结。命中一次就 ``pop`` 会导致第二条消息被当成主动闲聊拦掉。
        """

        if not session_id:
            return False
        deadline = self._control_reply_until.get(session_id)
        if deadline is None:
            return False
        now = time.time()
        if now > deadline:
            self._control_reply_until.pop(session_id, None)
            return False
        self._control_reply_until[session_id] = now + self._command_window_seconds()
        return True

    # -- Hook：入站闸门 -----------------------------------------------------

    @HookHandler(
        "chat.receive.after_process",
        name="talk_rate_inbound",
        description="按会话规则决定是否允许本轮消息进入回复链路；频率为 0 时只有 @ 能触发",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=2000,
    )
    async def gate_inbound(self, **kwargs: Any) -> dict[str, Any]:
        """只做「记录判定结果」，不再 abort。

        为什么不再 abort：``chat.receive.after_process`` 的 abort 发生在
        ``bot.py`` 的命令链之前，会把**所有**其他插件的命令一起掐掉
        （例如随机图片插件的 ``随机图片`` 触发词）。本插件只负责「说不说话」，
        不应该替别的插件决定它们收不收得到消息。

        真正的静默由 ``gate_planner`` 摘掉 ``reply`` 工具完成：
        命令照常执行，只是普通聊天不再触发回复。
        """

        try:
            self._load_state()
            chat = ChatRef.from_payload(kwargs)
            allowed, reason = self.decide(chat)
            self._last_reason = reason
            if allowed:
                self._stats["allowed"] += 1
            else:
                self._stats["blocked"] += 1
                self._logger().debug("发言频率控制：本轮不回复（%s，会话 %s）", reason, chat.primary_key)
            # 规划器阶段拿不到本批消息，只能靠这里把「这次到底 @ 没 @ 我」记下来。
            self._remember_inbound(chat.chat_id or chat.primary_key, allowed, reason)

            # 命令消息：它的输出属于「别的插件/主程序在干活」，不是麦麦在闲聊。
            # 记一个豁免，否则零频率会话会把别人的命令结果一起吞掉。
            if _looks_like_command(chat.text, self._command_keywords()):
                target = chat.chat_id or chat.primary_key
                self._mark_control_reply(target)
                self._logger().debug("发言频率控制：识别为命令，放行其输出（%s）", target)
            else:
                # 来了新的普通消息，说明上一轮命令早已结束；
                # 清掉残留豁免，避免它把这条闲聊的回复误放行。
                self._control_reply_until.pop(chat.chat_id or chat.primary_key, None)
        except Exception:
            self._logger().exception("发言频率控制：入站判定异常，保持默认行为")
        # 一律放行，让命令链与其他插件正常工作。
        return {"action": "continue"}

    # -- Hook：规划器兜底 ---------------------------------------------------

    @HookHandler(
        "maisaka.planner.before_request",
        name="talk_rate_planner",
        description="兜底：本轮不该回复时把 reply 工具摘掉，确保不会真的发出回复",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=2000,
    )
    async def gate_planner(self, **kwargs: Any) -> dict[str, Any]:
        try:
            self._load_state()
            if not self._enabled():
                return {"action": "continue"}

            # 规划器 Hook 没有 message/is_at 字段，只能按 session_id 取回入站判定。
            session_id = _as_str(kwargs.get("session_id"))
            recalled = self._recall_inbound(session_id)

            if recalled is None:
                # 没有入站记忆（例如插件热重载后第一批、或消息来自主动任务）：
                # 退回按会话身份判断，但**不拿 @ 说事**，因为这里根本看不到消息内容。
                rule = self.resolve_rule(ChatRef.from_payload(kwargs))
                if rule.rate >= 1.0:
                    return {"action": "continue"}
                reason = f"{rule.source}：频率={_fmt_rate(rule.rate)}，无入站记忆，默认不主动回复"
            else:
                allowed, reason = recalled
                if allowed:
                    return {"action": "continue"}

            changed = False
            for key in ("tool_definitions", "tools"):
                value = kwargs.get(key)
                if isinstance(value, list) and value:
                    kept = [item for item in value if not _is_reply_tool(item)]
                    if len(kept) != len(value):
                        kwargs[key] = kept
                        changed = True
            if not changed:
                return {"action": "continue"}

            self._stats["silenced"] += 1
            self._last_reason = reason
            self._logger().debug("发言频率控制：本轮摘掉 reply 工具（%s）", reason)
            return {"action": "continue", "modified_kwargs": kwargs}
        except Exception:
            self._logger().exception("发言频率控制：规划器兜底异常，保持原样")
            return {"action": "continue"}

    # -- Hook：发送闸门（真正生效的位置） -----------------------------------

    @HookHandler(
        "send_service.before_send",
        name="talk_rate_outbound",
        description="本轮不该回复时取消这条出站消息；这是唯一能真正拦住主动回复的位置",
        mode=HookMode.BLOCKING,
        order=HookOrder.EARLY,
        error_policy=ErrorPolicy.SKIP,
        timeout_ms=2000,
    )
    async def gate_outbound(self, **kwargs: Any) -> dict[str, Any]:
        """在真正发出前裁决。

        为什么必须在这里拦：``reply`` 在 Maisaka 里是 **deferred 工具**，
        规划器顶层工具表里本来就没有它（要靠 ``tool_search`` 才检索出来），
        所以在 ``planner.before_request`` 里摘 ``reply`` 等于什么都没做。

        出站消息**不带** ``is_at``，因此只看入站阶段记下的判定：
        「这一轮被 @ 了 / 规则允许」才放行，否则取消。
        """

        try:
            self._load_state()
            if not self._enabled():
                return {"action": "continue"}
            if not self._outbound_guard_enabled():
                return {"action": "continue"}

            # 只处理出站消息；其它调用（无 message）直接放行。
            if not isinstance(kwargs.get("message"), dict):
                return {"action": "continue"}

            stream_id = _as_str(kwargs.get("stream_id")) or _as_str(
                (kwargs.get("message") or {}).get("session_id")
            )
            chat = ChatRef.from_payload(kwargs)

            # 命令结果（含本插件自己的）必须豁免，否则命令永远收不到回复。
            if self._is_control_reply(stream_id):
                self._logger().debug("发言频率控制：放行命令回复（%s）", stream_id)
                return {"action": "continue"}

            # 非文本出站（图片、表情、转发、语音）几乎都来自别的插件的命令结果，
            # 不是麦麦在「聊天」。一律放行，避免把别人的功能一起关掉。
            if _has_media_segment(kwargs.get("message")):
                self._logger().debug("发言频率控制：非文本出站放行（%s）", stream_id)
                return {"action": "continue"}

            recalled = self._recall_inbound(stream_id)
            if recalled is None:
                recalled = self._recall_inbound(chat.chat_id) or self._recall_inbound(chat.group_id)

            if recalled is None:
                # 没有入站记忆（主动消息、或被 @ 之外的链路触发）：
                # 用会话规则兜底，零频率会话不主动开口。
                rule = self.resolve_rule(chat)
                allowed = rule.rate >= 1.0
                reason = f"{rule.source}：频率={_fmt_rate(rule.rate)}，无入站记忆"
            else:
                allowed, reason = recalled

            if allowed:
                return {"action": "continue"}

            self._stats["silenced"] += 1
            self._last_reason = reason
            # 用 warning 而不是 info：万一误拦了别的插件，用户能直接看到是哪条消息，
            # 也便于按提示往「额外命令触发词」里补词。
            self._logger().warning(
                "发言频率控制：取消一条出站消息（%s）｜会话=%s｜内容=%r。"
                "若这是其他插件的结果而非麦麦的闲聊，请把它的触发词加到「额外命令触发词」，"
                "或关闭「启用出站拦截」",
                reason,
                stream_id,
                _preview_text(kwargs.get("message")),
            )
            return {"action": "abort"}
        except Exception:
            self._logger().exception("发言频率控制：出站判定异常，放行本条消息")
            return {"action": "continue"}

    # -- 命令 ---------------------------------------------------------------

    @Command(
        "talk_rate",
        description="查看或调整某个会话的发言频率（仅操作员）",
        pattern=(
            r"^[/／]\s*(?:tr|talk[-_ ]?rate|发言频率|说话频率|频率)"
            r"(?:\s+(?P<action>\S+))?(?:\s+(?P<arg>.*))?$"
        ),
        permission="operator",
    )
    async def cmd_talk_rate(self, **kwargs: Any) -> tuple[bool, str, int]:
        self._load_state()
        chat = ChatRef.from_payload(kwargs)
        matched = kwargs.get("matched_groups") or {}
        action = _as_str(matched.get("action")).lower()
        argument = _as_str(matched.get("arg"))
        key = chat.primary_key

        def apply_override(target: str, rate: float) -> list[str]:
            """把频率写到该会话的所有已知标识上。

            命令里能拿到群号，规划器里只有 session_id；两边都要能查到，
            否则「设置成功但不生效」。
            """

            written: list[str] = []
            for candidate in (target, *self._override_keys(chat)):
                if candidate and candidate not in written:
                    self._state.overrides[candidate] = rate
                    written.append(candidate)
            return written

        try:
            success, text, level = self._dispatch_command(chat, action, argument, key)
        except Exception:
            self._logger().exception("发言频率控制：命令执行异常")
            success, text, level = False, "执行失败，请查看插件日志。", 1

        # 命令返回值本身**不会**被主程序自动发出去（`_handle_command_processing_result`
        # 只记日志），所以这里主动把结果发到群里。
        await self._reply(chat, text, level, success)
        return success, text, level

    def _dispatch_command(
        self, chat: "ChatRef", action: str, argument: str, key: str
    ) -> tuple[bool, str, int]:
        """纯逻辑：算出命令结果文本，不涉及发送。"""

        def apply_override(target: str, rate: float) -> list[str]:
            """把频率写到该会话的所有已知标识上。

            命令里能拿到群号，规划器/出站只有 session_id；两边都要能查到，
            否则「设置成功但不生效」。
            """

            written: list[str] = []
            for candidate in (target, *self._override_keys(chat)):
                if candidate and candidate not in written:
                    self._state.overrides[candidate] = rate
                    written.append(candidate)
            return written

        if action in {"", "status", "状态"}:
            return True, self._status_text(chat), 2

        if action in {"help", "?", "帮助"}:
            return True, self._help_text(), 2

        if action in {"reset", "重置", "clear"}:
            for item in list(self._state.overrides):
                if item in self._override_keys(chat):
                    self._state.overrides.pop(item, None)
            self._save_state()
            return True, f"已清除 {key or '当前会话'} 的临时频率，恢复使用配置规则。", 2

        if action in {"set", "设置", "rate", "freq"}:
            if not argument:
                return False, "用法：/tr set <频率>，例如 /tr set 0 或 /tr set 5%", 1
            target_key, value_text = key, argument
            # 允许「显式群号 + 频率」的写法：最后一个空格前是群号，后面是频率。
            explicit_group = False
            if " " in argument:
                head, _, tail = argument.rpartition(" ")
                if parse_rate(tail) is not None and head.strip():
                    target_key, value_text = head.strip(), tail.strip()
                    explicit_group = True
            if not target_key:
                return False, "无法确定要设置哪个会话，请在群里使用或写明群号。", 1
            rate = parse_rate(value_text)
            if rate is None:
                return False, f"无法识别频率 {value_text!r}，可用 0 / 0.05 / 5% / on / off。", 1
            if explicit_group:
                self._state.overrides[target_key] = rate
            else:
                apply_override(target_key, rate)
            self._save_state()
            return True, f"已把 {target_key} 的频率设为 {_fmt_rate(rate)}（立即生效）。", 2

        # 直接写数字：/tr 0、/tr 5%、/tr 0.05
        if action:
            rate = parse_rate(action)
            if rate is not None and key:
                apply_override(key, rate)
                self._save_state()
                return True, f"已把 {key} 的频率设为 {_fmt_rate(rate)}（立即生效）。", 2

        return False, f"未知子命令 {action!r}。\n{self._help_text()}", 1

    async def _reply(self, chat: "ChatRef", text: str, level: int, success: bool) -> None:
        """把命令结果主动发到 QQ。

        多路兜底：``stream_id`` → ``chat_id`` → ``group_id``，
        取第一个发送成功的。发送失败不影响命令本身的结果。
        """

        del success, level  # 目前所有分支都发送，保留参数便于以后分级
        if not text:
            return

        targets: list[str] = []
        for candidate in (chat.chat_id, chat.group_id, chat.user_id):
            if candidate and candidate not in targets:
                targets.append(candidate)
        if not targets:
            self._logger().warning("发言频率控制：没有可用的会话标识，命令结果无法发送")
            return

        for stream_id in targets:
            try:
                # 先登记豁免，避免这条命令回复被自己的出站闸门拦下。
                self._mark_control_reply(stream_id)
                sent = await self.ctx.send.text(text, stream_id)
            except Exception as exc:
                self._control_reply_until.pop(stream_id, None)
                self._logger().debug("发言频率控制：向 %s 发送命令结果失败: %s", stream_id, exc)
                continue
            if sent:
                return
            # 没发出去就别留着豁免，避免误放行下一条普通消息。
            self._control_reply_until.pop(stream_id, None)
        self._logger().warning("发言频率控制：命令结果发送失败（目标 %s）", targets)

    # -- 工具 ---------------------------------------------------------------

    @Tool(
        "talk_rate_status",
        description="查询当前聊天流的发言频率规则，以及为什么说/不说",
        parameters=[
            ToolParameterInfo(
                name="chat_id",
                param_type=ToolParamType.STRING,
                description="要查询的会话 ID，留空表示当前工具调用所在的会话",
                required=False,
            ),
        ],
    )
    async def tool_talk_rate_status(self, chat_id: str = "", **kwargs: Any) -> dict[str, Any]:
        try:
            self._load_state()
            payload = dict(kwargs)
            if chat_id:
                payload["chat_id"] = chat_id
            chat = ChatRef.from_payload(payload)
            return {"content": self._status_text(chat), "success": True}
        except Exception:
            return {"content": "查询发言频率失败，请查看插件日志。", "success": False}

    # -- 文案 ---------------------------------------------------------------

    def _status_text(self, chat: ChatRef) -> str:
        rule = self.resolve_rule(chat)
        allowed, reason = self.decide(chat)
        keywords = parse_keywords(_cfg_get(self.config.group_rules, "keywords", ""))
        keyword_line = (
            f"关键词：{len(keywords)} 个（{normalize_keyword_mode(_cfg_get(self.config.group_rules, 'keyword_match', 'contains'))}）"
            f" {', '.join(keywords[:8])}{' …' if len(keywords) > 8 else ''}"
            if keywords
            else "关键词：未配置"
        )
        lines = [
            f"会话：{chat.primary_key or '（未识别）'}"
            + ("（私聊）" if chat.is_private else "（群聊）" if chat.group_id else ""),
            f"生效规则：{rule.source}",
            f"当前频率：{_fmt_rate(rule.rate)}",
            keyword_line,
            f"本次判定：{'可以回复' if allowed else '不回复'} —— {reason}",
            f"统计：放行 {self._stats['allowed']} / 拦截 {self._stats['blocked']} / 摘工具 {self._stats['silenced']}",
        ]
        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "发言频率控制\n"
            "/tr —— 查看当前会话规则\n"
            "/tr 0 —— 当前会话：只有被 @ 或命中关键词才回复\n"
            "/tr 5% —— 当前会话：5% 概率回复（@ 与关键词必回）\n"
            "/tr set 123456789 0 —— 指定群号设为「只在被 @ 时回复」\n"
            "/tr reset —— 清除当前会话的临时设置\n"
            "/tr help —— 显示本帮助\n"
            "频率写法：0 / 0.05 / 5% / on / off\n"
            "关键词放行在插件配置 group_rules.keywords 里设置。"
        )

    # -- 生命周期 -----------------------------------------------------------

    async def on_load(self) -> None:
        self._load_state()
        logger = self._logger()
        logger.info("发言频率控制已加载：enabled=%s", self._enabled())
        if not self._enabled():
            logger.info("发言频率控制当前处于停用状态，不会影响任何回复。")

    async def on_unload(self) -> None:
        self._save_state()
        self._logger().info("发言频率控制已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        del config_data
        if scope and scope not in {"self", "plugin"}:
            return
        self._last_reason = "配置已更新"
        self._logger().info("发言频率控制配置已更新（version=%s）", version)


def _is_reply_tool(definition: Any) -> bool:
    """判断一个工具定义是不是内置的 ``reply`` 工具。"""

    if isinstance(definition, dict):
        function = definition.get("function")
        if isinstance(function, dict):
            name = _as_str(function.get("name"))
            if name:
                return name == _REPLY_TOOL_NAME
        return _as_str(definition.get("name")) == _REPLY_TOOL_NAME
    name = _as_str(getattr(definition, "name", ""))
    if name:
        return name == _REPLY_TOOL_NAME
    function = getattr(definition, "function", None)
    return _as_str(getattr(function, "name", "")) == _REPLY_TOOL_NAME


def create_plugin() -> TalkRatePlugin:
    """MaiBot 插件工厂函数。"""

    return TalkRatePlugin()
