"""离线单元测试：用桩 SDK 导入 plugin.py，不需要真实 MaiBot 环境。

运行方式（插件目录下）：``python tests/test_talk_rate.py``
也可以用 pytest 收集：``pytest tests -q``
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_sdk_stub() -> None:
    """在没有 maibot_sdk 的环境里造一个最小可用的桩模块。"""

    if "maibot_sdk" in sys.modules:
        return

    sdk = types.ModuleType("maibot_sdk")
    types_mod = types.ModuleType("maibot_sdk.types")

    class MaiBotPlugin:
        config_reload_subscriptions = ()

        @property
        def config(self):
            config = getattr(self, "_fake_config", None)
            if config is None:
                raise RuntimeError("插件上下文尚未初始化")
            return config

        def get_plugin_config_data(self):
            return {}

    class _FieldInfo:
        """模拟 pydantic 的 FieldInfo：保留描述与额外 schema，供可读性测试检查。"""

        def __init__(self, default, default_factory, description, extra):
            self.default = default_factory() if default_factory is not None else default
            self.description = description
            self.json_schema_extra = extra

    def Field(default=None, default_factory=None, description="", **kwargs):
        extra = kwargs.get("json_schema_extra")
        return _FieldInfo(default, default_factory, description, extra)

    def _identity_decorator(*args, **kwargs):
        del args, kwargs

        def wrapper(func):
            return func

        return wrapper

    class _PluginConfigMeta(type):
        """收集类属性上的 Field 声明，模拟 pydantic 的 ``model_fields``。"""

        def __new__(mcls, name, bases, namespace, **kw):
            cls = super().__new__(mcls, name, bases, namespace, **kw)
            collected = {}
            for base in bases:
                collected.update(getattr(base, "model_fields", {}) or {})
            for key, value in namespace.items():
                if isinstance(value, _FieldInfo):
                    collected[key] = value
                    # 类属性交给配置桩使用，保持与旧行为一致（直接是默认值）
                    setattr(cls, key, value.default)
            cls.model_fields = collected
            return cls

    class PluginConfigBase(metaclass=_PluginConfigMeta):
        model_fields: dict = {}

    class _Value:
        def __init__(self, name: str) -> None:
            self.name = name

        def __repr__(self) -> str:
            return f"<{self.name}>"

    class HookMode:
        BLOCKING = _Value("blocking")
        OBSERVE = _Value("observe")

    class HookOrder:
        EARLY = _Value("early")
        NORMAL = _Value("normal")
        LATE = _Value("late")

    class ErrorPolicy:
        ABORT = _Value("abort")
        SKIP = _Value("skip")
        LOG = _Value("log")

    class ToolParamType:
        STRING = _Value("string")
        INTEGER = _Value("integer")
        NUMBER = _Value("number")
        FLOAT = _Value("float")
        BOOLEAN = _Value("boolean")
        ARRAY = _Value("array")
        OBJECT = _Value("object")

    class ToolParameterInfo:
        def __init__(
            self,
            name: str,
            param_type=None,
            description: str = "",
            required: bool = True,
            default=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.name = name
            self.param_type = param_type
            self.description = description
            self.required = required
            self.default = default

    sdk.MaiBotPlugin = MaiBotPlugin
    sdk.PluginConfigBase = PluginConfigBase
    sdk.Field = Field
    sdk.Command = _identity_decorator
    sdk.Tool = _identity_decorator
    sdk.HookHandler = _identity_decorator
    sdk.types = types_mod

    types_mod.HookMode = HookMode
    types_mod.HookOrder = HookOrder
    types_mod.ErrorPolicy = ErrorPolicy
    types_mod.ToolParamType = ToolParamType
    types_mod.ToolParameterInfo = ToolParameterInfo

    sys.modules["maibot_sdk"] = sdk
    sys.modules["maibot_sdk.types"] = types_mod


_install_sdk_stub()

import plugin as tr  # noqa: E402  （必须在桩安装之后导入）


# ---------------------------------------------------------------------------
# 测试脚手架
# ---------------------------------------------------------------------------


class _Section:
    """可随意设置属性的配置小节桩。"""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _Config:
    """插件配置桩：三个小节，缺省值对齐 plugin.py 的默认声明。"""

    def __init__(self, **overrides) -> None:
        self.plugin = _Section(enabled=True, config_version="1.0.0")
        self.default_rule = _Section(group_rate="default", private_rate="default", blocked_whitelist="")
        self.group_rules = _Section(
            rules="", at_bypass=True, keywords="", keyword_match="contains", keyword_case_sensitive=False
        )
        for name, values in overrides.items():
            section = getattr(self, name)
            for key, value in values.items():
                setattr(section, key, value)


class _Paths:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class _Send:
    """记录插件主动发出的消息，并可模拟发送失败。"""

    def __init__(self, fail_targets: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_targets = fail_targets

    async def text(self, text: str, stream_id: str = "") -> bool:
        self.calls.append((stream_id, text))
        return stream_id not in self.fail_targets


class _Ctx:
    def __init__(self, data_dir: Path) -> None:
        self.paths = _Paths(data_dir)
        self.logger = _QuietLogger()
        self.send = _Send()


class _QuietLogger:
    def debug(self, *args, **kwargs):
        pass

    info = debug
    warning = debug
    exception = debug
    error = debug


def _make_plugin(data_dir: Path, **overrides) -> "tr.TalkRatePlugin":
    plugin = tr.TalkRatePlugin()
    plugin._fake_config = _Config(**overrides)
    plugin.ctx = _Ctx(data_dir)
    return plugin


def _message_payload(
    *,
    group_id: str = "",
    user_id: str = "10001",
    chat_id: str = "",
    text: str = "",
    is_at: bool = False,
    is_mentioned: bool = False,
    platform: str = "qq",
) -> dict:
    """构造一个接近真实 Hook 参数的载荷。"""

    message_info: dict = {"user_info": {"user_id": user_id, "user_nickname": "某人"}, "additional_config": {}}
    if group_id:
        message_info["group_info"] = {"group_id": group_id, "group_name": f"群{group_id}"}
    return {
        "message": {
            "message_id": "m1",
            "platform": platform,
            "session_id": chat_id or f"session-{group_id or user_id}",
            "is_at": is_at,
            "is_mentioned": is_mentioned,
            "processed_plain_text": text,
            "message_info": message_info,
        }
    }


class _TempDir:
    """在插件目录下建临时目录，避免 tempfile 在受限环境下写不进去。"""

    def __enter__(self) -> Path:
        base = ROOT / "tests" / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / f"case-{os.getpid()}-{time.perf_counter_ns()}"
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 频率解析
# ---------------------------------------------------------------------------


def test_parse_rate_basics() -> None:
    assert tr.parse_rate("0") == 0.0
    assert tr.parse_rate(0) == 0.0
    assert tr.parse_rate("0.05") == 0.05
    assert tr.parse_rate("5%") == 0.05
    assert tr.parse_rate("5％") == 0.05
    assert tr.parse_rate("100%") == 1.0
    assert tr.parse_rate("1") == 1.0


def test_parse_rate_words() -> None:
    assert tr.parse_rate("off") == 0.0
    assert tr.parse_rate("关") == 0.0
    assert tr.parse_rate("关闭") == 0.0
    assert tr.parse_rate("on") == 1.0
    assert tr.parse_rate("开") == 1.0


def test_parse_rate_invalid() -> None:
    assert tr.parse_rate("default") is None
    assert tr.parse_rate("") is None
    assert tr.parse_rate("abc") is None


def test_parse_rate_clamped() -> None:
    assert tr.parse_rate("150%") == 1.0
    assert tr.parse_rate("-3") == 0.0


# ---------------------------------------------------------------------------
# 规则解析
# ---------------------------------------------------------------------------


def test_parse_group_rules() -> None:
    rules = tr.parse_group_rules(
        "\n".join(
            [
                "# 注释行",
                "123=0",
                "456 = 0.05",
                "// 另一种注释",
                "789=off  # 行尾注释",
                "abc",  # 无效行，忽略
                "999=abc",  # 频率非法，忽略
            ]
        )
    )
    assert rules == [("123", 0.0), ("456", 0.05), ("789", 0.0)]


def test_parse_group_rules_semicolons() -> None:
    assert tr.parse_group_rules("1=0;2=0.5；3=on") == [("1", 0.0), ("2", 0.5), ("3", 1.0)]


# ---------------------------------------------------------------------------
# 核心行为：只在被 @ 时回复
# ---------------------------------------------------------------------------


def test_zero_rate_silences_plain_chat() -> None:
    """用户的核心诉求：频率 0 时，普通聊天内容不能触发回复。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123456=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123456", text="今天天气不错"))
        allowed, reason = plugin.decide(chat)
        assert not allowed, reason


def test_zero_rate_allows_at() -> None:
    """同一个群，被 @ 时必须回复。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123456=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123456", text="@麦麦 在吗", is_at=True))
        allowed, reason = plugin.decide(chat)
        assert allowed, reason


def test_zero_rate_allows_mention_name() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123456=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123456", text="麦麦你好", is_mentioned=True))
        assert plugin.decide(chat)[0]


def test_at_bypass_can_be_disabled() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(
            data_dir,
            group_rules={"rules": "123456=0", "at_bypass": False},
        )
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123456", text="@麦麦", is_at=True))
        assert not plugin.decide(chat)[0]


def test_unlisted_group_uses_default() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123456=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="999999", text="随便聊聊"))
        allowed, reason = plugin.decide(chat)
        assert allowed, reason
        assert "默认" in reason


def test_default_zero_applies_everywhere() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, default_rule={"group_rate": "0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="424242", text="随便聊聊"))
        assert not plugin.decide(chat)[0]
        at_chat = tr.ChatRef.from_payload(_message_payload(group_id="424242", text="@麦麦", is_at=True))
        assert plugin.decide(at_chat)[0]


def test_wildcard_rule() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="777", text="闲聊"))
        assert not plugin.decide(chat)[0]


def test_whitelist_overrides_zero() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(
            data_dir,
            default_rule={"group_rate": "0", "blocked_whitelist": "888, 999"},
        )
        assert plugin.decide(tr.ChatRef.from_payload(_message_payload(group_id="888", text="闲聊")))[0]
        assert not plugin.decide(tr.ChatRef.from_payload(_message_payload(group_id="777", text="闲聊")))[0]


def test_partial_rate_allows_at_always() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0.01"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="@麦麦", is_at=True))
        assert plugin.decide(chat)[0]


def test_disabled_plugin_always_allows() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, plugin={"enabled": False}, group_rules={"rules": "*=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="闲聊"))
        allowed, reason = plugin.decide(chat)
        assert allowed and "停用" in reason


def test_control_command_always_passes() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0"})
        for text in ("/tr", "/tr 0", "/TR status", "／发言频率 help"):
            chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text=text))
            allowed, reason = plugin.decide(chat)
            assert allowed, (text, reason)


def test_private_chat_default_passes() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0"})
        payload = _message_payload(user_id="555", text="私聊消息")
        chat = tr.ChatRef.from_payload(payload)
        assert chat.is_private
        allowed, reason = plugin.decide(chat)
        assert allowed, reason


# ---------------------------------------------------------------------------
# Hook 行为
# ---------------------------------------------------------------------------


def test_inbound_hook_never_aborts_when_silenced() -> None:
    """入站阶段必须放行：否则会掐掉其他插件的命令（随机图片等）。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        result = _run(plugin.gate_inbound(**_message_payload(group_id="123", text="闲聊")))
        assert result == {"action": "continue"}


def test_inbound_hook_continues_on_at() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        result = _run(plugin.gate_inbound(**_message_payload(group_id="123", text="@麦麦", is_at=True)))
        assert result == {"action": "continue"}


def test_inbound_hook_fails_open_on_bad_payload() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0"})
        result = _run(plugin.gate_inbound(message="不是字典"))
        assert result == {"action": "continue"}


def test_inbound_records_memory_for_planner() -> None:
    """入站判定要被记下来，规划器才知道这轮该不该说话。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        payload = _message_payload(group_id="123", text="闲聊")
        session_id = payload["message"]["session_id"]
        _run(plugin.gate_inbound(**payload))
        recalled = plugin._recall_inbound(session_id)
        assert recalled is not None
        assert recalled[0] is False


def test_planner_silences_plain_chat_via_memory() -> None:
    """入站放行 + 规划器按记忆摘工具 = 普通聊天不回复，命令仍能执行。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="闲聊")
        _run(plugin.gate_inbound(**inbound))

        session_id = inbound["message"]["session_id"]
        planner_payload = {
            "session_id": session_id,
            "tool_definitions": [
                {"type": "function", "function": {"name": "reply"}},
                {"type": "function", "function": {"name": "search"}},
            ],
        }
        result = _run(plugin.gate_planner(**planner_payload))
        kept = result["modified_kwargs"]["tool_definitions"]
        assert [item["function"]["name"] for item in kept] == ["search"]


def test_planner_keeps_reply_when_at_via_memory() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="@麦麦", is_at=True)
        _run(plugin.gate_inbound(**inbound))

        planner_payload = {
            "session_id": inbound["message"]["session_id"],
            "tool_definitions": [{"type": "function", "function": {"name": "reply"}}],
        }
        result = _run(plugin.gate_planner(**planner_payload))
        assert "modified_kwargs" not in result


def test_planner_hook_removes_reply_tool() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="闲聊")
        _run(plugin.gate_inbound(**inbound))

        payload = {
            "session_id": inbound["message"]["session_id"],
            "tool_definitions": [
                {"type": "function", "function": {"name": "reply"}},
                {"type": "function", "function": {"name": "search"}},
            ],
        }
        result = _run(plugin.gate_planner(**payload))
        assert result["action"] == "continue"
        kept = result["modified_kwargs"]["tool_definitions"]
        assert [item["function"]["name"] for item in kept] == ["search"]


def test_planner_hook_keeps_reply_when_at() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="@麦麦", is_at=True)
        _run(plugin.gate_inbound(**inbound))

        payload = {
            "session_id": inbound["message"]["session_id"],
            "tool_definitions": [{"type": "function", "function": {"name": "reply"}}],
        }
        result = _run(plugin.gate_planner(**payload))
        assert "modified_kwargs" not in result


def test_planner_without_memory_fails_closed_for_zero_rate() -> None:
    """没有入站记忆时（热重载后的第一批）零频率会话不应主动开口。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        payload = {
            "session_id": "123",
            "tool_definitions": [{"type": "function", "function": {"name": "reply"}}],
        }
        result = _run(plugin.gate_planner(**payload))
        assert result.get("modified_kwargs"), "零频率会话在无记忆时也应摘到 reply"
        assert not result["modified_kwargs"]["tool_definitions"]


def test_is_reply_tool_variants() -> None:
    assert tr._is_reply_tool({"type": "function", "function": {"name": "reply"}})
    assert tr._is_reply_tool({"name": "reply"})
    assert not tr._is_reply_tool({"type": "function", "function": {"name": "search"}})
    assert not tr._is_reply_tool("reply")


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


def _command_kwargs(group_id: str = "123", action: str = "", arg: str = "", text: str = "/tr") -> dict:
    payload = _message_payload(group_id=group_id, text=text)
    payload["matched_groups"] = {"action": action, "arg": arg}
    return payload


def test_command_status() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        ok, message, level = _run(plugin.cmd_talk_rate(**_command_kwargs()))
        assert ok and level == 2
        assert "只在被 @" in message or "0" in message


def test_override_survives_session_id_lookup() -> None:
    """核心回归：/tr 0 之后，规划器用 session_id 也要能查到这条设置。

    真实环境里命令拿得到群号，规划器只有 session_id（主程序算的哈希），
    两者必须互通，否则就会出现「设置成功但照样回复」。
    """

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)

        # 命令链路：能同时看到群号和 session_id
        command_payload = _message_payload(
            group_id="327222747",
            chat_id="c58628a91bab2b87117e274bd05f05fa",
            text="/tr 0",
        )
        command_payload["matched_groups"] = {"action": "0", "arg": ""}
        ok, _, _ = _run(plugin.cmd_talk_rate(**command_payload))
        assert ok

        # 规划器链路：只剩 session_id
        planner_payload = {
            "session_id": "c58628a91bab2b87117e274bd05f05fa",
            "tool_definitions": [{"type": "function", "function": {"name": "reply"}}],
        }
        rule = plugin.resolve_rule(tr.ChatRef.from_payload(planner_payload))
        assert rule.rate == 0.0, f"规划器查不到 override，实际={rule}"

    # 群号单独作为键时也要能命中
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        plugin._state.overrides["327222747"] = 0.0
        rule = plugin.resolve_rule(
            tr.ChatRef.from_payload({"session_id": "whatever-hash", "message": {"group_id": "327222747"}})
        )
        assert rule.rate == 0.0


def test_override_reset_clears_all_aliases() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        payload = _message_payload(group_id="123", chat_id="hash-123")
        payload["matched_groups"] = {"action": "0", "arg": ""}
        _run(plugin.cmd_talk_rate(**payload))
        assert len(plugin._state.overrides) >= 2  # 群号 + session_id 都写了

        reset_payload = _message_payload(group_id="123", chat_id="hash-123")
        reset_payload["matched_groups"] = {"action": "reset", "arg": ""}
        _run(plugin.cmd_talk_rate(**reset_payload))
        assert not plugin._state.overrides


def test_group_member_id_not_treated_as_chat_key() -> None:
    """群管理员的 QQ 号不是会话标识，写进 override 会误伤他的私聊。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        payload = _message_payload(group_id="123", chat_id="hash-123", user_id="1242973939")
        payload["matched_groups"] = {"action": "0", "arg": ""}
        _run(plugin.cmd_talk_rate(**payload))

        assert "1242973939" not in plugin._state.overrides, plugin._state.overrides
        # 但该群的设置仍然生效
        assert plugin._state.overrides.get("123") == 0.0


def test_private_chat_still_uses_user_id() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        payload = _message_payload(user_id="555", text="/tr 0")
        payload["matched_groups"] = {"action": "0", "arg": ""}
        _run(plugin.cmd_talk_rate(**payload))
        assert plugin._state.overrides.get("555") == 0.0


# ---------------------------------------------------------------------------
# 出站闸门（真正拦住回复的位置）
# ---------------------------------------------------------------------------


def test_outbound_aborts_plain_chat_after_zero_rate() -> None:
    """核心：reply 是 deferred 工具，摘工具没用，必须靠出站拦截。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="闲聊")
        _run(plugin.gate_inbound(**inbound))

        session_id = inbound["message"]["session_id"]
        result = _run(
            plugin.gate_outbound(
                stream_id=session_id,
                message={"session_id": session_id, "processed_plain_text": "回复内容"},
            )
        )
        assert result == {"action": "abort"}


def test_outbound_allows_after_at() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        inbound = _message_payload(group_id="123", text="@麦麦", is_at=True)
        _run(plugin.gate_inbound(**inbound))

        session_id = inbound["message"]["session_id"]
        result = _run(plugin.gate_outbound(stream_id=session_id, message={"session_id": session_id}))
        assert result == {"action": "continue"}


def test_outbound_ignores_non_message_calls() -> None:
    """没有 message 的调用不是发送请求，不能拦。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0"})
        assert _run(plugin.gate_outbound(stream_id="x", typing=True)) == {"action": "continue"}


def test_outbound_fails_open_without_memory_in_normal_group() -> None:
    """没有入站记忆且会话频率正常时，不应误拦。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "999=0"})
        result = _run(plugin.gate_outbound(stream_id="other-session", message={"session_id": "other-session"}))
        assert result == {"action": "continue"}


# ---------------------------------------------------------------------------
# 不得误伤其他插件
# ---------------------------------------------------------------------------


def test_looks_like_command_detection() -> None:
    """命令识别必须覆盖「不带斜杠」的第三方触发词。"""

    for text in ("/tr 0", "/clear", "随机图片", "来张图", "涩图", "setu 3", "点歌 晴天"):
        assert tr._looks_like_command(text), text
    for text in ("今天天气不错", "麦麦你好", ""):
        assert not tr._looks_like_command(text), text


def test_third_party_command_output_is_released() -> None:
    """核心回归：/tr 0 之后，随机图片等命令的**输出**不能被拦。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})

        # 用户发「随机图片」→ 入站应识别为命令并登记放行
        inbound = _message_payload(group_id="123", text="随机图片")
        _run(plugin.gate_inbound(**inbound))
        session_id = inbound["message"]["session_id"]

        # 该命令的文本输出必须放行
        text_out = _run(
            plugin.gate_outbound(
                stream_id=session_id,
                message={"session_id": session_id, "raw_message": [{"type": "text", "data": "正在获取…"}]},
            )
        )
        assert text_out == {"action": "continue"}, "第三方命令的文本输出被误拦"


def test_media_output_always_passes() -> None:
    """图片/表情/转发类出站一律放行，它们几乎都来自别的插件。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        # 入站普通聊天，判定为不回复
        inbound = _message_payload(group_id="123", text="闲聊")
        _run(plugin.gate_inbound(**inbound))
        session_id = inbound["message"]["session_id"]

        for seg in ("image", "emoji", "forward", "voice"):
            result = _run(
                plugin.gate_outbound(
                    stream_id=session_id,
                    message={"session_id": session_id, "raw_message": [{"type": seg, "data": "x"}]},
                )
            )
            assert result == {"action": "continue"}, f"{seg} 出站被误拦"


def test_plain_reply_still_blocked_after_command() -> None:
    """命令豁免是一次性的：豁免用掉后，普通回复仍要被拦。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        cmd = _message_payload(group_id="123", text="随机图片")
        _run(plugin.gate_inbound(**cmd))
        session_id = cmd["message"]["session_id"]

        # 第一次：命令输出放行
        first = _run(
            plugin.gate_outbound(
                stream_id=session_id,
                message={"session_id": session_id, "raw_message": [{"type": "text", "data": "ok"}]},
            )
        )
        assert first == {"action": "continue"}

        # 之后来一条普通闲聊，判定为不回复，出站应被拦
        chat = _message_payload(group_id="123", text="随便聊聊")
        _run(plugin.gate_inbound(**chat))
        second = _run(
            plugin.gate_outbound(
                stream_id=session_id,
                message={"session_id": session_id, "raw_message": [{"type": "text", "data": "闲聊回复"}]},
            )
        )
        assert second == {"action": "abort"}, "普通聊天回复不该被命令豁免放行"


# ---------------------------------------------------------------------------
# 关键词放行
# ---------------------------------------------------------------------------


def test_parse_keywords() -> None:
    parsed = tr.parse_keywords("麦麦\n你好,在吗，帮我看看\n# 注释行\n// 另一种注释\n麦麦")
    assert parsed == ["麦麦", "你好", "在吗", "帮我看看"]


def test_match_keyword_modes() -> None:
    words = ["麦麦"]
    assert tr.match_keyword("请问麦麦在吗", words, "contains", False) == "麦麦"
    assert tr.match_keyword("麦麦在吗", words, "prefix", False) == "麦麦"
    assert tr.match_keyword("请问麦麦", words, "prefix", False) is None
    assert tr.match_keyword("麦麦", words, "exact", False) == "麦麦"
    assert tr.match_keyword("麦麦在吗", words, "exact", False) is None
    assert tr.match_keyword("hello 123", ["\\d+"], "regex", False) == "\\d+"


def test_match_keyword_case_sensitivity() -> None:
    assert tr.match_keyword("HELLO world", ["hello"], "contains", False) == "hello"
    assert tr.match_keyword("HELLO world", ["hello"], "contains", True) is None


def test_bad_regex_does_not_crash() -> None:
    assert tr.match_keyword("abc", ["[invalid"], "regex", False) is None


def test_zero_rate_keyword_allows_reply() -> None:
    """核心需求：频率为 0 时，命中关键词仍要回复。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0", "keywords": "麦麦"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="麦麦帮我看看吧"))
        allowed, reason = plugin.decide(chat)
        assert allowed, reason
        assert "关键词" in reason


def test_zero_rate_no_keyword_still_silent() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0", "keywords": "麦麦"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="今天天气不错"))
        assert not plugin.decide(chat)[0]


def test_keyword_end_to_end_outbound_released() -> None:
    """关键词消息的回复必须能真正发出去（走完入站→出站全链路）。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0", "keywords": "麦麦"})
        inbound = _message_payload(group_id="123", text="麦麦在吗")
        _run(plugin.gate_inbound(**inbound))
        session_id = inbound["message"]["session_id"]

        result = _run(
            plugin.gate_outbound(
                stream_id=session_id,
                message={"session_id": session_id, "raw_message": [{"type": "text", "data": "在的"}]},
            )
        )
        assert result == {"action": "continue"}, "关键词命中的回复被出站闸门拦掉了"


def test_keyword_disabled_by_default() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="麦麦在吗"))
        assert not plugin.decide(chat)[0]


def test_keyword_beats_at_bypass_disabled() -> None:
    """即使关掉 @ 放行，关键词仍应生效。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(
            data_dir,
            group_rules={"rules": "123=0", "at_bypass": False, "keywords": "帮我看看"},
        )
        # 只 @ 不提关键词：at_bypass 关闭，不回复
        at_chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="@麦麦", is_at=True))
        assert not plugin.decide(at_chat)[0]

        # 命中关键词：回复
        kw_chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="帮我看看这个"))
        assert plugin.decide(kw_chat)[0]


def test_keyword_does_not_apply_to_at_private_chat() -> None:
    """私聊默认放行，关键词不影响其行为。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "*=0", "keywords": "麦麦"})
        chat = tr.ChatRef.from_payload(_message_payload(user_id="555", text="随便说说"))
        assert chat.is_private
        assert plugin.decide(chat)[0]


# ---------------------------------------------------------------------------
# 命令结果主动发送
# ---------------------------------------------------------------------------


def test_command_sends_result_to_chat() -> None:
    """命令结果必须真的发到 QQ：主程序不会自动发。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        _run(plugin.cmd_talk_rate(**_command_kwargs()))
        assert plugin.ctx.send.calls, "命令没有发送任何消息"
        stream_id, text = plugin.ctx.send.calls[0]
        assert "发言频率" in text or "频率" in text
        # 优先发到 session_id
        assert stream_id == _message_payload(group_id="123")["message"]["session_id"]


def test_command_set_sends_confirmation() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        _run(plugin.cmd_talk_rate(**_command_kwargs(action="0")))
        assert plugin.ctx.send.calls
        assert "立即生效" in plugin.ctx.send.calls[0][1]


def test_command_falls_back_to_group_id_when_session_fails() -> None:
    """session_id 发送失败时应回退到群号。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        session_id = _message_payload(group_id="123")["message"]["session_id"]
        plugin.ctx.send = _Send(fail_targets=(session_id,))
        _run(plugin.cmd_talk_rate(**_command_kwargs()))
        targets = [c[0] for c in plugin.ctx.send.calls]
        assert targets[0] == session_id
        assert "123" in targets


def test_command_reply_is_exempt_from_outbound_gate() -> None:
    """命令回复不能被我自己的出站闸门拦掉。"""

    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        # 入站一条普通消息，判定为「不回复」
        inbound = _message_payload(group_id="123", text="闲聊")
        _run(plugin.gate_inbound(**inbound))
        session_id = inbound["message"]["session_id"]

        # 出站本应被拦
        assert _run(plugin.gate_outbound(stream_id=session_id, message={"session_id": session_id})) == {
            "action": "abort"
        }

        # 用户发命令：入站自动登记豁免，出站应放行
        cmd = _message_payload(group_id="123", text="/tr")
        _run(plugin.gate_inbound(**cmd))
        assert _run(plugin.gate_outbound(stream_id=session_id, message={"session_id": session_id})) == {
            "action": "continue"
        }


def test_command_set_value() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=default"})
        ok, message, _ = _run(plugin.cmd_talk_rate(**_command_kwargs(action="0")))
        assert ok and "立即生效" in message
        chat = tr.ChatRef.from_payload(_message_payload(group_id="123", text="闲聊"))
        assert not plugin.decide(chat)[0]


def test_command_set_explicit_group() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        ok, message, _ = _run(plugin.cmd_talk_rate(**_command_kwargs(action="set", arg="987 0")))
        assert ok
        assert plugin._state.overrides.get("987") == 0.0


def test_command_reset_restores_config() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=default"})
        _run(plugin.cmd_talk_rate(**_command_kwargs(action="0")))
        assert "123" in plugin._state.overrides
        ok, _, _ = _run(plugin.cmd_talk_rate(**_command_kwargs(action="reset")))
        assert ok and "123" not in plugin._state.overrides


def test_command_persists_across_reload() -> None:
    with _TempDir() as data_dir:
        first = _make_plugin(data_dir)
        _run(first.cmd_talk_rate(**_command_kwargs(action="set", arg="321 0")))
        second = _make_plugin(data_dir)
        second._load_state()
        assert second._state.overrides.get("321") == 0.0


def test_command_help() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        ok, message, _ = _run(plugin.cmd_talk_rate(**_command_kwargs(action="help")))
        assert ok and "/tr" in message


def test_command_rejects_unparsable_rate() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        ok, message, level = _run(plugin.cmd_talk_rate(**_command_kwargs(action="set", arg="abc")))
        assert not ok and level == 1


# ---------------------------------------------------------------------------
# 工具与生命周期
# ---------------------------------------------------------------------------


def test_status_tool() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir, group_rules={"rules": "123=0"})
        payload = _message_payload(group_id="123", text="闲聊")
        result = _run(plugin.tool_talk_rate_status(**payload))
        assert result["success"] and "不回复" in result["content"]


def test_lifecycle_smoke() -> None:
    with _TempDir() as data_dir:
        plugin = _make_plugin(data_dir)
        _run(plugin.on_load())
        _run(plugin.on_config_update("self", {}, "1.0.0"))
        _run(plugin.on_unload())


def test_create_plugin() -> None:
    instance = tr.create_plugin()
    assert isinstance(instance, tr.TalkRatePlugin)


def test_manifest_declares_every_used_capability() -> None:
    """回归：代码里用了 ``ctx.send.text``，manifest 就必须声明 ``send.text``。

    没有声明时主程序会返回 ``E_CAPABILITY_DENIED``，表现为
    「命令结果总是发送失败」，而单测用桩替换 ctx.send 时完全看不出来。
    """

    import json
    import re

    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    declared = set(manifest.get("capabilities") or [])
    source = (ROOT / "plugin.py").read_text(encoding="utf-8")

    used = {
        f"{group}.{match.group(1)}"
        for group in ("send", "emoji", "llm")
        for match in re.finditer(rf"self\.ctx\.{group}\.(\w+)\s*\(", source)
    }
    missing = sorted(used - declared)
    assert not missing, f"manifest 缺少能力声明: {missing}（现有 {sorted(declared)}）"


def test_config_fields_have_chinese_labels_and_descriptions() -> None:
    """回归：WebUI 插件配置界面的可读性。

    界面显示的是 ``json_schema_extra["label"]``，没配就会回退成英文字段名
    （例如 ``keyword_match``），描述则来自 ``Field(description=...)``。
    新增配置字段时必须同时补两者，否则界面上一片天书。
    """

    sections = [
        ("plugin", tr.PluginSectionConfig),
        ("default_rule", tr.DefaultRuleConfig),
        ("group_rules", tr.GroupRuleConfig),
    ]

    problems: list[str] = []
    for section_name, cls in sections:
        for field_name, info in cls.model_fields.items():
            description = (info.description or "").strip()
            if not description:
                problems.append(f"{section_name}.{field_name}: 缺少 description")

            extra = info.json_schema_extra if isinstance(getattr(info, "json_schema_extra", None), dict) else {}
            label = str(extra.get("label") or "").strip()
            if not label:
                problems.append(f"{section_name}.{field_name}: 缺少 label（界面会显示英文字段名）")
            elif not any("\u4e00" <= ch <= "\u9fff" for ch in label):
                problems.append(f"{section_name}.{field_name}: label 不是中文 -> {label!r}")

    assert not problems, "配置字段可读性不足:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    failures: list[tuple[str, str]] = []
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
    if failures:
        print(f"\n{len(failures)} 个用例失败：\n")
        for name, detail in failures:
            print(f"===== {name} =====\n{detail}")
        return 1
    print(f"\n全部 {len(tests)} 个用例通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
