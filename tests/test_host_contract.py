"""用真实的 MaiBot 语义做集成校验（不依赖 maibot_sdk，纯离线）。

这个脚本模拟主程序的三处真实行为，用来证明插件确实能实现
「频率 0 时只有 @ 才回复」：

1. ``chat.receive.after_process`` 返回 abort 时，消息不进入 Maisaka。
2. 进入 Maisaka 后，按原版 ``runtime._get_effective_reply_frequency`` +
   ``scheduler._evaluate`` + ``reasoning_engine._handle_silent_turn`` 的真实顺序判定：
   静默消费分支会先于 @ 强制触发命中，并且会 ``_clear_forced_turn_state()``。
3. ``maisaka.planner.before_request`` 允许改 ``tool_definitions``，但不允许 abort。

运行：``python tests/test_host_contract.py``
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用单元测试里的桩，避免两份实现漂移。
sys.path.insert(0, str(ROOT / "tests"))
import test_talk_rate as harness  # noqa: E402

tr = harness.tr


class FakeHost:
    """极简的主程序替身：只复刻与本插件相关的判定顺序。"""

    def __init__(self, plugin: "tr.TalkRatePlugin") -> None:
        self.plugin = plugin
        self.log: list[str] = []

    # -- 原版行为 ----------------------------------------------------------

    @staticmethod
    def vanilla_effective_reply_frequency(base_talk_value: float, at: bool) -> float:
        """原版 ``_get_effective_reply_frequency``：频率就是频率，@ 不影响它。"""

        del at
        return max(0.0, base_talk_value)

    @staticmethod
    def vanilla_scheduler_decision(frequency: float) -> str:
        """原版 ``scheduler._evaluate`` 的前两个分支：静默消费先于强制触发。"""

        if frequency <= 0.0:
            return "silent_consume"
        return "normal_path"

    @staticmethod
    def vanilla_silent_turn_clears_forced_turn() -> bool:
        """原版 ``_handle_silent_turn`` 第一行就是清掉 @ 强制触发状态。"""

        return True

    # -- 接入插件后 --------------------------------------------------------

    def inbound(self, payload: dict) -> str:
        result = harness._run(self.plugin.gate_inbound(**payload))
        return result.get("action", "continue")

    def planner(self, payload: dict) -> list[str]:
        """返回这一轮模型实际拿到的工具名列表。"""

        reply_result = harness._run(self.plugin.gate_planner(**payload))
        tool_definitions = payload.get("tool_definitions") or []
        if reply_result.get("modified_kwargs"):
            tool_definitions = reply_result["modified_kwargs"].get("tool_definitions") or []
        return [
            (item.get("function") or {}).get("name", "") if isinstance(item, dict) else ""
            for item in tool_definitions
        ]


def _payload(*, group_id: str, text: str, at: bool = False) -> dict:
    payload = harness._message_payload(group_id=group_id, text=text, is_at=at)
    payload["session_id"] = f"session-{group_id}"
    payload["tool_definitions"] = [
        {"type": "function", "function": {"name": "reply"}},
        {"type": "function", "function": {"name": "search"}},
    ]
    return payload


def test_vanilla_zero_rate_swallows_at() -> None:
    """先证明「原版为什么做不到」：频率 0 时 @ 也会被静默吞掉。"""

    frequency = FakeHost.vanilla_effective_reply_frequency(0.0, at=True)
    assert FakeHost.vanilla_scheduler_decision(frequency) == "silent_consume"
    assert FakeHost.vanilla_silent_turn_clears_forced_turn()


def test_plugin_blocks_plain_chat_end_to_end() -> None:
    """普通聊天：入站放行（不打断命令链），但规划器里没有 reply 工具。"""

    with harness._TempDir() as data_dir:
        plugin = harness._make_plugin(data_dir, group_rules={"rules": "123456=0"})
        host = FakeHost(plugin)

        payload = _payload(group_id="123456", text="今天天气不错")
        assert host.inbound(payload) == "continue"  # 放行，命令链照常
        assert "reply" not in host.planner(payload)  # 但这一轮说不了话


def test_plugin_allows_at_end_to_end() -> None:
    with harness._TempDir() as data_dir:
        plugin = harness._make_plugin(data_dir, group_rules={"rules": "123456=0"})
        host = FakeHost(plugin)

        payload = _payload(group_id="123456", text="@麦麦 在吗", at=True)
        assert host.inbound(payload) == "continue"
        assert "reply" in host.planner(payload)


def test_other_plugin_command_still_works() -> None:
    """核心回归：随机图片之类不带斜杠的命令，入站不能被拦掉。"""

    with harness._TempDir() as data_dir:
        plugin = harness._make_plugin(data_dir, group_rules={"rules": "123456=0"})
        host = FakeHost(plugin)

        for text in ("随机图片", "来张图", "涩图", "/tr", "/clear"):
            assert host.inbound(_payload(group_id="123456", text=text)) == "continue", text


def test_other_group_unaffected() -> None:
    with harness._TempDir() as data_dir:
        plugin = harness._make_plugin(data_dir, group_rules={"rules": "123456=0"})
        host = FakeHost(plugin)
        assert host.inbound(_payload(group_id="654321", text="闲聊")) == "continue"
        assert "reply" in host.planner(_payload(group_id="654321", text="闲聊"))


def test_hook_signatures_match_host_payloads() -> None:
    """主程序传给 Hook 的参数是 ``hook_name`` + kwargs，插件必须能吃掉多余字段。"""

    with harness._TempDir() as data_dir:
        plugin = harness._make_plugin(data_dir, group_rules={"rules": "123456=0"})
        payload = _payload(group_id="123456", text="闲聊")
        payload["hook_name"] = "chat.receive.after_process"
        payload["matched_groups"] = {}
        assert harness._run(plugin.gate_inbound(**payload)).get("action") == "continue"


def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    if failures:
        print(f"\n{failures} 个用例失败。")
        return 1
    print(f"\n全部 {len(tests)} 个契约用例通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
