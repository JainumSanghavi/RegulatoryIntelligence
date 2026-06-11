from regintel.llm.base import ChatMessage
from regintel.llm.claude_provider import ClaudeProvider


class _FakeMessages:
    def create(self, **kw):
        class _Block:
            text = '{"ok": true}' if kw.get("_structured") else "plain text"
        class _Resp:
            content = [_Block()]
        assert "messages" in kw
        return _Resp()


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_chat_extracts_text():
    p = ClaudeProvider(api_key="x", default_model="claude-sonnet-4-6", client=_FakeClient())
    assert p.chat([ChatMessage("user", "hi")]) == "plain text"


def test_chat_separates_system():
    p = ClaudeProvider(api_key="x", default_model="m", client=_FakeClient())
    out = p.chat([ChatMessage("system", "be brief"), ChatMessage("user", "hi")])
    assert out == "plain text"
