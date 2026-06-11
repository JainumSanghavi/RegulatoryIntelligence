from regintel.llm.base import ChatMessage, LLMError, LLMProvider


def test_chat_message():
    m = ChatMessage(role="user", content="hi")
    assert m.as_dict() == {"role": "user", "content": "hi"}


def test_provider_is_protocol():
    class Stub:
        def chat(self, messages, *, model=None, temperature=0.0, **kw): return "ok"
        def chat_structured(self, messages, *, schema, model=None, temperature=0.0, **kw): return {}

    p: LLMProvider = Stub()
    assert p.chat([]) == "ok"


def test_llm_error():
    assert issubclass(LLMError, Exception)
