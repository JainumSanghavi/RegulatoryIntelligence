import httpx
import respx

from regintel.llm.base import ChatMessage
from regintel.llm.ollama_provider import OllamaProvider


@respx.mock
def test_chat_returns_content():
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "hello world"}})
    )
    p = OllamaProvider(host="http://localhost:11434", default_model="gpt-oss:120b-cloud")
    out = p.chat([ChatMessage("user", "hi")])
    assert out == "hello world"
    assert route.called
    sent = route.calls.last.request
    assert b'"stream": false' in sent.content or b'"stream":false' in sent.content


@respx.mock
def test_chat_structured_parses_json_content():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": '{"score": 5}'}})
    )
    p = OllamaProvider(host="http://localhost:11434", default_model="m")
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
    out = p.chat_structured([ChatMessage("user", "rate")], schema=schema)
    assert out == {"score": 5}


@respx.mock
def test_chat_raises_llmerror_on_500():
    from regintel.llm.base import LLMError
    respx.post("http://localhost:11434/api/chat").mock(return_value=httpx.Response(500))
    p = OllamaProvider(host="http://localhost:11434", default_model="m", max_retries=1)
    import pytest
    with pytest.raises(LLMError):
        p.chat([ChatMessage("user", "hi")])
