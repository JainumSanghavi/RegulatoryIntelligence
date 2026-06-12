import httpx
import respx

from regintel.llm.base import ChatMessage
from regintel.llm.ollama_provider import OllamaProvider, _parse_json


def test_parse_json_plain():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_fences():
    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_extracts_from_prose():
    # Cloud reasoning models often wrap JSON in explanation; extract the object.
    text = 'Here is my answer:\n{"query_type": "lookup", "reasoning": "factual"}\nHope that helps!'
    assert _parse_json(text) == {"query_type": "lookup", "reasoning": "factual"}


def test_parse_json_handles_nested_and_strings_with_braces():
    text = 'noise {"findings": [{"topic": "a {b}", "gap": true}]} trailing'
    assert _parse_json(text) == {"findings": [{"topic": "a {b}", "gap": True}]}


def test_parse_json_returns_none_on_pure_prose():
    assert _parse_json("I cannot help with that.") is None


@respx.mock
def test_chat_structured_embeds_schema_in_prompt():
    captured = {}

    def _handler(request):
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

    respx.post("http://localhost:11434/api/chat").mock(side_effect=_handler)
    p = OllamaProvider(host="http://localhost:11434", default_model="m")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    out = p.chat_structured([ChatMessage("user", "go")], schema=schema)
    assert out == {"ok": True}
    # The schema instruction is appended as a final message (cloud models ignore `format`).
    assert "JSON" in captured["body"]["messages"][-1]["content"]
    assert "ok" in captured["body"]["messages"][-1]["content"]


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
