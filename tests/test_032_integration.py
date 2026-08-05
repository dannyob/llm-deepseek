"""Integration tests exercising the plugin against the llm 0.32
structured-messages / streaming-events machinery, with the DeepSeek API
mocked via pytest-httpx."""
import json

import pytest
from pytest_httpx import IteratorStream

import llm
import llm.parts
import llm_deepseek

pytestmark = pytest.mark.usefixtures("mock_deepseek_key")

API_URL = "https://api.deepseek.com/chat/completions"


def _sse(delta, finish_reason=None, usage=None, tool_calls=None):
    chunk = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if tool_calls is not None:
        chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n".encode()


@pytest.fixture
def mock_deepseek_key(monkeypatch):
    monkeypatch.setenv("LLM_DEEPSEEK_KEY", "fake-key")


def test_models_registered_when_key_set(mock_deepseek_key):
    assert llm.get_model("deepseek-v4-flash").model_id == "deepseek-v4-flash"


def test_simple_prompt_streams_text(httpx_mock):
    def gen():
        yield _sse({"role": "assistant", "content": ""})
        yield _sse({"content": "Hel"})
        yield _sse({"content": "lo"})
        yield _sse({}, finish_reason="stop")
        yield b"data: [DONE]\n\n"

    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        stream=IteratorStream(gen()),
        headers={"Content-Type": "text/event-stream"},
    )
    model = llm.get_model("deepseek-chat")
    response = model.prompt("hi")
    events = list(response.stream_events())
    assert all(e.type == "text" for e in events)
    assert "".join(e.chunk for e in events) == "Hello"


def test_reasoning_content_yields_reasoning_events(httpx_mock):
    """deepseek-reasoner streams reasoning_content before content."""
    def gen():
        yield _sse({"role": "assistant", "content": ""})
        yield _sse({"reasoning_content": "Let me think"})
        yield _sse({"reasoning_content": " step by step."})
        yield _sse({"content": "The answer is"})
        yield _sse({"content": " 42."})
        yield _sse({}, finish_reason="stop")
        yield b"data: [DONE]\n\n"

    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        stream=IteratorStream(gen()),
        headers={"Content-Type": "text/event-stream"},
    )
    model = llm.get_model("deepseek-reasoner")
    response = model.prompt("What is 6*7?")
    parts = response.messages()[0].parts
    kinds = [type(p) for p in parts]
    assert llm.parts.ReasoningPart in kinds
    assert llm.parts.TextPart in kinds
    reasoning = next(p for p in parts if isinstance(p, llm.parts.ReasoningPart))
    text = next(p for p in parts if isinstance(p, llm.parts.TextPart))
    assert reasoning.text == "Let me think step by step."
    assert text.text == "The answer is 42."


def test_reasoning_replayed_for_tool_call_turn(httpx_mock):
    """After an assistant turn with a tool call, reasoning_content MUST be
    sent back — DeepSeek errors otherwise."""
    def gen():
        yield _sse({"role": "assistant", "content": ""})
        yield _sse({"reasoning_content": "Need to look this up"})
        yield _sse(
            {},
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ],
        )
        yield _sse(
            {},
            tool_calls=[
                {
                    "index": 0,
                    "function": {"arguments": '{"city":"Paris"}'},
                }
            ],
        )
        yield _sse({}, finish_reason="tool_calls")
        yield b"data: [DONE]\n\n"

    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        stream=IteratorStream(gen()),
        headers={"Content-Type": "text/event-stream"},
    )

    def get_weather(city: str) -> str:
        "Look up the weather."
        return "sunny"

    model = llm.get_model("deepseek-reasoner")
    conversation = llm.Conversation(model=model)
    response = conversation.prompt("weather?", tools=[get_weather])
    response.text()
    tcs = response.tool_calls()
    assert len(tcs) == 1

    # Continue the conversation with the tool result; the conversation
    # folds the assistant's output (reasoning + tool call) into the chain.
    results = [
        llm.ToolResult(
            name="get_weather",
            output="sunny",
            tool_call_id=tcs[0].tool_call_id,
        )
    ]
    response2 = conversation.prompt("And now?", tool_results=results)
    sent = model.build_messages(response2.prompt, conversation)
    # The assistant message that carried the tool call must also carry
    # reasoning_content.
    assistant_msgs = [m for m in sent if m["role"] == "assistant"]
    assert assistant_msgs, sent
    assert "reasoning_content" in assistant_msgs[-1]
    assert assistant_msgs[-1]["reasoning_content"] == "Need to look this up"


def test_non_streaming_reasoning_content(httpx_mock):
    """Non-streaming responses put reasoning on the message object."""
    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_content": "Let me think step by step.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 12,
                "total_tokens": 19,
            },
        },
    )
    model = llm.get_model("deepseek-reasoner")
    response = model.prompt("What is 6*7?", stream=False)
    parts = response.messages()[0].parts
    kinds = [type(p) for p in parts]
    assert llm.parts.ReasoningPart in kinds
    reasoning = next(p for p in parts if isinstance(p, llm.parts.ReasoningPart))
    assert reasoning.text == "Let me think step by step."


def test_hide_reasoning_still_produces_reasoning_parts(httpx_mock):
    """hide_reasoning hides display, not the structured reasoning output."""
    def gen():
        yield _sse({"role": "assistant", "content": ""})
        yield _sse({"reasoning_content": "thinking"})
        yield _sse({"content": "answer"})
        yield _sse({}, finish_reason="stop")
        yield b"data: [DONE]\n\n"

    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        stream=IteratorStream(gen()),
        headers={"Content-Type": "text/event-stream"},
    )
    model = llm.get_model("deepseek-v4-flash")
    response = model.prompt("q", hide_reasoning=True)
    parts = response.messages()[0].parts
    assert any(isinstance(p, llm.parts.ReasoningPart) for p in parts)


@pytest.mark.asyncio
async def test_async_reasoning_events(httpx_mock):
    def gen():
        yield _sse({"role": "assistant", "content": ""})
        yield _sse({"reasoning_content": "Let me think"})
        yield _sse({"content": "42"})
        yield _sse({}, finish_reason="stop")
        yield b"data: [DONE]\n\n"

    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        stream=IteratorStream(gen()),
        headers={"Content-Type": "text/event-stream"},
    )
    model = llm.get_async_model("deepseek-reasoner")
    response = await model.prompt("What is 6*7?")
    messages = await response.messages()
    parts = messages[0].parts
    assert any(isinstance(p, llm.parts.ReasoningPart) for p in parts)
    reasoning = next(p for p in parts if isinstance(p, llm.parts.ReasoningPart))
    assert reasoning.text == "Let me think"


@pytest.mark.asyncio
async def test_async_non_streaming_reasoning(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url=API_URL,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "42",
                        "reasoning_content": "async thinking",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 12,
                "total_tokens": 19,
            },
        },
    )
    model = llm.get_async_model("deepseek-reasoner")
    response = await model.prompt("What is 6*7?", stream=False)
    messages = await response.messages()
    parts = messages[0].parts
    reasoning = next(p for p in parts if isinstance(p, llm.parts.ReasoningPart))
    assert reasoning.text == "async thinking"
