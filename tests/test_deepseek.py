import llm
import pytest
from llm.plugins import pm

import llm_deepseek


def test_plugin_is_installed():
    try:
        from llm.plugins import load_plugins
        load_plugins()
    except ImportError:
        pass

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_deepseek" in names


def test_models_include_deepseek_v4_models():
    assert "deepseek-v4-flash" in llm_deepseek.MODELS
    assert "deepseek-v4-pro" in llm_deepseek.MODELS


def test_register_models_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm_deepseek.llm, "get_key", lambda *args: None)
    registered_model_ids = []

    def register(*models):
        registered_model_ids.append(models[0].model_id)

    llm_deepseek.register_models(register)

    assert registered_model_ids == []


def test_register_models_registers_all_models_with_api_key(monkeypatch):
    monkeypatch.setattr(llm_deepseek.llm, "get_key", lambda *args: "fake-key")
    registered_model_ids = []

    def register(*models):
        registered_model_ids.append(models[0].model_id)

    llm_deepseek.register_models(register)

    assert registered_model_ids == list(llm_deepseek.MODELS)


def test_registered_models_support_tools(monkeypatch):
    monkeypatch.setattr(llm_deepseek.llm, "get_key", lambda *args: "fake-key")
    registered = []

    llm_deepseek.register_models(lambda *models: registered.extend(models))

    assert registered
    assert all(m.supports_tools for m in registered)


def test_deepseek_options_build_request_kwargs():
    model = llm_deepseek.DeepSeekChat("deepseek-v4-pro")
    prompt = llm.Prompt(
        "hello",
        model,
        options=model.Options(
            temperature=0.4,
            thinking="enabled",
            reasoning_effort="xhigh",
        ),
    )

    assert model.build_kwargs(prompt, stream=False) == {
        "temperature": 0.4,
        "reasoning_effort": "xhigh",
        "extra_body": {"thinking": {"type": "enabled"}},
    }


def test_deepseek_options_preserve_openai_compatible_options():
    model = llm_deepseek.DeepSeekChat("deepseek-chat")
    prompt = llm.Prompt(
        "hello",
        model,
        options=model.Options(
            max_tokens=100,
            top_p=0.8,
            stop="END",
            json_object=True,
        ),
    )

    assert model.build_kwargs(prompt, stream=False) == {
        "max_tokens": 100,
        "top_p": 0.8,
        "stop": "END",
        "response_format": {"type": "json_object"},
    }


def test_deepseek_options_reject_invalid_thinking_type():
    model = llm_deepseek.DeepSeekChat("deepseek-v4-flash")

    with pytest.raises(ValueError):
        model.Options(thinking="maybe")


def test_build_messages_consumes_prompt_messages_not_conversation():
    """0.32 contract: build_messages reads prompt.messages, the complete
    input chain, and must not double-emit from conversation.responses."""
    model = llm_deepseek.DeepSeekChat("deepseek-chat")
    prompt = llm.Prompt(
        "hi",
        model,
        messages=[
            llm.system("be brief"),
            llm.user("hi"),
        ],
    )
    assert model.build_messages(prompt, None) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_build_messages_replays_reasoning_content():
    """DeepSeek requires reasoning_content to be passed back when the
    assistant made a tool call — a ReasoningPart on an assistant message
    must be emitted as reasoning_content."""
    model = llm_deepseek.DeepSeekChat("deepseek-reasoner")
    from llm.parts import ReasoningPart, ToolCallPart

    prompt = llm.Prompt(
        None,
        model,
        messages=[
            llm.user("weather?"),
            llm.assistant(
                ReasoningPart(text="Need to look this up"),
                ToolCallPart(
                    name="get_weather",
                    arguments={"city": "Paris"},
                    tool_call_id="call_1",
                ),
            ),
            llm.tool_message(
                llm.parts.ToolResultPart(
                    tool_call_id="call_1",
                    name="get_weather",
                    output="sunny",
                )
            ),
        ],
    )
    sent = model.build_messages(prompt, None)
    assistant = [m for m in sent if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["reasoning_content"] == "Need to look this up"
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_build_messages_skips_redacted_reasoning():
    """Redacted ReasoningParts (empty chunk) have nothing to echo back."""
    model = llm_deepseek.DeepSeekChat("deepseek-reasoner")
    from llm.parts import ReasoningPart, ToolCallPart

    prompt = llm.Prompt(
        None,
        model,
        messages=[
            llm.user("q"),
            llm.assistant(
                ReasoningPart(text="", redacted=True),
                ToolCallPart(
                    name="get_weather",
                    arguments={"city": "Paris"},
                    tool_call_id="call_1",
                ),
            ),
        ],
    )
    sent = model.build_messages(prompt, None)
    assistant = [m for m in sent if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant[0]


def test_execute_yields_reasoning_stream_events():
    """Streamed reasoning_content must surface as StreamEvent reasoning
    events so the framework assembles ReasoningPart objects."""
    model = llm_deepseek.DeepSeekChat("deepseek-reasoner")

    class FakeDelta:
        role = "assistant"
        content = None
        tool_calls = None

        def __init__(self, reasoning_content=None, content=None):
            self.reasoning_content = reasoning_content
            if content is not None:
                self.content = content

    class FakeChoice:
        logprobs = None
        finish_reason = None

        def __init__(self, delta):
            self.delta = delta

    class FakeChunk:
        usage = None

        def __init__(self, delta):
            self.choices = [FakeChoice(delta)]

    chunks = iter(
        [
            FakeChunk(FakeDelta(reasoning_content="Let me think")),
            FakeChunk(FakeDelta(reasoning_content=" step by step.")),
            FakeChunk(FakeDelta(content="42")),
        ]
    )

    def fake_create(**kwargs):
        return chunks

    completions = type("Completions", (), {"create": staticmethod(fake_create)})()
    chat = type("Chat", (), {"completions": completions})()
    model.get_client = lambda key: type("Client", (), {"chat": chat})()

    prompt = llm.Prompt("q", model)
    response = llm.Response(prompt, model, stream=True)
    events = list(model.execute(prompt, True, response, None, "k"))
    reasoning = [e for e in events if e.type == "reasoning"]
    assert reasoning
    assert "".join(e.chunk for e in reasoning) == "Let me think step by step."
