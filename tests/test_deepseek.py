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
