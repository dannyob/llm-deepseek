from typing import Literal, Optional

import llm
from llm.default_plugins.openai_models import Chat
from pydantic import Field

# Try to import AsyncChat, but don't fail if it's not available
try:
    from llm.default_plugins.openai_models import AsyncChat

    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False

MODELS = (
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
)


class DeepSeekOptions(Chat.Options):
    temperature: Optional[float] = Field(
        description="Sampling temperature to use, between 0 and 2.",
        ge=0,
        le=2,
        default=None,
    )
    thinking: Optional[Literal["enabled", "disabled"]] = Field(
        description="DeepSeek thinking mode type, sent as thinking.type.",
        default=None,
    )
    reasoning_effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = Field(
        description="Constrains effort on reasoning for supported DeepSeek models.",
        default=None,
    )


class DeepSeekMixin:
    Options = DeepSeekOptions

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        thinking_type = kwargs.pop("thinking", None)
        if thinking_type is not None:
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": thinking_type}
        return kwargs


class DeepSeekChat(DeepSeekMixin, Chat):
    needs_key = "deepseek"
    key_env_var = "LLM_DEEPSEEK_KEY"

    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            model_id=model_name,
            supports_tools=True,
            api_base="https://api.deepseek.com",
        )

    def __str__(self):
        return "DeepSeek: {}".format(self.model_id)


# Only define AsyncChat class if async support is available
if HAS_ASYNC:

    class DeepSeekAsyncChat(DeepSeekMixin, AsyncChat):
        needs_key = "deepseek"
        key_env_var = "LLM_DEEPSEEK_KEY"

        def __init__(self, model_name):
            super().__init__(
                model_name=model_name,
                model_id=model_name,
                supports_tools=True,
                api_base="https://api.deepseek.com",
            )

        def __str__(self):
            return "DeepSeek: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "deepseek", DeepSeekChat.key_env_var)
    if not key:
        return
    for model_id in MODELS:
        if HAS_ASYNC:
            register(
                DeepSeekChat(model_id),
                DeepSeekAsyncChat(model_id),
            )
        else:
            register(DeepSeekChat(model_id))
