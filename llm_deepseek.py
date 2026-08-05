import json
from typing import Iterator, Literal, Optional

import llm
from llm.default_plugins.openai_models import (
    Chat,
    _attachment,
    combine_chunks,
    redact_data,
)
from llm.parts import StreamEvent
from llm.utils import remove_dict_none_values
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

    @staticmethod
    def _delta_reasoning(chunk):
        """Extract reasoning_content from a streamed chunk, if present."""
        if chunk.choices and chunk.choices[0].delta:
            return getattr(chunk.choices[0].delta, "reasoning_content", None) or ""
        return ""

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        thinking_type = kwargs.pop("thinking", None)
        if thinking_type is not None:
            kwargs.setdefault("extra_body", {})["thinking"] = {"type": thinking_type}
        return kwargs

    def build_messages(self, prompt, conversation, image_detail=None):
        """Translate prompt.messages into DeepSeek's wire format.

        Under the 0.32 structured-messages contract, ``prompt.messages``
        is the complete input chain for this turn (prior turns' reasoning
        included), so we build exclusively from it — never from
        ``conversation.responses``.

        DeepSeek requires an assistant's ``reasoning_content`` to be passed
        back on subsequent requests whenever that assistant made a tool
        call (it returns a 400 error otherwise). Reasoning is stored by the
        framework as a ``ReasoningPart`` inside the assistant message, so
        ``_append_llm_message`` surfaces it as ``reasoning_content``.
        """
        messages = []
        if image_detail is not None:
            image_detail = image_detail.value
        current_system = None
        for msg in prompt.messages:
            current_system = self._append_llm_message(
                messages, msg, current_system, image_detail=image_detail
            )
        return messages

    def _append_llm_message(self, out, message, current_system, image_detail=None):
        """Translate one llm.Message into DeepSeek/OpenAI wire format.

        Mirrors ``_Shared._append_llm_message`` but additionally maps a
        ``ReasoningPart`` on an assistant message to ``reasoning_content``.
        """
        from llm.parts import (
            AttachmentPart,
            ReasoningPart,
            TextPart,
            ToolCallPart,
            ToolResultPart,
        )

        text_bits = []
        reasoning_bits = []
        attachment_items = []
        tool_calls = []
        tool_results = []

        for part in message.parts:
            if isinstance(part, TextPart):
                text_bits.append(part.text)
            elif isinstance(part, ReasoningPart):
                # Redacted reasoning (empty chunk) has nothing to echo back.
                if not part.redacted and part.text:
                    reasoning_bits.append(part.text)
            elif isinstance(part, AttachmentPart) and part.attachment:
                attachment_items.append(
                    _attachment(part.attachment, image_detail=image_detail)
                )
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    {
                        "type": "function",
                        "id": part.tool_call_id,
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.arguments),
                        },
                    }
                )
            elif isinstance(part, ToolResultPart):
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.output,
                    }
                )

        # Role "tool" emits one DeepSeek "tool" message per ToolResultPart.
        if message.role == "tool":
            out.extend(tool_results)
            return current_system

        # System dedup: skip if this text is already the active system prompt.
        if message.role == "system":
            text = "".join(text_bits)
            if text == current_system:
                return current_system
            current_system = text

        if attachment_items:
            content = []
            if text_bits:
                content.append({"type": "text", "text": "".join(text_bits)})
            content.extend(attachment_items)
            entry = {"role": message.role, "content": content}
        else:
            entry = {
                "role": message.role,
                "content": "".join(text_bits) if text_bits else None,
            }

        if tool_calls:
            entry["tool_calls"] = tool_calls
            # DeepSeek expects content=null when only tool_calls are present.
            if not text_bits:
                entry["content"] = None
        elif entry["content"] is None and message.role != "assistant":
            # For user/system, an empty message is pointless — drop it.
            return current_system

        if reasoning_bits:
            entry["reasoning_content"] = "".join(reasoning_bits)

        out.append(entry)
        return current_system


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

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
        key: str | None = None,
    ) -> Iterator[str | StreamEvent]:
        if prompt.system and not self.allows_system_prompt:
            raise NotImplementedError("Model does not support system prompts")
        messages = self.build_messages(
            prompt,
            conversation,
            image_detail=getattr(prompt.options, "image_detail", None),
        )
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None
        yielded_reasoning = False
        if stream:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            tool_calls = {}
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                reasoning = self._delta_reasoning(chunk)
                if reasoning:
                    yielded_reasoning = True
                    yield StreamEvent(type="reasoning", chunk=reasoning)
                if chunk.choices and chunk.choices[0].delta:
                    for tool_call in chunk.choices[0].delta.tool_calls or []:
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        idx = tool_call.index
                        if idx not in tool_calls:
                            tool_calls[idx] = tool_call
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_call.function.name or "",
                                tool_call_id=tool_call.id,
                            )
                        else:
                            tool_calls[
                                idx
                            ].function.arguments += tool_call.function.arguments
                        if tool_call.function.arguments:
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=tool_call.function.arguments,
                                tool_call_id=tool_calls[idx].id,
                            )
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content:
                    # Empty strings are noise (DeepSeek's first chunk
                    # with role=assistant has content="").
                    yield StreamEvent(type="text", chunk=content)
            response.response_json = remove_dict_none_values(combine_chunks(chunks))
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.id,
                            name=value.function.name,
                            arguments=json.loads(value.function.arguments or "{}"),
                        )
                    )
        else:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump()
            response.response_json = remove_dict_none_values(completion.model_dump())
            message = completion.choices[0].message
            reasoning = getattr(message, "reasoning_content", None) or ""
            if reasoning:
                yielded_reasoning = True
                yield StreamEvent(type="reasoning", chunk=reasoning)
            for tool_call in message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name or "",
                    tool_call_id=tool_call.id,
                )
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=tool_call.function.arguments or "",
                    tool_call_id=tool_call.id,
                )
            if message.content is not None:
                yield StreamEvent(type="text", chunk=message.content)
        self.set_usage(response, usage)
        if (
            usage
            and not yielded_reasoning
            and (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
        ):
            yield StreamEvent(type="reasoning", chunk="", redacted=True)
        response._prompt_json = redact_data({"messages": messages})


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

        async def execute(
            self,
            prompt: llm.Prompt,
            stream: bool,
            response: llm.AsyncResponse,
            conversation: llm.AsyncConversation | None = None,
            key: str | None = None,
        ):
            if prompt.system and not self.allows_system_prompt:
                raise NotImplementedError("Model does not support system prompts")
            messages = self.build_messages(
                prompt,
                conversation,
                image_detail=getattr(prompt.options, "image_detail", None),
            )
            kwargs = self.build_kwargs(prompt, stream)
            client = self.get_client(key, async_=True)
            usage = None
            yielded_reasoning = False
            if stream:
                completion = await client.chat.completions.create(
                    model=self.model_name or self.model_id,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )
                chunks = []
                tool_calls = {}
                async for chunk in completion:
                    chunks.append(chunk)
                    if chunk.usage:
                        usage = chunk.usage.model_dump()
                    reasoning = self._delta_reasoning(chunk)
                    if reasoning:
                        yielded_reasoning = True
                        yield StreamEvent(type="reasoning", chunk=reasoning)
                    if chunk.choices and chunk.choices[0].delta:
                        for tool_call in chunk.choices[0].delta.tool_calls or []:
                            if tool_call.function.arguments is None:
                                tool_call.function.arguments = ""
                            idx = tool_call.index
                            if idx not in tool_calls:
                                tool_calls[idx] = tool_call
                                yield StreamEvent(
                                    type="tool_call_name",
                                    chunk=tool_call.function.name or "",
                                    tool_call_id=tool_call.id,
                                )
                            else:
                                tool_calls[
                                    idx
                                ].function.arguments += tool_call.function.arguments
                            if tool_call.function.arguments:
                                yield StreamEvent(
                                    type="tool_call_args",
                                    chunk=tool_call.function.arguments,
                                    tool_call_id=tool_calls[idx].id,
                                )
                    try:
                        content = chunk.choices[0].delta.content
                    except IndexError:
                        content = None
                    if content:
                        yield StreamEvent(type="text", chunk=content)
                if tool_calls:
                    for value in tool_calls.values():
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=value.id,
                                name=value.function.name,
                                arguments=json.loads(
                                    value.function.arguments or "{}"
                                ),
                            )
                        )
                response.response_json = remove_dict_none_values(
                    combine_chunks(chunks)
                )
            else:
                completion = await client.chat.completions.create(
                    model=self.model_name or self.model_id,
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
                response.response_json = remove_dict_none_values(
                    completion.model_dump()
                )
                usage = completion.usage.model_dump()
                message = completion.choices[0].message
                reasoning = getattr(message, "reasoning_content", None) or ""
                if reasoning:
                    yielded_reasoning = True
                    yield StreamEvent(type="reasoning", chunk=reasoning)
                for tool_call in message.tool_calls or []:
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=json.loads(
                                tool_call.function.arguments or "{}"
                            ),
                        )
                    )
                    yield StreamEvent(
                        type="tool_call_name",
                        chunk=tool_call.function.name or "",
                        tool_call_id=tool_call.id,
                    )
                    yield StreamEvent(
                        type="tool_call_args",
                        chunk=tool_call.function.arguments or "",
                        tool_call_id=tool_call.id,
                    )
                if message.content is not None:
                    yield StreamEvent(type="text", chunk=message.content)
            self.set_usage(response, usage)
            if (
                usage
                and not yielded_reasoning
                and (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
            ):
                yield StreamEvent(type="reasoning", chunk="", redacted=True)
            response._prompt_json = redact_data({"messages": messages})


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
