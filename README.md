# llm-deepseek

[![Changelog](https://img.shields.io/github/v/release/dannyob/llm-deepseek?include_prereleases&label=changelog)](https://github.com/dannyob/llm-deepseek/releases)
[![Tests](https://github.com/dannyob/llm-deepseek/actions/workflows/test.yml/badge.svg)](https://github.com/dannyob/llm-deepseek/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/dannyob/llm-deepseek/blob/main/LICENSE)

DeepSeek v4 (and the older chat/coder/reasoner models) for the [LLM](https://llm.datasette.io/) CLI, with tool calling.

```bash
llm -m deepseek-v4-pro 'write a haiku about databases' \
  -o thinking enabled \
  -o reasoning_effort xhigh
```

This is a fork of [abrasumente233/llm-deepseek](https://github.com/abrasumente233/llm-deepseek), which has stopped accepting PRs. It picks up two contributions that were never merged upstream:

- DeepSeek v4 models (`deepseek-v4-flash`, `deepseek-v4-pro`) plus the `thinking` and `reasoning_effort` options, by [@microdog](https://github.com/microdog) ([PR](https://github.com/rumisle/llm-deepseek/pull/5))
- Tool calling for all DeepSeek models, by [@michaelmdeng](https://github.com/michaelmdeng) ([PR](https://github.com/rumisle/llm-deepseek/pull/3))

## Install

Not published to PyPI — install from source:

```bash
git clone https://github.com/dannyob/llm-deepseek.git
llm install -e ./llm-deepseek
```

## Usage

Obtain a [DeepSeek API key](https://platform.deepseek.com/api_keys) and save it:

```bash
llm keys set deepseek
# <Paste key here>
```

Run `llm models` to see registered models. Then:

```bash
llm -m deepseek-chat 'five great names for a pet ocelot'
llm -m deepseek-reasoner 'solve \int \frac{\ln(x)\arctan(x)}{x^2+1} dx'
llm -m deepseek-coder 'how to reverse a linked list in python'
llm -m deepseek-v4-flash 'summarize the benefits of unit tests'
llm -m deepseek-v4-pro 'explain how vector databases work'
```

Standard OpenAI-compatible options work (`max_tokens`, `top_p`, `stop`, `json_object`), plus DeepSeek-specific ones (`thinking`, `reasoning_effort`).

## Development

```bash
cd llm-deepseek
python3 -m venv venv
source venv/bin/activate
llm install -e '.[test]'
pytest
```
