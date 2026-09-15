from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.providers.deepseek import DeepSeekProvider, LlmRequestError


class FakeCompletions:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            ),
        ],
    )


def stream_chunk(
    *,
    reasoning: str = "",
    content: str = "",
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(
                    reasoning_content=reasoning,
                    content=content,
                ),
            ),
        ],
    )


def client(completions: FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_provider_uses_chat_defaults() -> None:
    completions = FakeCompletions(completion("Готово"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    assert provider.generate(messages=[{"role": "user", "content": "Вопрос"}]) == "Готово"
    assert completions.requests == [{
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Вопрос"}],
        "max_tokens": 2_000,
        "stream": False,
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }]


def test_provider_forwards_agent_context() -> None:
    completions = FakeCompletions(completion("Новый ответ"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
        {"role": "user", "content": "Продолжение"},
    ]

    assert provider.generate(messages=messages) == "Новый ответ"
    assert completions.requests[0]["messages"] == messages


def test_agent_provider_can_disable_thinking() -> None:
    completions = FakeCompletions(completion("Ответ"))
    provider = DeepSeekProvider(
        client=client(completions),  # type: ignore[arg-type]
        model="deepseek-v4-pro",
        thinking_enabled=False,
    )

    provider.generate(messages=[{"role": "user", "content": "Вопрос"}], max_tokens=512)

    request = completions.requests[0]
    assert request["model"] == "deepseek-v4-pro"
    assert request["max_tokens"] == 512
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in request


def test_provider_retries_empty_response_without_thinking() -> None:
    completions = FakeCompletions(
        completion(""),
        completion("Ответ после повтора"),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    assert provider.generate(messages=[{"role": "user", "content": "Вопрос"}]) == "Ответ после повтора"
    assert len(completions.requests) == 2
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in completions.requests[1]


def test_provider_does_not_retry_content_filtered_response() -> None:
    completions = FakeCompletions(completion("", finish_reason="content_filter"))
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    with pytest.raises(LlmRequestError, match="finish_reason=content_filter"):
        provider.generate(messages=[{"role": "user", "content": "Вопрос"}])

    assert len(completions.requests) == 1


def test_provider_streams_reasoning_and_content() -> None:
    completions = FakeCompletions(
        iter(
            [
                stream_chunk(reasoning="Сначала проверю факты. "),
                stream_chunk(content="Итоговый ответ.", finish_reason="stop"),
            ],
        ),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    chunks = list(provider.stream(system_prompt="system", user_prompt="user"))

    assert [chunk.reasoning for chunk in chunks if chunk.reasoning] == [
        "Сначала проверю факты. ",
    ]
    assert [chunk.content for chunk in chunks if chunk.content] == [
        "Итоговый ответ.",
    ]
    assert completions.requests[0]["stream"] is True


def test_provider_retries_empty_stream_without_thinking() -> None:
    completions = FakeCompletions(
        iter(
            [
                stream_chunk(reasoning="Думаю, но финала пока нет."),
                stream_chunk(finish_reason="stop"),
            ],
        ),
        iter([stream_chunk(content="Ответ после повтора.", finish_reason="stop")]),
    )
    provider = DeepSeekProvider(client=client(completions))  # type: ignore[arg-type]

    chunks = list(provider.stream(system_prompt="system", user_prompt="user"))

    assert any(chunk.status for chunk in chunks)
    assert "Ответ после повтора." in "".join(chunk.content for chunk in chunks)
    assert len(completions.requests) == 2
    assert completions.requests[1]["extra_body"] == {
        "thinking": {"type": "disabled"},
    }
