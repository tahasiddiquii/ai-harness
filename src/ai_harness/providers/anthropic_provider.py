"""Anthropic provider (imported lazily so `anthropic` is only required when used)."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ai_harness.providers.base import LLMMessage, LLMResult


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            from anthropic import Anthropic  # lazy import

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        client = self._client_or_init()
        system = "\n".join(m.content for m in messages if m.role == "system")
        conversation = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        resp = client.messages.create(
            model=model,
            system=system or None,
            messages=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = resp.usage
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
            model=model,
        )
