"""OpenAI provider (imported lazily so `openai` is only required when used)."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ai_harness.providers.base import LLMMessage, LLMResult


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(api_key=self._api_key)
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
        payload = [
            {"role": "user" if m.role == "tool" else m.role, "content": m.content} for m in messages
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            model=model,
        )
