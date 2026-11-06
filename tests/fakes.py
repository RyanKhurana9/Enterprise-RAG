"""Test doubles shared by unit and integration tests."""

from __future__ import annotations

from collections.abc import Callable

from enterprise_rag.llm.base import LLM, Message


class FakeLLM(LLM):
    """Returns scripted replies (a list consumed in order, or a function of the messages)."""

    def __init__(self, replies: list[str] | Callable[[list[Message]], str]) -> None:
        self._replies = replies
        self.calls: list[list[Message]] = []

    async def chat(
        self, messages: list[Message], *, json_mode: bool = False, max_tokens: int | None = None
    ) -> str:
        self.calls.append(messages)
        if callable(self._replies):
            return self._replies(messages)
        if not self._replies:
            raise RuntimeError("FakeLLM exhausted")
        return self._replies.pop(0)
