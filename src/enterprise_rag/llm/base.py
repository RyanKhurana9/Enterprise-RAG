"""LLM abstraction so generation/routing/Text2SQL can be tested with a fake model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class LLM(ABC):
    @abstractmethod
    async def chat(self, messages: list[Message], *, json_mode: bool = False) -> str:
        """Return the assistant's reply. ``json_mode`` asks for a JSON object."""

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        return await self.chat(
            [Message("system", system), Message("user", user)], json_mode=json_mode
        )
