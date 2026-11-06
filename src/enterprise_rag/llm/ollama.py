"""Ollama HTTP client (chat + embeddings) over a shared, pooled httpx.AsyncClient."""

from __future__ import annotations

from typing import Any

import httpx

from enterprise_rag.llm.base import LLM, Message


class OllamaError(RuntimeError):
    pass


class OllamaLLM(LLM):
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        keep_alive: str = "30m",
    ) -> None:
        self._client = client
        self.model = model
        self._options = {"temperature": temperature, "num_ctx": num_ctx, "seed": 0}
        self._keep_alive = keep_alive

    async def chat(self, messages: list[Message], *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": self._options,
            "keep_alive": self._keep_alive,
        }
        if json_mode:
            payload["format"] = "json"
        resp = await self._client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            raise OllamaError(f"chat failed ({resp.status_code}): {resp.text[:200]}")
        return str(resp.json()["message"]["content"])


async def ollama_embed(
    client: httpx.AsyncClient, model: str, inputs: list[str]
) -> list[list[float]]:
    resp = await client.post(
        "/api/embed", json={"model": model, "input": inputs, "keep_alive": "30m"}
    )
    if resp.status_code != 200:
        raise OllamaError(f"embed failed ({resp.status_code}): {resp.text[:200]}")
    embeddings: list[list[float]] = resp.json()["embeddings"]
    if len(embeddings) != len(inputs):
        raise OllamaError("embedding count mismatch")
    return embeddings
