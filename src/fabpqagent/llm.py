"""PROVIDED. OpenRouter chat client with tool calling, model fallback and usage tracking.

`LLM` is the interface the agent depends on. `ScriptedLLM` replays fixed replies so the agent
can be tested without network calls (same idea as CS329z's ScriptedLM).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = ("google/gemma-4-26b-a4b-it:free", "nvidia/nemotron-3.5-lightning:free")
REQUEST_TIMEOUT_S = 90


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model wrote it


@dataclass(frozen=True)
class LLMReply:
    content: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


class LLMError(RuntimeError):
    pass


class OpenRouterLLM:
    """Tries each model in order; free models are often rate-limited, so fallback matters."""

    def __init__(self, models: tuple[str, ...] | None = None, api_key: str | None = None, max_tokens: int = 1200):
        from openai import OpenAI  # imported here so tests never need network setup

        load_dotenv()
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set (see .env.example)")
        env_models = os.environ.get("PQ_MODELS")
        self.models = models or (tuple(m.strip() for m in env_models.split(",")) if env_models else DEFAULT_MODELS)
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, timeout=REQUEST_TIMEOUT_S, max_retries=1)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        errors = []
        for model in self.models:
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=model, messages=messages, tools=tools or None,
                    max_tokens=self.max_tokens, temperature=0.1,
                )
            except Exception as err:  # SDK raises many types (rate limit, 5xx, timeout)
                errors.append(f"{model}: {err}")
                continue
            if not resp.choices:
                errors.append(f"{model}: no choices returned")
                continue
            msg = resp.choices[0].message
            calls = tuple(ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (msg.tool_calls or []))
            usage = resp.usage
            return LLMReply(
                content=msg.content or "",
                tool_calls=calls,
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_s=time.perf_counter() - start,
            )
        raise LLMError("All models failed:\n" + "\n".join(errors))


@dataclass
class ScriptedLLM:
    """Test double: returns `replies` in order and records every request it received."""

    replies: list[LLMReply]
    requests: list[list[dict[str, Any]]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMReply:
        self.requests.append([dict(m) for m in messages])
        if not self.replies:
            raise LLMError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


def text_reply(content: str) -> LLMReply:
    return LLMReply(content=content, tool_calls=(), model="scripted")


def tool_reply(name: str, arguments: str, call_id: str = "call_1") -> LLMReply:
    return LLMReply(content="", tool_calls=(ToolCall(call_id, name, arguments),), model="scripted")
