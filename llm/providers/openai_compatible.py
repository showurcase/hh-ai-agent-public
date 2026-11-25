from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import aiohttp

from llm.errors import (
    LLMError,
    LLMInvalidResponseError,
    LLMTimeoutError,
    LLMTransientError,
    error_for_http_status,
)
from llm.types import LLMRequest, LLMResponse


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        json_mode: bool,
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        session: Any = None,
    ):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.json_mode = json_mode
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_effort = reasoning_effort
        self._session = session
        self._owns_session = session is None
        # OpenCode Go rejects requests without a session id.
        self._extra_headers: dict[str, str] = {}
        if "opencode.ai" in base_url:
            self._extra_headers["x-opencode-session"] = uuid.uuid4().hex

    def _get_session(self) -> Any:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def complete(self, request: LLMRequest) -> LLMResponse:
        system_instructions = request.system_instructions
        if request.json_schema is not None:
            system_instructions += (
                "\nReturn only a JSON object matching this JSON schema:\n"
                + json.dumps(request.json_schema, ensure_ascii=False)
            )
        user_content: object = request.user_content
        if request.image_base64:
            user_content = [
                {"type": "text", "text": request.user_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{request.image_mime_type};base64,"
                            f"{request.image_base64}"
                        )
                    },
                },
            ]
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_content},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_schema is not None and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        elif self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}
        started = time.perf_counter()
        try:
            async with self._get_session().post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self._extra_headers,
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=request.timeout_seconds,
                    sock_connect=min(15, request.timeout_seconds),
                    sock_read=request.timeout_seconds,
                ),
            ) as response:
                if not 200 <= response.status < 300:
                    raise error_for_http_status(
                        response.status, response.headers.get("Retry-After")
                    )
                try:
                    data = await response.json()
                except (TypeError, ValueError) as exc:
                    raise LLMInvalidResponseError() from exc
        except LLMError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise LLMTimeoutError() from exc
        except aiohttp.ClientError as exc:
            raise LLMTransientError() from exc
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError() from exc
        if not isinstance(text, str):
            raise LLMInvalidResponseError()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model") if isinstance(data.get("model"), str) else request.model,
            request_id=data.get("id") if isinstance(data.get("id"), str) else None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), int)
            else None,
            output_tokens=usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), int)
            else None,
            finish_reason=choice.get("finish_reason")
            if isinstance(choice.get("finish_reason"), str)
            else None,
        )

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
