"""
结构化输出中间件
确保 LLM 返回的结果能够稳定解析为 JSON 格式，并映射到 Pydantic 模型。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm import chat_completion


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def call_with_json_retry(
    *,
    messages: list[dict[str, str]],
    schema_model: type[BaseModel],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    retries: int = 2,
) -> tuple[BaseModel | None, str]:
    """
    Call chat completion in JSON mode and validate with Pydantic.

    Returns (validated_model_or_none, last_raw_text).
    """
    attempt_messages = list(messages)
    last_raw = ""

    for _ in range(retries + 1):
        raw = chat_completion(
            attempt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        last_raw = raw

        parsed = _parse_json(raw)
        if parsed is not None:
            try:
                return schema_model.model_validate(parsed), raw
            except ValidationError:
                pass

        # Self-heal retry: ask model to re-output strict JSON that matches schema.
        attempt_messages = attempt_messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "上一次输出未通过结构化校验。请只输出一个合法 JSON 对象，"
                    f"严格满足该 schema（字段名必须一致）: {schema_model.model_json_schema()}"
                ),
            },
        ]

    return None, last_raw
