"""
LLM 服务封装模块
对 DeepSeek API 进行统一封装，为所有 Agent 提供底层推理能力。
"""

from openai import OpenAI
from app.config import settings


import time
from app.core.logger import get_logger

logger = get_logger(__name__)

def get_llm_client() -> OpenAI:
    """获取 DeepSeek LLM 客户端 (OpenAI 兼容接口)"""
    return OpenAI(
        api_key=settings.deepseek.api_key,
        base_url=settings.deepseek.base_url,
    )


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
    **kwargs,
) -> str:
    """
    调用 DeepSeek Chat Completion API。
    """
    client = get_llm_client()
    model = model or settings.deepseek.chat_model

    start_time = time.time()
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        duration = time.time() - start_time
        usage = response.usage
        logger.info(
            f"LLM Call Success | Model: {model} | JSON_Mode: {json_mode} | "
            f"Tokens: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}) | "
            f"Latency: {duration:.3f}s"
        )
        return response.choices[0].message.content
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"LLM Call Failed | Model: {model} | Error: {str(e)} | Latency: {duration:.3f}s", exc_info=True)
        raise e
