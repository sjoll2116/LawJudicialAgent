"""
向量嵌入函数模块
基于 SiliconFlow 的 BGE-M3 模型实现文本向量生成，并提供重试机制。
"""

from __future__ import annotations
import time
import logging
from typing import List, Union

from openai import OpenAI
import chromadb.utils.embedding_functions as ef
from app.config import settings

logger = logging.getLogger(__name__)

Documents = Union[str, List[str]]
Embeddings = List[List[float]]

class SiliconFlowEmbeddingFunction(ef.EmbeddingFunction):
    """基于 SiliconFlow 的向量生成函数，含重试机制"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        dimension: int = 1024,
    ):
        self._api_key = api_key or settings.siliconflow.api_key
        self._base_url = base_url or settings.siliconflow.base_url
        self._model_name = model_name or settings.siliconflow.embedding_model
        self._dimension = dimension
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def __call__(self, input: Documents) -> Embeddings:
        """
        将文本转换为向量。
        实现分批处理（每批 60 个）以及指数退避重试 (1s, 2s, 4s, 8s, 16s)
        """
        if isinstance(input, str):
            input = [input]

        batch_size = 60
        all_embeddings: Embeddings = []

        # 分批处理逻辑
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            max_retries = 5
            batch_success = False

            for attempt in range(max_retries):
                try:
                    response = self._client.embeddings.create(
                        model=self._model_name,
                        input=batch,
                    )
                    all_embeddings.extend([item.embedding for item in response.data])
                    batch_success = True
                    break
                except Exception as e:
                    wait_time = 2 ** attempt
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Embedding API 批次调用失败 (起始索引 {i}): {e}。将在 {wait_time}s 后进行第 {attempt + 2} 次重试..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Embedding API 在 {max_retries} 次重试后仍然失败。")
                        raise e
            
            if not batch_success:
                raise RuntimeError(f"批次 {i} 向量生成失败")

        return all_embeddings