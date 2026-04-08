"""应用配置模型及环境加载。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DeepSeekConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    chat_model: str = "deepseek-chat"


class SiliconFlowConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    embedding_model: str = "BAAI/bge-m3"


class SeekDBConfig(BaseModel):
    path: str = Field(default_factory=lambda: str((PROJECT_ROOT / "data" / "seekdb").resolve()))
    database: str = Field(default_factory=lambda: os.getenv("SEEKDB_DATABASE", "law_judge"))


class SystemConfig(BaseModel):
    max_evidence_rounds: int = 3
    max_debate_rounds: int = 2
    max_slot_ask_rounds: int = 3
    law_weight_alpha: float = 0.4
    case_weight_alpha: float = 0.7
    whoosh_path: str = Field(default_factory=lambda: str((PROJECT_ROOT / "data" / "whoosh").resolve()))
    anti_hallucination_prefix: str = (
        "【重要约束】你必须严格且仅基于检索到的内容进行法律解释与适用。"
        "不得编造法条、判例或裁判规则。"
        "若检索结果不足，请明确说明当前证据不足，无法给出确定法律结论。"
    )


class AppConfig(BaseModel):
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    siliconflow: SiliconFlowConfig = Field(default_factory=SiliconFlowConfig)
    seekdb: SeekDBConfig = Field(default_factory=SeekDBConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


settings = AppConfig()
