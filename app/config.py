"""
全局配置管理模块
集中管理 LLM、SeekDB 和系统级别的配置项。
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 加载 .env 文件
load_dotenv()

# --- 项目根路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DeepSeekConfig(BaseModel):
    """DeepSeek API 配置"""
    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    chat_model: str = "deepseek-chat"


class SiliconFlowConfig(BaseModel):
    """硅基流动 API 配置 (用于 Embedding)"""
    api_key: str = Field(default_factory=lambda: os.getenv("SILICONFLOW_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    embedding_model: str = "BAAI/bge-m3"


class SeekDBConfig(BaseModel):
    """SeekDB 嵌入式数据库配置"""
    path: str = Field(default_factory=lambda: str((PROJECT_ROOT / "data" / "seekdb").resolve()))
    database: str = Field(default_factory=lambda: os.getenv("SEEKDB_DATABASE", "law_judge"))


class SystemConfig(BaseModel):
    """系统级配置"""
    # 庭前举证质证循环最大轮数
    max_evidence_rounds: int = 3
    # 法庭辩论最大交锋轮数
    max_debate_rounds: int = 2
    # 混合检索权重 (0.0=纯全文, 1.0=纯向量)
    law_weight_alpha: float = 0.4
    case_weight_alpha: float = 0.7
    # Whoosh 索引路径
    whoosh_path: str = Field(default_factory=lambda: str((PROJECT_ROOT / "data" / "whoosh").resolve()))
    # 防幻觉 Prompt 前缀
    anti_hallucination_prefix: str = (
        "【重要约束】你必须严格且仅基于以下召回文本进行法律规则的解释 and 应用。"
        "不得使用你的预训练知识捏造任何法律条文、案例或裁判规则。"
        "若召回文本未提供相关信息，你必须明确回答：依据现有检索结果，尚无法提供该问题的确切法律依据。"
    )


class AppConfig(BaseModel):
    """整合所有配置的根配置"""
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    siliconflow: SiliconFlowConfig = Field(default_factory=SiliconFlowConfig)
    seekdb: SeekDBConfig = Field(default_factory=SeekDBConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


# 全局单例配置实例
settings = AppConfig()
