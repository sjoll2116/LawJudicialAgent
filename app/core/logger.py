"""
全局日志配置中心
提供分布式链式追踪 (Request ID) 与滚动文件存储功能。
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# 全局上下文变量，用于追踪 Request ID
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

class ContextFilter(logging.Filter):
    """
    日志过滤器：将 ContextVar 中的 request_id 注入到日志记录中。
    """
    def filter(self, record):
        record.request_id = request_id_ctx.get() or "SYSTEM"
        return True

def setup_app_logging(project_root: Path):
    """
    初始化全局日志配置：支持控制台与滚动文件双输出。
    """
    log_dir = project_root / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = log_dir / "app.log"
    
    # 1. 定义统一格式
    # 包含：时间 | 级别 | [追踪ID] | Logger名 | 消息内容
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 2. 设置根日志器 (Root Logger)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清理旧的处理程序
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. 滚动文件处理程序 (Rolling File Handler)
    # 单个文件 10MB，保留最近 5 个备份
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10 * 1024 * 1024, 
        backupCount=5, 
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    # 4. 控制台处理程序 (Console Handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Project Root: {project_root}")
    logging.info(f"Persistent log file: {log_file}")

def get_logger(name: str):
    """
    获取带名称的 Logger 实例。
    """
    return logging.getLogger(name)
