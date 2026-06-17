"""
通用工具函数
日志、时间戳、配置加载等
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False


# --- 项目根目录 ---
def get_project_root() -> Path:
    """获取项目根目录（包含 config.yaml 的目录）"""
    # 从当前文件向上查找
    current = Path(__file__).resolve().parent.parent
    while current != current.parent:
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    # fallback: common/ 的父目录
    return Path(__file__).resolve().parent.parent


# --- 配置加载 ---
def load_config(config_path: Optional[str] = None) -> dict:
    """加载 YAML 配置文件"""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"[WARNING] 配置文件不存在: {config_path}，使用默认值")
        return {}

    if not _YAML_AVAILABLE:
        print("[WARNING] PyYAML 未安装，返回空配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- 日志 ---
def setup_logging(name: str = "audio_monitor",
                  level: str = "INFO",
                  log_dir: Optional[str] = None) -> logging.Logger:
    """配置日志系统，同时输出到控制台和文件"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 格式
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --- 时间戳 ---
def now() -> float:
    """获取当前时间戳 (秒)"""
    return time.time()


def timestamp_str(ts: Optional[float] = None) -> str:
    """时间戳 → 可读字符串 HH:MM:SS.mmm"""
    if ts is None:
        ts = time.time()
    t = time.localtime(ts)
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{int((ts % 1) * 1000):03d}"


# --- 延迟计算 ---
def calc_latency_ms(send_time: float, receive_time: Optional[float] = None) -> float:
    """计算延迟 (ms)"""
    if receive_time is None:
        receive_time = time.time()
    return (receive_time - send_time) * 1000
