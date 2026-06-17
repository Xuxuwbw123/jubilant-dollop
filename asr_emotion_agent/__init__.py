"""
asr-emotion-agent — 基于 Qwen3.5-Omni-Flash 的语音情绪判断智能体

前端麦克风采集 → WebSocket 流式传输 → 后端调 Omni API → 情绪识别 → 危险/正常

快速开始:
  from asr_emotion_agent import OmniEmotionAgent, OmniConfig, create_agent

  # 方式1: 工厂函数
  agent = create_agent(api_key="sk-xxx")

  # 方式2: 手动配置
  config = OmniConfig()
  config.dashscope_api_key = "sk-xxx"
  agent = OmniEmotionAgent(config)

  # 判断音频
  audio = ...  # (32000,) float32 numpy 数组, 2秒 16kHz
  result = agent.judge(audio)

  print(result.danger_level)  # "危险" | "关注" | "正常"
  print(result.emotion_cn)    # "恐惧" | "愤怒" | ...
  print(result.text)          # 转写文本
  print(result.reason)        # 判定理由

模块:
  OmniEmotionAgent  — 顶层智能体 (judge + 历史 + 统计)
  QwenOmniClient    — OpenAI 兼容 API 封装 (流式接收 + JSON 解析)
  OmniConfig        — 配置类
  EmotionResult     — 情绪分析结果数据结构
"""

from .config import (
    OmniConfig,
    SYSTEM_PROMPT,
    EMOTION_DANGER_MAP,
    EMOTION_CN_MAP,
    VALID_EMOTIONS,
)
from .qwen_omni_client import (
    QwenOmniClient,
    EmotionResult,
    check_environment,
)
from .omni_emotion_agent import (
    OmniEmotionAgent,
    create_agent,
)

__all__ = [
    # 核心类
    "OmniEmotionAgent",
    "QwenOmniClient",
    "OmniConfig",
    "EmotionResult",
    # 工厂函数
    "create_agent",
    # 工具
    "check_environment",
    # 常量
    "SYSTEM_PROMPT",
    "EMOTION_DANGER_MAP",
    "EMOTION_CN_MAP",
    "VALID_EMOTIONS",
]

__version__ = "2.0.0"
