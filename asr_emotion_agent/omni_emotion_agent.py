"""
OmniEmotionAgent — Qwen3.5-Omni-Flash 语音情绪判断智能体 (顶层协调器)

职责:
  - 接收音频 → 交给 QwenOmniClient 调 API
  - 获取 EmotionResult → 管理历史 → 提供统计
  - 提供批量判断和趋势分析

用法:
  from asr_emotion_agent import OmniEmotionAgent, OmniConfig

  config = OmniConfig()
  config.dashscope_api_key = "sk-xxx"
  agent = OmniEmotionAgent(config)

  audio = ...  # (32000,) float32 numpy 数组, 2秒 16kHz
  result = agent.judge(audio)
  print(result.danger_level)  # "危险" | "关注" | "正常"
  print(result.reason)        # 自然语言判定理由
"""

import time
import logging
from typing import Optional, List
from collections import deque

import numpy as np

from .config import OmniConfig, EMOTION_DANGER_MAP, EMOTION_CN_MAP
from .qwen_omni_client import QwenOmniClient, EmotionResult, check_environment

logger = logging.getLogger("asr-emotion-agent.agent")


class OmniEmotionAgent:
    """
    Qwen3.5-Omni-Flash 语音情绪判断智能体

    将音频发送给 Omni 全模态大模型，通过 System Prompt 引导模型:
      1. 转写语音内容
      2. 识别情绪状态 (7种)
      3. 判断危险等级 (危险/关注/正常)
      4. 给出判定理由

    特性:
      - 自动重试 + 降级
      - 历史记录 (滑动窗口)
      - 趋势分析
      - 统计信息
    """

    # ═══════════════════════════════════════════════════════════
    # 智能体初始化 — 配置校验 + 客户端设置 + 统计计数器
    # ═══════════════════════════════════════════════════════════

    def __init__(self, config: OmniConfig):
        """
        Args:
            config: OmniConfig 配置对象
        """
        self.config = config

        # 加载环境变量
        config.load_from_env()

        # 校验
        issues = config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"配置问题: {issue}")

        # 初始化客户端
        self.client = QwenOmniClient(config)

        # 历史记录
        self._history: deque[EmotionResult] = deque(
            maxlen=config.max_history
        )

        # 统计
        self._total_judgments: int = 0
        self._danger_count: int = 0
        self._attention_count: int = 0
        self._normal_count: int = 0
        self._api_fail_count: int = 0
        self._total_latency_ms: float = 0.0

        logger.info(
            f"OmniEmotionAgent 初始化完成 | "
            f"model={config.model} | "
            f"API={'可用' if self.client.is_available else '不可用'} | "
            f"窗口={config.window_seconds}s | "
            f"阈值: 危险≥{config.danger_threshold}, 关注≥{config.attention_threshold}"
        )

    # ═══════════════════════════════════════════════════════════
    # 主判断接口 — 音频校验 + API 调用 + 历史记录更新
    # ═══════════════════════════════════════════════════════════

    def judge(self, audio: np.ndarray, sample_rate: int = None) -> EmotionResult:
        """
        对音频做情绪/危险判断

        Args:
            audio: float32 numpy 数组, shape (n_samples,)
                   建议 2 秒 = 32000 样本 @ 16kHz
            sample_rate: 采样率 (默认使用配置值)

        Returns:
            EmotionResult
        """
        sr = sample_rate or self.config.sample_rate

        # 0. 校验音频
        min_samples = int(sr * 0.3)  # 最少 0.3 秒
        if len(audio) < min_samples:
            logger.debug(f"音频太短 ({len(audio)} 样本 < {min_samples})，跳过分析")
            result = EmotionResult()
            result.danger_level = "正常"
            result.reason = f"音频太短 ({len(audio)/sr:.1f}s)，跳过分析"
            result.api_success = False
            result.timestamp = time.time()
            return result

        # 1. 调 Omni API
        result = self.client.analyze(audio, sr)

        # 2. 更新历史
        self._history.append(result)

        # 3. 更新统计
        self._total_judgments += 1
        self._total_latency_ms += result.api_latency_ms

        if result.danger_level == "危险":
            self._danger_count += 1
        elif result.danger_level == "关注":
            self._attention_count += 1
        else:
            self._normal_count += 1

        if not result.api_success:
            self._api_fail_count += 1

        # 4. 日志
        if result.danger_level == "危险":
            logger.warning(
                f"⚠️ 危险! | emotion={result.emotion_cn} | "
                f"score={result.danger_score:.2f} | {result.reason[:80]}"
            )
        elif result.danger_level == "关注":
            logger.info(
                f"🟡 关注 | emotion={result.emotion_cn} | "
                f"score={result.danger_score:.2f}"
            )

        return result

    # ═══════════════════════════════════════════════════════════
    # 批量判断 — 顺序处理多个音频窗口
    # ═══════════════════════════════════════════════════════════

    def judge_batch(self,
                    audios: List[np.ndarray],
                    sample_rate: int = None) -> List[EmotionResult]:
        """
        批量判断（顺序处理）

        Args:
            audios: 音频窗口列表
            sample_rate: 采样率

        Returns:
            EmotionResult 列表
        """
        return [self.judge(audio, sample_rate) for audio in audios]

    # ═══════════════════════════════════════════════════════════
    # 历史管理 — 最近结果的滑动窗口队列
    # ═══════════════════════════════════════════════════════════

    def get_recent_results(self, n: int = None) -> List[EmotionResult]:
        """获取最近的判定结果"""
        if n is None:
            return list(self._history)
        items = list(self._history)
        return items[-n:] if n < len(items) else items

    def get_latest_result(self) -> Optional[EmotionResult]:
        """获取最近一次判定"""
        return self._history[-1] if self._history else None

    def is_recently_dangerous(self, window: int = 5) -> bool:
        """
        最近 N 次判定中是否有"危险"

        Args:
            window: 检查最近几条记录

        Returns:
            True 表示最近 window 条记录中有危险
        """
        recent = list(self._history)[-window:]
        return any(r.danger_level == "危险" for r in recent)

    # ═══════════════════════════════════════════════════════════
    # 危险比例计算 — 最近窗口中危险判定的占比
    # ═══════════════════════════════════════════════════════════

    def get_danger_ratio(self, window: int = 10) -> float:
        """
        最近 N 次判定中危险的比例

        Returns:
            0.0 ~ 1.0
        """
        recent = list(self._history)[-window:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.danger_level == "危险") / len(recent)

    # ═══════════════════════════════════════════════════════════
    # 趋势分析 — 前后半段危险分数对比
    # ═══════════════════════════════════════════════════════════

    def get_trend(self, window: int = 10) -> str:
        """
        危险趋势分析

        比较前后半段的 danger_score 均值。

        Returns:
            "rising" (上升) | "falling" (下降) | "stable" (平稳) | "none" (数据不足)
        """
        recent = list(self._history)[-window:]
        if len(recent) < 4:
            return "none"

        mid = len(recent) // 2
        first_half = sum(r.danger_score for r in recent[:mid]) / mid
        second_half = sum(r.danger_score for r in recent[mid:]) / (len(recent) - mid)

        diff = second_half - first_half
        if diff > 0.10:
            return "rising"
        elif diff < -0.10:
            return "falling"
        else:
            return "stable"

    # ═══════════════════════════════════════════════════════════
    # 统计报告 — 聚合计数、比率、延迟、摘要
    # ═══════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """获取运行统计"""
        avg_latency = (
            self._total_latency_ms / max(1, self._total_judgments)
        )
        return {
            "total_judgments": self._total_judgments,
            "danger_count": self._danger_count,
            "attention_count": self._attention_count,
            "normal_count": self._normal_count,
            "danger_ratio": (
                self._danger_count / max(1, self._total_judgments)
            ),
            "api_fail_count": self._api_fail_count,
            "api_success_rate": (
                1.0 - self._api_fail_count / max(1, self._total_judgments)
            ),
            "avg_latency_ms": round(avg_latency, 1),
            "model": self.config.model,
            "api_available": self.client.is_available,
            "history_size": len(self._history),
            "trend": self.get_trend(),
            "recent_danger_ratio": self.get_danger_ratio(),
        }

    def get_summary(self) -> str:
        """生成人类可读的统计摘要"""
        s = self.get_stats()
        trend_cn = {"rising": "↑上升", "falling": "↓下降", "stable": "→平稳", "none": "-"}
        return (
            f"OmniEmotionAgent 统计:\n"
            f"  总判定: {s['total_judgments']} | "
            f"危险: {s['danger_count']} | "
            f"关注: {s['attention_count']} | "
            f"正常: {s['normal_count']}\n"
            f"  危险率: {s['danger_ratio']:.1%} | "
            f"API成功率: {s['api_success_rate']:.1%} | "
            f"平均延迟: {s['avg_latency_ms']}ms\n"
            f"  趋势: {trend_cn.get(s['trend'], '-')} | "
            f"API: {'✓在线' if s['api_available'] else '✗离线'}"
        )

    # ═══════════════════════════════════════════════════════════
    # 重置功能 — 清空所有历史与计数器
    # ═══════════════════════════════════════════════════════════

    def reset(self):
        """重置所有状态"""
        self._history.clear()
        self._total_judgments = 0
        self._danger_count = 0
        self._attention_count = 0
        self._normal_count = 0
        self._api_fail_count = 0
        self._total_latency_ms = 0.0
        logger.info("OmniEmotionAgent 状态已重置")


# ═══════════════════════════════════════════════════════════
# 工厂函数 — 一行创建智能体（合理默认值）
# ═══════════════════════════════════════════════════════════

def create_agent(
    api_key: str = None,
    model: str = "qwen3.5-omni-flash",
    window_seconds: float = 2.0,
    danger_threshold: float = 0.60,
    **kwargs,
) -> OmniEmotionAgent:
    """
    快速创建 OmniEmotionAgent

    Args:
        api_key: DashScope API Key (不传则从环境变量 DASHSCOPE_API_KEY 读取)
        model: 模型名
        window_seconds: 音频窗口大小 (秒)
        danger_threshold: 危险判定阈值
        **kwargs: 其他 OmniConfig 参数

    Returns:
        OmniEmotionAgent
    """
    config = OmniConfig()
    config.model = model
    config.window_seconds = window_seconds
    config.danger_threshold = danger_threshold

    if api_key:
        config.dashscope_api_key = api_key

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.load_from_env()
    return OmniEmotionAgent(config)
