"""
QwenOmniClient — Qwen3.5-Omni-Flash OpenAI 兼容 API 封装

职责:
  - 将 numpy 音频编码为 Base64 WAV
  - 构建 OpenAI 兼容请求 (含 System Prompt)
  - 流式接收 SSE 响应 (stream=True 是强制的)
  - 解析模型返回的 JSON → EmotionResult

依赖: pip install openai numpy
"""

import re
import json
import time
import base64
import io
import wave
import logging
from typing import Optional

import numpy as np

from .config import (
    OmniConfig,
    EMOTION_DANGER_MAP,
    EMOTION_CN_MAP,
    VALID_EMOTIONS,
    SYSTEM_PROMPT,
)

logger = logging.getLogger("asr-emotion-agent.client")

# OpenAI SDK 可选导入
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    logger.warning("openai 未安装。请运行: pip install openai")


# ═══════════════════════════════════════════════════════════
# 数据结构 — EmotionResult 数据类（Omni 全部输出字段）
# ═══════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class EmotionResult:
    """Omni 情绪分析结果（最终输出）"""

    # Omni 返回
    text: str = ""                      # 转写文本
    emotion: str = "neutral"            # 情绪标签 (英文)
    emotion_cn: str = "中性"            # 情绪标签 (中文)
    emotion_confidence: float = 0.0     # 情绪置信度 [0, 1]

    # 判定
    danger_level: str = "正常"          # "危险" | "关注" | "正常"
    danger_score: float = 0.0           # 危险评分 [0, 1]

    # 详情
    reason: str = ""                    # 判定理由
    tone_description: str = ""          # 语气描述
    keywords_detected: list = field(default_factory=list)

    # 元信息
    api_success: bool = False           # API 是否成功
    api_latency_ms: float = 0.0         # API 延迟 (毫秒)
    model_used: str = ""                # 实际使用的模型
    raw_response: str = ""              # 原始响应文本 (调试用)
    error_message: str = ""             # 错误信息
    timestamp: float = 0.0              # Unix 时间戳


# ═══════════════════════════════════════════════════════════
# QwenOmniClient — OpenAI-compatible API wrapper
# ═══════════════════════════════════════════════════════════

class QwenOmniClient:
    """
    Qwen3.5-Omni-Flash OpenAI 兼容 API 封装

    用法:
        config = OmniConfig()
        config.dashscope_api_key = "sk-xxx"
        client = QwenOmniClient(config)

        result = client.analyze(audio_2sec)  # audio: (32000,) float32
        print(result.danger_level, result.reason)
    """

    # ═══════════════════════════════════════════════════════════
    # 客户端初始化 — OpenAI SDK 配置 DashScope 端点
    # ═══════════════════════════════════════════════════════════

    def __init__(self, config: OmniConfig):
        self.config = config
        self._openai_client = None
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        if not OPENAI_AVAILABLE:
            logger.error("openai SDK 未安装，无法调用 Omni API")
            return

        if not self.config.dashscope_api_key:
            logger.warning("dashscope_api_key 未设置，将无法调用 API")
            return

        try:
            self._openai_client = OpenAI(
                api_key=self.config.dashscope_api_key,
                base_url=self.config.api_base_url,
                timeout=self.config.timeout,
            )
            logger.info(
                f"QwenOmniClient 初始化完成 | model={self.config.model} | "
                f"base_url={self.config.api_base_url}"
            )
        except Exception as e:
            logger.error(f"OpenAI 客户端初始化失败: {e}")
            self._openai_client = None

    # ═══════════════════════════════════════════════════════════
    # 主分析接口 — 重试循环 + 音频编码 + API 调用 + JSON 解析
    # ═══════════════════════════════════════════════════════════

    def analyze(self,
                audio: np.ndarray,
                sample_rate: int = None) -> EmotionResult:
        """
        发送音频到 Omni 模型，获取情绪分析结果

        Args:
            audio: float32 numpy 数组, shape (n_samples,)
            sample_rate: 采样率，默认使用配置值

        Returns:
            EmotionResult
        """
        sr = sample_rate or self.config.sample_rate
        t_start = time.time()

        # 0. 检查客户端可用性
        if not self._openai_client:
            return self._fallback_result("OpenAI 客户端未初始化", t_start)

        # 1. 音频编码
        try:
            b64_audio = self._audio_to_b64(audio, sr)
        except Exception as e:
            return self._fallback_result(f"音频编码失败: {e}", t_start)

        # 2. 重试循环
        last_error = None
        for attempt in range(self.config.retry_count + 1):
            try:
                # 3. 构建消息
                messages = self._build_messages(b64_audio)

                # 4. 调 API (流式)
                full_text = self._call_api_stream(messages)

                # 5. 解析 JSON
                result = self._parse_response(full_text)
                result.api_success = True
                result.api_latency_ms = (time.time() - t_start) * 1000
                result.model_used = self.config.model
                result.timestamp = t_start
                result.raw_response = full_text[:500]  # 截断保存

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Omni API 调用失败 (attempt {attempt + 1}/"
                    f"{self.config.retry_count + 1}): {e}"
                )
                if attempt < self.config.retry_count:
                    backoff = 1.0 * (2 ** attempt)
                    logger.info(f"等待 {backoff}s 后重试...")
                    time.sleep(backoff)

        # 全部重试失败 → 降级
        logger.error(f"Omni API 全部重试失败: {last_error}")
        return self._fallback_result(str(last_error), t_start)

    # ═══════════════════════════════════════════════════════════
    # 音频编码 — float32→int16→WAV 字节→base64 字符串
    # ═══════════════════════════════════════════════════════════

    def _audio_to_b64(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        float32 numpy 音频 → Base64 WAV 字符串

        Args:
            audio: (n_samples,) float32
            sample_rate: 采样率

        Returns:
            Base64 编码的 WAV 数据 (不含 data:xxx 前缀)
        """
        # float32 → int16
        audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        # 写入内存 WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)       # 单声道
            wf.setsampwidth(2)       # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_i16.tobytes())

        wav_bytes = buf.getvalue()
        return base64.b64encode(wav_bytes).decode("ascii")

    # ═══════════════════════════════════════════════════════════
    # 消息构建 — System Prompt + 音频（OpenAI 兼容格式）
    # ═══════════════════════════════════════════════════════════

    def _build_messages(self, b64_audio: str) -> list[dict]:
        """
        构建 OpenAI 兼容的消息列表

        System Prompt 放在 messages[0]，音频+指令放在 messages[1]

        注意: Qwen-Omni 要求 base64 音频数据必须以 "data:;base64," 为前缀,
              不是标准 OpenAI 的 "data:audio/wav;base64," 格式!
        """
        # Qwen-Omni 特有格式: data:;base64,<base64_data>
        data_url = f"data:;base64,{b64_audio}"

        return [
            {
                "role": "system",
                "content": self.config.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_url,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请分析这段音频。",
                    },
                ],
            },
        ]

    # ═══════════════════════════════════════════════════════════
    # 流式 API 调用 — SSE 分块拼接（stream=True 强制要求）
    # ═══════════════════════════════════════════════════════════

    def _call_api_stream(self, messages: list[dict]) -> str:
        """
        调用 Omni API (流式模式) 并拼接完整响应

        stream=True 是 Omni 模型的强制要求。

        Returns:
            模型输出的完整文本
        """
        response = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            modalities=["text"],
            stream=True,                              # ★ 必须
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stream_options={"include_usage": True},
        )

        full_text = ""
        chunk_count = 0

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
                chunk_count += 1

        logger.debug(
            f"Omni 流式响应完成 | chunks={chunk_count} | "
            f"text_len={len(full_text)}"
        )

        if not full_text:
            raise RuntimeError("Omni 返回了空响应（可能音频太短或无人声）")

        return full_text

    # ═══════════════════════════════════════════════════════════
    # 响应解析 — 三策略 JSON 提取 + 优雅降级
    # ═══════════════════════════════════════════════════════════

    def _parse_response(self, raw_text: str) -> EmotionResult:
        """
        解析模型返回的 JSON → EmotionResult

        处理策略:
          1. 尝试直接解析整个文本为 JSON
          2. 如果失败，尝试用正则提取 JSON 块
          3. 如果仍失败，降级
        """
        # 策略1: 直接解析
        try:
            data = json.loads(raw_text.strip())
            return self._validate_and_build(data)
        except json.JSONDecodeError:
            pass

        # 策略2: 正则提取 JSON 块
        json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_build(data)
            except json.JSONDecodeError:
                pass

        # 策略3: 尝试找更宽松的 JSON (允许嵌套)
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_and_build(data)
            except json.JSONDecodeError:
                pass

        # 全部失败
        logger.warning(f"JSON 解析失败，原始文本: {raw_text[:200]}")
        result = EmotionResult()
        result.api_success = True
        result.raw_response = raw_text
        result.reason = f"JSON解析失败，原始输出: {raw_text[:100]}"
        # 使用降级逻辑
        result.danger_level = "正常"
        result.danger_score = 0.0
        result.emotion = "neutral"
        result.emotion_cn = "中性"
        return result

    def _validate_and_build(self, data: dict) -> EmotionResult:
        """
        校验 JSON 字段并构建 EmotionResult

        对缺失或非法的字段做容错处理。
        """
        result = EmotionResult()
        result.api_success = True

        # 转写文本
        result.text = str(data.get("text", "")).strip()

        # 情绪
        emotion = str(data.get("emotion", "neutral")).lower().strip()
        if emotion in VALID_EMOTIONS:
            result.emotion = emotion
        else:
            logger.warning(f"模型返回了未知情绪: '{emotion}'，降级为 neutral")
            result.emotion = "neutral"
        result.emotion_cn = EMOTION_CN_MAP.get(result.emotion, "未知")
        result.emotion_confidence = self._safe_float(
            data.get("emotion_confidence"), 0.0, 0.0, 1.0
        )

        # 危险判定
        danger_level = str(data.get("danger_level", "正常")).strip()
        if danger_level in ("危险", "关注", "正常"):
            result.danger_level = danger_level
        else:
            # 尝试映射英文
            danger_map = {"danger": "危险", "attention": "关注", "normal": "正常"}
            result.danger_level = danger_map.get(danger_level.lower(), "正常")

        result.danger_score = self._safe_float(
            data.get("danger_score"), 0.0, 0.0, 1.0
        )

        # 详情
        result.reason = str(data.get("reason", ""))
        result.tone_description = str(data.get("tone_description", ""))

        # 如果模型返回了危险评分但没有正确设置 danger_level，自动修正
        if result.danger_score >= self.config.danger_threshold:
            result.danger_level = "危险"
        elif result.danger_score >= self.config.attention_threshold:
            if result.danger_level == "正常":
                result.danger_level = "关注"

        # 如果模型返回了情绪但没有评分，用本地映射补充
        if result.danger_score == 0.0 and result.emotion in EMOTION_DANGER_MAP:
            result.danger_score = EMOTION_DANGER_MAP[result.emotion]
            if result.danger_score >= self.config.danger_threshold:
                result.danger_level = "危险"

        # 敏感词检测 (本地补充)
        if result.text:
            result.keywords_detected = self._scan_keywords(result.text)

        return result

    def _safe_float(self, value, default: float,
                    min_val: float = None, max_val: float = None) -> float:
        """安全地将值转为 float，超出范围则 clamp"""
        try:
            v = float(value)
            if min_val is not None and v < min_val:
                v = min_val
            if max_val is not None and v > max_val:
                v = max_val
            return v
        except (TypeError, ValueError):
            return default

    def _scan_keywords(self, text: str) -> list[str]:
        """扫描文本中的敏感关键词"""
        keywords = [
            "救命", "救救我", "来人", "帮帮我", "报警", "不行了", "放开我",
            "打死", "杀了", "去死", "不要", "呜呜",
        ]
        return [kw for kw in keywords if kw in text]

    # ═══════════════════════════════════════════════════════════
    # 降级与错误处理 — API 失败时返回默认安全结果
    # ═══════════════════════════════════════════════════════════

    def _fallback_result(self, error_msg: str, t_start: float) -> EmotionResult:
        """生成降级结果 (API 不可用)"""
        return EmotionResult(
            text="",
            emotion="neutral",
            emotion_cn="中性",
            emotion_confidence=0.0,
            danger_level="正常",
            danger_score=0.0,
            reason=f"Omni API 不可用: {error_msg}",
            api_success=False,
            api_latency_ms=(time.time() - t_start) * 1000,
            model_used=self.config.model,
            error_message=error_msg,
            timestamp=t_start,
        )

    # ═══════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════

    @property
    def is_available(self) -> bool:
        return self._openai_client is not None

    @property
    def is_api_key_set(self) -> bool:
        return bool(self.config.dashscope_api_key)


# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def check_environment() -> dict:
    """检查运行环境"""
    import os
    return {
        "openai_available": OPENAI_AVAILABLE,
        "api_key_set": bool(os.getenv("DASHSCOPE_API_KEY")),
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
