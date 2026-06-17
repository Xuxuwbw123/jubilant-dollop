"""
音频参数常量定义
前后端共用，确保参数一致
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ═══════════════════════════════════════════════════════════
# 音频配置 — 16kHz · 单声道 · 16bit · 30ms 帧参数定义
# ═══════════════════════════════════════════════════════════

@dataclass
class AudioConfig:
    """音频采集/处理参数"""
    sample_rate: int = 16000          # 采样率 (Hz)
    channels: int = 1                  # 声道数
    sample_width: int = 2              # 位深 (bytes): 2=16bit
    frame_duration_ms: int = 30        # 每帧时长 (ms)

    @property
    def frame_size(self) -> int:
        """每帧采样点数 = sample_rate * frame_duration / 1000"""
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def bytes_per_frame(self) -> int:
        """每帧字节数"""
        return self.frame_size * self.channels * self.sample_width

    def validate(self) -> bool:
        """校验参数合法性"""
        if self.sample_rate not in (8000, 16000, 22050, 44100, 48000):
            return False
        if self.channels < 1:
            return False
        if self.sample_width not in (1, 2, 4):
            return False
        if self.frame_duration_ms not in (10, 20, 30, 40):
            # webrtcvad 要求 10/20/30ms 帧长
            return False
        return True

    @classmethod
    def from_yaml(cls, cfg: dict) -> "AudioConfig":
        """从 YAML 配置字典创建"""
        audio_cfg = cfg.get("audio", {})
        return cls(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            sample_width=audio_cfg.get("sample_width", 2),
            frame_duration_ms=audio_cfg.get("frame_duration", 30),
        )


# ═══════════════════════════════════════════════════════════
# 音频帧 — 前后端之间传递的基本音频数据单元
# ═══════════════════════════════════════════════════════════

@dataclass
class AudioFrame:
    """音频帧 - 前后端之间传递的音频数据单元"""
    data: np.ndarray              # 原始PCM音频数据 (frame_size,)
    sample_rate: int = 16000
    timestamp: float = 0.0        # 采集时间戳
    frame_index: int = 0          # 帧序号 (递增)
    is_voice: bool = False        # VAD 检测结果 (True=有人声)

    @property
    def duration_ms(self) -> float:
        """帧时长 (ms)"""
        return len(self.data) / self.sample_rate * 1000

    def __repr__(self) -> str:
        return (f"AudioFrame(idx={self.frame_index}, "
                f"len={len(self.data)}, sr={self.sample_rate}, "
                f"voice={self.is_voice})")


# ═══════════════════════════════════════════════════════════
# VAD 配置 — WebRTC 语音活动检测参数
# ═══════════════════════════════════════════════════════════

@dataclass
class VADConfig:
    """VAD 静音检测配置"""
    mode: int = 1                   # 0=安静, 1=适中, 2=低灵敏度, 3=很激进
    enabled: bool = True

    @classmethod
    def from_yaml(cls, cfg: dict) -> "VADConfig":
        vad_cfg = cfg.get("vad", {})
        return cls(
            mode=vad_cfg.get("mode", 1),
            enabled=vad_cfg.get("enabled", True),
        )


# ═══════════════════════════════════════════════════════════
# 分类结果 — AI 模型推理输出（情绪类别 + 置信度）
# ═══════════════════════════════════════════════════════════

@dataclass
class ClassifyResult:
    """AI 分类结果"""
    class_name: str = "normal"      # 类别名
    confidence: float = 0.0         # 置信度 [0, 1]
    is_abnormal: bool = False       # 是否异常
    timestamp: float = 0.0          # 分类时间戳
    frame_index: int = 0            # 对应的帧序号
    all_probs: dict = None          # 所有类别的概率 (可选)

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "is_abnormal": self.is_abnormal,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
        }

    def __repr__(self) -> str:
        return (f"ClassifyResult({self.class_name}, "
                f"conf={self.confidence:.3f}, abnormal={self.is_abnormal})")


# ═══════════════════════════════════════════════════════════
# 报警记录 — 含严重度分级与中文显示
# ═══════════════════════════════════════════════════════════

@dataclass
class Alert:
    """报警记录"""
    level: str                      # "warning" | "critical"
    message: str                    # 报警描述
    class_name: str = ""            # 触发的异常类别
    timestamp: float = 0.0
    frame_index: int = 0
    severity: float = 0.0           # 严重度评分 [0, 1]

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "message": self.message,
            "class_name": self.class_name,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "severity": round(self.severity, 4),
        }

    def __repr__(self) -> str:
        return f"Alert[{self.level}] {self.message}"


# ═══════════════════════════════════════════════════════════
# 报警配置 — 滑动窗口 + 三级阈值报警规则
# ═══════════════════════════════════════════════════════════

@dataclass
class AlertConfig:
    """报警规则配置（滑动窗口 + 严重度评分 + 多级报警）"""
    # 滑动窗口
    window_size: int = 15           # 分析窗口大小（帧数）
    min_abnormal_ratio: float = 0.4  # 窗口内异常比例下限（低于此值不报警）

    # 阈值
    confidence_threshold: float = 0.60  # 置信度下限（低于此值视为无效，需 EMA 平滑后达标）
    pre_alert_threshold: float = 0.35   # 预报警严重度阈值
    warning_threshold: float = 0.55     # 警告严重度阈值
    critical_threshold: float = 0.75    # 危急严重度阈值

    # 冷却
    cooldown_seconds: float = 8.0       # 报警冷却时间（拉长避免重复报警）
    critical_cooldown_seconds: float = 3.0  # 危急级别冷却更短

    # 各类别的严重度权重（自己可调）
    class_severity: dict = field(default_factory=lambda: {
        "normal": 0.0,
        "laugh": 0.5,    # 大笑 — 轻度异常
        "cry": 0.8,      # 大哭 — 较重异常
        "scream": 1.0,   # 尖叫 — 最严重
    })

    # 保留旧字段以兼容
    consecutive_frames: int = 3

    @classmethod
    def from_yaml(cls, cfg: dict) -> "AlertConfig":
        alert_cfg = cfg.get("alert", {})
        return cls(
            window_size=alert_cfg.get("window_size", 15),
            min_abnormal_ratio=alert_cfg.get("min_abnormal_ratio", 0.4),
            confidence_threshold=alert_cfg.get("confidence_threshold", 0.55),
            pre_alert_threshold=alert_cfg.get("pre_alert_threshold", 0.30),
            warning_threshold=alert_cfg.get("warning_threshold", 0.50),
            critical_threshold=alert_cfg.get("critical_threshold", 0.75),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 5.0),
            critical_cooldown_seconds=alert_cfg.get("critical_cooldown_seconds", 2.0),
            class_severity=alert_cfg.get("class_severity", {
                "normal": 0.0, "laugh": 0.5, "cry": 0.8, "scream": 1.0
            }),
            consecutive_frames=alert_cfg.get("consecutive_frames", 3),
        )
