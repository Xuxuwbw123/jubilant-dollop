"""
消息协议定义
定义前端-后端之间的 WebSocket 通信消息格式

消息统一为 JSON 文本帧，音频数据用 base64 编码
"""

import json
import base64
import time
from enum import Enum
from typing import Optional, Any
import numpy as np

from .audio_config import AudioFrame


# ═══════════════════════════════════════════════════════════
# 消息类型枚举 — WebSocket 协议消息类型定义
# ═══════════════════════════════════════════════════════════

class MessageType(str, Enum):
    """消息类型枚举"""
    AUDIO_FRAME = "audio_frame"         # 音频数据帧 (前端→后端)
    HEARTBEAT = "heartbeat"             # 心跳 (双向)
    CONTROL_START = "control.start"     # 开始采集 (后端→前端)
    CONTROL_STOP = "control.stop"       # 停止采集 (后端→前端)
    CONTROL_ACK = "control.ack"         # 控制确认 (前端→后端)
    CLASSIFY_RESULT = "classify_result" # 分类结果 (后端内部)
    ALERT = "alert"                     # 报警通知 (后端内部)


# ═══════════════════════════════════════════════════════════
# 音频帧编解码 — numpy 数组 ↔ base64 JSON 序列化
# ═══════════════════════════════════════════════════════════

def encode_audio_frame(frame: AudioFrame) -> str:
    """
    将 AudioFrame 编码为 JSON 字符串
    - 音频数据: numpy → bytes → base64
    """
    # numpy → bytes (16bit PCM little-endian)
    audio_bytes = frame.data.astype(np.int16).tobytes()
    # base64 编码
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    message = {
        "type": MessageType.AUDIO_FRAME,
        "timestamp": frame.timestamp or time.time(),
        "sample_rate": frame.sample_rate,
        "frame_index": frame.frame_index,
        "is_voice": bool(frame.is_voice),  # 确保是 Python bool，不是 numpy.bool_
        "data": audio_b64,
    }
    return json.dumps(message, ensure_ascii=False)


def decode_audio_frame(json_str: str) -> AudioFrame:
    """
    从 JSON 字符串解码为 AudioFrame
    """
    msg = json.loads(json_str)
    # base64 → bytes → numpy
    audio_bytes = base64.b64decode(msg["data"])
    data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    return AudioFrame(
        data=data,
        sample_rate=msg.get("sample_rate", 16000),
        timestamp=msg.get("timestamp", 0.0),
        frame_index=msg.get("frame_index", 0),
        is_voice=msg.get("is_voice", False),
    )


# ═══════════════════════════════════════════════════════════
# 心跳编码 — 每 5 秒保活 Ping
# ═══════════════════════════════════════════════════════════

def encode_heartbeat() -> str:
    """编码心跳消息"""
    return json.dumps({
        "type": MessageType.HEARTBEAT,
        "timestamp": time.time(),
    })


# ═══════════════════════════════════════════════════════════
# 控制消息编码 — 后端下发启停指令
# ═══════════════════════════════════════════════════════════

def encode_control(command: str) -> str:
    """编码控制消息 (start / stop)"""
    msg_type = (MessageType.CONTROL_START if command == "start"
                else MessageType.CONTROL_STOP)
    return json.dumps({
        "type": msg_type,
        "timestamp": time.time(),
    })


def encode_control_ack() -> str:
    """编码控制确认消息"""
    return json.dumps({
        "type": MessageType.CONTROL_ACK,
        "timestamp": time.time(),
    })


# ═══════════════════════════════════════════════════════════
# 消息解析工具 — 轻量 type/timestamp 提取
# ═══════════════════════════════════════════════════════════

def parse_message_type(json_str: str) -> Optional[str]:
    """解析消息类型（不完整解析，仅获取 type 字段）"""
    try:
        msg = json.loads(json_str)
        return msg.get("type")
    except (json.JSONDecodeError, KeyError):
        return None


def parse_timestamp(json_str: str) -> float:
    """从消息中提取时间戳"""
    try:
        msg = json.loads(json_str)
        return msg.get("timestamp", 0.0)
    except json.JSONDecodeError:
        return 0.0
