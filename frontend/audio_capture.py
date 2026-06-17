

import time
import logging
import threading
from typing import Optional, Generator

import numpy as np

from common.audio_config import AudioFrame, AudioConfig, VADConfig

logger = logging.getLogger("frontend.capture")


# ============================================================
# 成员A：AudioCapture — 麦克风录音 + VAD
# ============================================================

class AudioCapture:
    """
    实时音频采集器
    - 封装 PyAudio/sounddevice
    - 按帧输出 AudioFrame（含 VAD 结果）
    - 支持模拟模式（随机数据）
    """

    def __init__(self, audio_config: AudioConfig,
                 vad_config: VADConfig = None,
                 simulate: bool = False,
                 input_device_index: int = None):
        """
        Args:
            audio_config: 音频参数 (采样率、帧长等)
            vad_config: VAD 参数
            simulate: 模拟模式（不实际使用麦克风）
            input_device_index: 麦克风设备索引 (None=默认)
        """
        self.config = audio_config
        self.vad_config = vad_config or VADConfig()
        self.simulate = simulate
        self.input_device_index = input_device_index

        self._stream = None
        self._pyaudio = None
        self._vad = None
        self._running = False
        self._frame_index = 0
        self._lock = threading.Lock()

        if not simulate:
            self._init_pyaudio()
            if self.vad_config.enabled:
                self._init_vad()

    # ═══════════════════════════════════════════════════════════
    # PyAudio 初始化 — 打开麦克风流进行实时采集
    # ═══════════════════════════════════════════════════════════

    def _init_pyaudio(self):
        """初始化 PyAudio"""
        try:
            import pyaudio
            self._pa = pyaudio  # 保存模块引用，用于访问常量
            self._pyaudio = pyaudio.PyAudio()
            logger.info("PyAudio 初始化成功")
        except ImportError:
            logger.warning("PyAudio 未安装，回退到模拟模式")
            self.simulate = True
        except Exception as e:
            logger.warning(f"PyAudio 初始化失败: {e}，回退到模拟模式")
            self.simulate = True

    # ═══════════════════════════════════════════════════════════
    # VAD 初始化 — WebRTC 语音活动检测设置
    # ═══════════════════════════════════════════════════════════

    def _init_vad(self):
        """初始化 WebRTC VAD"""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.vad_config.mode)
            logger.info(f"WebRTC VAD 初始化成功 (mode={self.vad_config.mode})")
        except ImportError:
            logger.warning("webrtcvad 未安装，VAD 将被禁用")
            self.vad_config.enabled = False
        except Exception as e:
            logger.warning(f"VAD 初始化失败: {e}")
            self.vad_config.enabled = False

    # ═══════════════════════════════════════════════════════════
    # 启停控制 — 开始与结束音频采集会话
    # ═══════════════════════════════════════════════════════════

    def start(self) -> bool:
        """开始录音"""
        if self._running:
            logger.warning("已经在录音中")
            return False

        self._running = True
        self._frame_index = 0

        if self.simulate:
            logger.info("模拟模式: 开始生成模拟音频")
            return True

        try:
            self._stream = self._pyaudio.open(
                format=self._pa.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.config.frame_size,
                stream_callback=None,  # 使用阻塞模式，简单可靠
            )
            logger.info(f"开始录音: sr={self.config.sample_rate}, "
                        f"frame_size={self.config.frame_size}")
            return True
        except Exception as e:
            logger.error(f"无法打开麦克风: {e}")
            self._running = False
            return False

    def stop(self):
        """停止录音"""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
            logger.info("录音已停止")

    @property
    def is_running(self) -> bool:
        return self._running

    # ═══════════════════════════════════════════════════════════
    # 帧采集（阻塞模式）— 从麦克风读取 30ms 音频块
    # ═══════════════════════════════════════════════════════════

    def get_frame(self) -> Optional[AudioFrame]:
        """
        获取下一帧音频（阻塞模式）
        Returns:
            AudioFrame 或 None（停止时）
        """
        if not self._running:
            return None

        timestamp = time.time()

        if self.simulate:
            data = self._generate_simulated_audio()
            is_voice = self._simulated_vad(data)
        else:
            try:
                raw_data = self._stream.read(
                    self.config.frame_size,
                    exception_on_overflow=False,
                )
                data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32767.0
            except Exception as e:
                logger.error(f"读取音频帧失败: {e}")
                return None

            # VAD 检测
            if self._vad and self.vad_config.enabled:
                # webrtcvad 需要 16bit PCM bytes
                pcm_bytes = data.astype(np.int16).tobytes()
                try:
                    is_voice = self._vad.is_speech(pcm_bytes, self.config.sample_rate)
                except Exception:
                    is_voice = True  # VAD 失败时视作有声音
            else:
                is_voice = True

        frame = AudioFrame(
            data=data,
            sample_rate=self.config.sample_rate,
            timestamp=timestamp,
            frame_index=self._frame_index,
            is_voice=is_voice,
        )

        with self._lock:
            self._frame_index += 1

        return frame

    def iter_frames(self) -> Generator[AudioFrame, None, None]:
        """
        迭代器模式获取音频帧
        Yields AudioFrame 直到停止
        """
        while self._running:
            frame = self.get_frame()
            if frame is None:
                break
            yield frame

    # ═══════════════════════════════════════════════════════════
    # 模拟音频生成 — 正弦波+噪声，无需硬件即可调试
    # ═══════════════════════════════════════════════════════════

    def _generate_simulated_audio(self) -> np.ndarray:
        """生成模拟音频数据（用于无麦克风调试）"""
        # 随机噪声 + 模拟正弦波
        t = np.linspace(0, self.config.frame_duration_ms / 1000,
                        self.config.frame_size, endpoint=False)
        freq = np.random.uniform(100, 2000)  # 随机频率
        amplitude = np.random.uniform(500, 5000)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 500, self.config.frame_size)
        return (signal + noise).astype(np.float32)

    def _simulated_vad(self, data: np.ndarray) -> bool:
        """模拟 VAD: 能量 > 阈值即认为有声音"""
        energy = np.sqrt(np.mean(data ** 2))
        return energy > 800  # 阈值

    # ═══════════════════════════════════════════════════════════
    # 设备枚举 — 列出可用的音频输入设备
    # ═══════════════════════════════════════════════════════════

    @classmethod
    def list_devices(cls) -> list:
        """列出可用音频输入设备"""
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "default_sample_rate": int(info["defaultSampleRate"]),
                    })
            pa.terminate()
            return devices
        except Exception:
            return []

    def __del__(self):
        self.stop()
        if self._pyaudio:
            self._pyaudio.terminate()
