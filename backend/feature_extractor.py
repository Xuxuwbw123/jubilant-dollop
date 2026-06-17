
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger("backend.feature_extractor")


class FeatureExtractor:
    """
    音频特征提取器
    从原始 PCM 音频中提取 MFCC 和梅尔频谱图特征
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 simulate: bool = False):
        """
        Args:
            sample_rate: 采样率
            n_mfcc: MFCC 系数个数
            n_mels: 梅尔滤波器组数量
            n_fft: FFT 窗口大小
            hop_length: 帧移
            simulate: 模拟模式（返回随机特征）
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.simulate = simulate

        self._librosa_available = False
        if not simulate:
            try:
                import librosa
                self._librosa_available = True
                logger.info("librosa 特征提取已就绪")
            except ImportError:
                logger.warning("librosa 未安装，将使用模拟特征")
                self.simulate = True

    # ═══════════════════════════════════════════════════════════
    # 3 通道 Mel 频谱提取 — 静态 + Δ 一阶差分 + ΔΔ 二阶差分（核心创新）
    # Channel 0: static Mel spectrogram (instantaneous timbre)
    # Channel 1: delta (1st-order difference, spectral velocity)
    # Channel 2: delta-delta (2nd-order difference, spectral acceleration)
    # Preserves full time dimension for CNN to learn temporal patterns:
    #   scream → short-duration high-frequency spikes
    #   cry    → sustained tremolo-like pitch wavering
    #   laugh  → rhythmic burst-decay patterns
    # ═══════════════════════════════════════════════════════════

    def extract_3channel_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 3 通道梅尔频谱图 (2D 特征，保留时间维度)

        改进 v2: 不再对时间轴取均值，输出完整 2D 频谱图
        - Channel 0: Mel 频谱 (静态音色)
        - Channel 1: Δ 一阶差分 (频率变化速度)
        - Channel 2: ΔΔ 二阶差分 (频率变化加速度)

        保留时间维度后，模型可以学到:
        - 尖叫: 短促的高频尖峰模式
        - 大哭: 持续的颤音波动模式
        - 大笑: 有节奏的爆发-衰减模式

        Args:
            audio: 原始音频 (n_samples,) float32
        Returns:
            features: (3, n_mels, n_frames) float32
        """
        if self.simulate or not self._librosa_available:
            n_frames = max(8, len(audio) // self.hop_length)
            return np.random.randn(3, self.n_mels, n_frames).astype(np.float32)

        import librosa

        # 确保音频足够长
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        # 梅尔频谱图 (保留时间维度)
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float64),
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )  # (n_mels, n_frames)

        # 转 dB
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, n_frames)

        # 一阶差分: 频率变化速度
        delta = librosa.feature.delta(mel_db)           # (n_mels, n_frames)

        # 二阶差分: 频率变化加速度
        delta2 = librosa.feature.delta(mel_db, order=2) # (n_mels, n_frames)

        # 堆叠为 3 通道 (如同 RGB 图像)
        features = np.stack([mel_db, delta, delta2], axis=0)  # (3, n_mels, n_frames)

        return features.astype(np.float32)

    # ═══════════════════════════════════════════════════════════
    # Mel 频谱提取 — 单通道基线方法
    # Produces a 2D (n_mels, n_frames) log-power spectrogram.
    # Used for visualization and as a fallback / reference feature.
    # ═══════════════════════════════════════════════════════════

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        提取梅尔频谱图
        Args:
            audio: 原始音频 (n_samples,) float32
        Returns:
            mel_spec: (n_mels, n_frames)
        """
        if self.simulate or not self._librosa_available:
            n_frames = max(1, len(audio) // self.hop_length)
            return np.random.randn(self.n_mels, n_frames).astype(np.float32)

        import librosa

        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float64),
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # 转 dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.astype(np.float32)

    # ═══════════════════════════════════════════════════════════
    # 特征提取分发 — 一次性返回所有特征变体
    # Single call site for the pipeline; returns a dict with both the
    # 3-channel mel (for CNN input) and the single-channel mel (for viz).
    # ═══════════════════════════════════════════════════════════

    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取所有特征
        Returns:
            {"mfcc": (n_mfcc,), "mel_spectrogram": (n_mels, n_frames)}
        """
        return {
            "3channel_mel": self.extract_3channel_mel(audio),
            "mel_spectrogram": self.extract_mel_spectrogram(audio),
        }

    def get_feature_dim(self) -> dict:
        """返回2D特征维度信息: channels, n_mels (时间维度可变)"""
        return {"channels": 3, "n_mels": self.n_mels}

    # ═══════════════════════════════════════════════════════════
    # 维纳熵语音检测 — 频谱平坦度 VAD 前置过滤
    # Distinguishes human vocalisations from silence / white noise.
    # - White noise / silence: all mel bands have similar energy →
    #   high spectral flatness (Wiener entropy near 1.0) → not speech
    # - Human voice: energy concentrates in formant bands →
    #   low spectral flatness → likely speech
    # Fast amplitude check first: peak < 5e-4 → immediate reject.
    # ═══════════════════════════════════════════════════════════

    def is_likely_speech(self, audio: np.ndarray) -> bool:
        """
        检测音频是否像是人声 (vs 静音/噪声)

        指标: mel 频谱带间方差
        - 白噪声/静音: 所有频带能量接近 → 低方差 → 不是人声
        - 人声: 能量集中在特定频带(共振峰) → 高方差 → 像人声

        Returns:
            True = 像人声，应该送入模型
            False = 静音或平坦噪声，直接判 normal
        """
        # ---- 全零/静音快速判断 ----
        peak = float(np.abs(audio).max())
        if peak < 5e-4:
            return False

        # ---- 频谱方差检测 ----
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float64),
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )  # (n_mels, n_frames)

            # 每个 mel 频带在时间上的平均能量
            mel_mean = mel.mean(axis=1)  # (n_mels,)
            # 归一化为概率分布
            total = mel_mean.sum()
            if total < 1e-9:
                return False
            mel_norm = mel_mean / total  # 和为 1

            # 计算频谱平坦度 (Wiener entropy)
            # flatness = geometric_mean / arithmetic_mean
            # 白噪声: 所有频带 ≈ 1/n_mels → flatness ≈ 1.0
            # 人声: 某些频带主导 → flatness << 1.0
            n_mels = len(mel_norm)
            geo_mean = np.exp(np.mean(np.log(mel_norm + 1e-9)))
            ari_mean = 1.0 / n_mels  # 归一化后均匀分布的算术平均
            flatness = geo_mean / ari_mean  # 0~1

            # 阈值: flatness > 0.65 → 频谱太平坦，不像是人声
            if flatness > 0.65:
                return False

        except Exception:
            logger.warning("频谱平坦度计算失败，默认视为有人声", exc_info=True)

        return True
