
import time
from collections import deque

import numpy as np
try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class WaveformWidget(QWidget):
    """
    实时音频波形显示
    - 显示最近 N 毫秒的音频波形
    - 用颜色区分有人声/静音段
    """

    # ═══════════════════════════════════════════════════════════
    # 组件初始化 — pyqtgraph 设置、曲线、坐标轴
    # ═══════════════════════════════════════════════════════════

    def __init__(self, sample_rate: int = 16000,
                 display_duration_ms: int = 3000,
                 parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.display_duration_ms = display_duration_ms

        # 数据缓冲
        buffer_size = int(sample_rate * display_duration_ms / 1000)
        self._buffer = deque(maxlen=buffer_size)
        self._timestamps = deque(maxlen=buffer_size)
        self._voice_flags = deque(maxlen=buffer_size // sample_rate)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title = QLabel("实时音频波形")
        title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        if PYQTGRAPH_AVAILABLE:
            # pyqtgraph 波形图
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setLabel("left", "幅度", **{"font-size": "11pt"})
            self._plot_widget.setLabel("bottom", "时间 (s)", **{"font-size": "11pt"})
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self._plot_widget.setYRange(-32768, 32767)
            self._plot_widget.setBackground("#0c1520")
            # 隐藏右侧 Y 轴刻度 (更简洁)
            self._plot_widget.showAxis("right", False)
            # 坐标轴颜色适配暗色主题
            self._plot_widget.getAxis("left").setPen(pg.mkPen(color="#6b7d95", width=1))
            self._plot_widget.getAxis("bottom").setPen(pg.mkPen(color="#6b7d95", width=1))
            self._plot_widget.getAxis("left").setTextPen(pg.mkPen(color="#a0b4d0"))
            self._plot_widget.getAxis("bottom").setTextPen(pg.mkPen(color="#a0b4d0"))

            # 曲线: 有人声时蓝色, 静音时灰色
            self._curve = self._plot_widget.plot(
                pen=pg.mkPen(color=(0, 100, 200), width=1.2)
            )
            # 静音段叠加曲线 (灰色)
            self._silence_curve = self._plot_widget.plot(
                pen=pg.mkPen(color=(180, 180, 180), width=0.8)
            )
            # 人声段叠加曲线 (蓝色高亮)
            self._voice_curve = self._plot_widget.plot(
                pen=pg.mkPen(color=(0, 120, 240), width=1.5)
            )

            layout.addWidget(self._plot_widget)
        else:
            # 降级：用 QLabel 显示文本信息
            self._fallback_label = QLabel("(请安装 pyqtgraph 查看波形)")
            self._fallback_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self._fallback_label)

    # ═══════════════════════════════════════════════════════════
    # 波形数据喂入 — 追加音频样本到环形缓冲区
    # ═══════════════════════════════════════════════════════════

    # ---- 数据输入 ----

    def feed(self, audio_data: np.ndarray, is_voice: bool = False,
             timestamp: float = None):
        """
        喂入音频数据
        Args:
            audio_data: PCM 采样点 (n_samples,) float32
            is_voice: VAD 结果
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        self._buffer.extend(audio_data.tolist())
        self._timestamps.append(timestamp)
        self._voice_flags.append(is_voice)

    # ═══════════════════════════════════════════════════════════
    # VAD 着色渲染 — 蓝色=语音 / 灰色=静音，自适应缩放
    # Auto-scaling time axis, segmented overlays
    # ═══════════════════════════════════════════════════════════

    def update_display(self):
        """刷新波形显示（由 QTimer 调用）"""
        if not PYQTGRAPH_AVAILABLE or len(self._buffer) == 0:
            return

        data = np.array(list(self._buffer), dtype=np.float32)
        n = len(data)
        if n == 0:
            return

        # 时间轴 (秒, 相对最新)
        t = np.linspace(-n / self.sample_rate, 0, n)

        # ★ 按人声/静音分段着色
        # 构建颜色掩码: 有人声=蓝色, 静音=灰色
        if hasattr(self, '_voice_curve') and len(self._voice_flags) > 0:
            # 将 voice_flags (每帧一次) 扩展到每个采样点
            samples_per_flag = max(1, n // max(1, len(self._voice_flags)))
            voice_mask = np.zeros(n, dtype=bool)
            for i, is_voice in enumerate(self._voice_flags):
                start = i * samples_per_flag
                end = start + samples_per_flag
                if is_voice:
                    voice_mask[start:end] = True

            # 有人声部分
            voice_data = data.copy()
            voice_data[~voice_mask] = np.nan
            # 静音部分
            silence_data = data.copy()
            silence_data[voice_mask] = np.nan

            self._voice_curve.setData(t, voice_data)
            self._silence_curve.setData(t, silence_data)
            self._curve.clear()  # 隐藏默认曲线
        else:
            # 降级: 无语音标志时用默认蓝色
            self._curve.setData(t, data)

    # ═══════════════════════════════════════════════════════════
    # 清理 — 重置所有缓冲区和曲线
    # ═══════════════════════════════════════════════════════════

    def clear(self):
        """清空波形"""
        self._buffer.clear()
        self._timestamps.clear()
        self._voice_flags.clear()
        if PYQTGRAPH_AVAILABLE:
            self._curve.clear()
