
import time
import logging
from collections import deque
from typing import Optional

import numpy as np
try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFrame, QSplitter,
        QStatusBar, QGroupBox, QGridLayout,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QColor, QPalette
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

from common.audio_config import AudioFrame, ClassifyResult, Alert
from asr_emotion_agent.qwen_omni_client import EmotionResult
from backend.gui_waveform import WaveformWidget
from backend.gui_history import HistoryWidget

logger = logging.getLogger("backend.gui")


# ═══════════════════════════════════════════════════════════
# 窗口初始化 — 暗色主题布局、UI 构建、样式设置
# ═══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    后端主 GUI 窗口
    数据流: AudioFrame → 特征提取 → AI分类 → 报警判断 → 界面更新
    """

    # 信号: 从 WebSocket 线程安全地更新 UI
    signal_frame = pyqtSignal(object)        # AudioFrame
    signal_result = pyqtSignal(object)       # ClassifyResult
    signal_alert = pyqtSignal(object)        # Alert
    signal_omni_result = pyqtSignal(object)  # EmotionResult
    signal_source = pyqtSignal(str, str)     # (来源文字, 颜色)
    signal_status = pyqtSignal(str)          # 状态文本
    signal_connection = pyqtSignal(bool)     # 连接状态

    def __init__(self, window_title: str = "双机实时音频监测系统"):
        super().__init__()

        self.setWindowTitle(window_title)
        self.setMinimumSize(1200, 750)
        self.resize(1400, 900)

        # 统计
        self._frame_count = 0
        self._frame_count_abnormal = 0
        self._start_time = time.time()
        self._recent_fps: deque = deque(maxlen=50)  # 最近50帧的间隔用于计算FPS

        # 当前状态
        self._current_result: Optional[ClassifyResult] = None
        self._connected = False
        self._source_is_omni = False

        self._init_ui()
        self._apply_dark_theme()
        self._connect_signals()

    def _init_ui(self):
        """构建 UI 布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---- 左侧: 波形 ----
        self._waveform = WaveformWidget(sample_rate=16000, display_duration_ms=5000)
        self._waveform.setMinimumHeight(300)
        waveform_group = QGroupBox("音频波形")
        waveform_layout = QVBoxLayout(waveform_group)
        waveform_layout.addWidget(self._waveform)

        # ---- 右侧面板 ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 分类结果
        result_group = QGroupBox("当前分类结果")
        result_layout = QGridLayout(result_group)

        self._label_class = QLabel("--")
        self._label_class.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        self._label_class.setAlignment(Qt.AlignCenter)

        self._label_confidence = QLabel("置信度: --")
        self._label_confidence.setFont(QFont("Microsoft YaHei", 12))
        self._label_confidence.setAlignment(Qt.AlignCenter)

        self._label_source = QLabel("来源: --")
        self._label_source.setFont(QFont("Microsoft YaHei", 8))
        self._label_source.setAlignment(Qt.AlignCenter)
        self._label_source.setStyleSheet("color: #95a5a6;")

        result_layout.addWidget(self._label_class, 0, 0)
        result_layout.addWidget(self._label_confidence, 1, 0)
        result_layout.addWidget(self._label_source, 2, 0)
        right_layout.addWidget(result_group)

        # 报警状态指示灯
        alert_group = QGroupBox("报警状态")
        alert_layout = QVBoxLayout(alert_group)

        self._alert_indicator = QLabel("● 正常")
        self._alert_indicator.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self._alert_indicator.setAlignment(Qt.AlignCenter)
        self._alert_indicator.setStyleSheet(
            "color: #2ecc71; padding: 8px;"
            "border: 2px solid #2ecc71; border-radius: 8px;"
        )
        self._alert_indicator.setMinimumHeight(50)
        alert_layout.addWidget(self._alert_indicator)

        self._label_last_alert = QLabel("最近报警: 无")
        self._label_last_alert.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(self._label_last_alert)
        right_layout.addWidget(alert_group)

        # Omni 智能体状态
        omni_group = QGroupBox("Omni 智能体 (云端)")
        omni_layout = QGridLayout(omni_group)

        self._omni_status = QLabel("等待初始化...")
        self._omni_status.setFont(QFont("Microsoft YaHei", 9))
        self._omni_status.setAlignment(Qt.AlignCenter)
        self._omni_status.setStyleSheet("color: #95a5a6; padding: 4px;")

        self._label_omni_emotion = QLabel("情绪: --")
        self._label_omni_emotion.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self._label_omni_danger = QLabel("等级: --")
        self._label_omni_danger.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self._label_omni_text = QLabel("转写: --")
        self._label_omni_text.setFont(QFont("Microsoft YaHei", 9))
        self._label_omni_text.setWordWrap(True)
        self._label_omni_reason = QLabel("理由: --")
        self._label_omni_reason.setFont(QFont("Microsoft YaHei", 9))
        self._label_omni_reason.setWordWrap(True)
        self._label_omni_latency = QLabel("延迟: --")
        self._label_omni_latency.setFont(QFont("Microsoft YaHei", 9))
        self._label_omni_online = QLabel("在线: --")
        self._label_omni_online.setFont(QFont("Microsoft YaHei", 9))

        omni_layout.addWidget(self._omni_status, 0, 0, 1, 2)
        omni_layout.addWidget(self._label_omni_emotion, 1, 0)
        omni_layout.addWidget(self._label_omni_danger, 1, 1)
        omni_layout.addWidget(self._label_omni_text, 2, 0, 1, 2)
        omni_layout.addWidget(self._label_omni_reason, 3, 0, 1, 2)
        omni_layout.addWidget(self._label_omni_latency, 4, 0)
        omni_layout.addWidget(self._label_omni_online, 4, 1)
        right_layout.addWidget(omni_group)

        # 统计信息
        stats_group = QGroupBox("运行统计")
        stats_layout = QGridLayout(stats_group)
        self._label_fps = QLabel("帧率: --")
        self._label_frames = QLabel("总帧数: 0")
        self._label_abnormal_rate = QLabel("异常率: --%")
        stats_layout.addWidget(self._label_fps, 0, 0)
        stats_layout.addWidget(self._label_frames, 0, 1)
        stats_layout.addWidget(self._label_abnormal_rate, 1, 0, 1, 2)
        right_layout.addWidget(stats_group)

        # 报警历史
        self._history = HistoryWidget()
        right_layout.addWidget(self._history, stretch=1)

        # ---- 使用 QSplitter 实现左右可拖拽调整 ----
        splitter = QSplitter(Qt.Horizontal)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(waveform_group)
        splitter.addWidget(left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)  # 左侧 71%
        splitter.setStretchFactor(1, 2)  # 右侧 29%

        main_layout.addWidget(splitter)

        # ---- 状态栏 ----
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        self._status_connection = QLabel("连接状态: 等待前端连接...")
        self._status_connection.setStyleSheet("color: #e67e22; padding: 2px 8px;")
        self._statusbar.addPermanentWidget(self._status_connection)

        # ---- 刷新定时器 ----
        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._on_refresh)
        self._refresh_timer.start(50)  # 20 FPS UI 刷新

    def _apply_dark_theme(self):
        """应用暗色主题样式"""
        if not PYQT5_AVAILABLE:
            return
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
                font-family: "Microsoft YaHei", sans-serif;
            }
            QGroupBox {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 16px;
                font-weight: bold;
                color: #a0b4d0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #7ec8e3;
            }
            QLabel {
                background-color: transparent;
            }
            QStatusBar {
                background-color: #0f3460;
                color: #a0b4d0;
            }
            QTableWidget {
                background-color: #16213e;
                alternate-background-color: #1a2744;
                gridline-color: #0f3460;
                border: 1px solid #0f3460;
                border-radius: 4px;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QHeaderView::section {
                background-color: #0f3460;
                color: #7ec8e3;
                border: none;
                padding: 4px 8px;
            }
            QSplitter::handle {
                background-color: #0f3460;
                width: 2px;
            }
        """)


    # ═══════════════════════════════════════════════════════════════════
    # Qt 信号槽连接 — 线程安全 UI 更新（队列连接）
    # ═══════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        """连接信号到槽"""
        self.signal_frame.connect(self._on_frame)
        self.signal_result.connect(self._on_result)
        self.signal_alert.connect(self._on_alert)
        self.signal_omni_result.connect(self._on_omni_result)
        self.signal_source.connect(self._on_source)
        self.signal_status.connect(self._on_status)
        self.signal_connection.connect(self._on_connection_change)

    # ═══════════════════════════════════════════════════════════
    # 数据喂入处理 — 从工作线程接收数据（线程安全）
    # ═══════════════════════════════════════════════════════════

    # ---- 输入 (从 WebSocket → 特征提取 → 分类 线程调用) ----

    def feed_audio_frame(self, frame: AudioFrame):
        """接收音频帧（线程安全）"""
        self.signal_frame.emit(frame)

    def feed_classify_result(self, result: ClassifyResult):
        """接收分类结果（线程安全）"""
        self.signal_result.emit(result)

    def feed_alert(self, alert: Alert):
        """接收报警（线程安全）"""
        self.signal_alert.emit(alert)

    def feed_emotion_result(self, result: EmotionResult):
        """接收 Omni 情绪分析结果（线程安全）"""
        self.signal_omni_result.emit(result)

    def feed_file_audio(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        将文件音频喂入波形组件（用于文件上传分析）。
        分~10个帧推送，避免信号风暴。
        自动缩放到 int16 量程以匹配波形 Y 轴。
        """
        if audio is None or len(audio) == 0:
            return
        # 缩放到 int16 量程 (波形 Y 轴 -32768~32767)
        peak = np.abs(audio).max()
        if peak > 0 and peak < 100:  # float32 [-1,1] 归一化范围
            audio = (audio * 32767).astype(np.float32)
        n_total = len(audio)
        # 每帧约 0.3 秒，最多 15 帧
        frame_size = int(sample_rate * 0.3)
        num_frames = min(15, max(1, n_total // frame_size))
        chunk_size = max(frame_size, n_total // num_frames)
        for i in range(0, n_total, chunk_size):
            chunk = audio[i:i + chunk_size].astype(np.float32)
            rms_val = float(np.sqrt(np.mean(chunk ** 2)))
            frame = AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                frame_index=i // chunk_size,
                is_voice=rms_val > 0.003,
            )
            self.signal_frame.emit(frame)

    def feed_source(self, text: str, color: str = "#95a5a6"):
        """更新判断来源指示（线程安全）"""
        self.signal_source.emit(text, color)

    def set_omni_status(self, text: str, color: str = "#95a5a6"):
        """更新 Omni 在线状态文本（线程安全，通过 signal_status 复用）"""
        self.signal_status.emit(f"__omni__{color}__{text}")

    def set_connection_status(self, connected: bool):
        """设置连接状态（线程安全）"""
        self.signal_connection.emit(connected)

    # ═══════════════════════════════════════════════════════════
    # 槽函数处理 — 保证在 Qt 主线程执行 UI 更新
    # ═══════════════════════════════════════════════════════════

    # ---- 槽: UI 更新 (主线程) ----

    def _on_frame(self, frame: AudioFrame):
        """收到音频帧 → 更新波形"""
        self._frame_count += 1

        # 更新 FPS 计算
        now = time.time()
        self._recent_fps.append(now)

        # 更新波形
        self._waveform.feed(frame.data, frame.is_voice, frame.timestamp)

    def _on_result(self, result: ClassifyResult):
        """收到分类结果 → 更新显示"""
        self._current_result = result
        if result.is_abnormal:
            self._frame_count_abnormal += 1

        # 标记来源 (CNN 本地) — 始终更新来源标签
        self._label_source.setText("来源: CNN 本地 ⚡")
        self._label_source.setStyleSheet("color: #7f8c8d;")
        self._source_is_omni = False

        # 类别文字与颜色 (5类非言语 + 9类情绪)
        class_name_cn = {
            "normal": "正常", "scream": "尖叫", "cry": "大哭",
            "laugh": "大笑",
            # 9 类情绪 Fallback
            "angry": "愤怒", "fearful": "恐惧", "sad": "悲伤",
            "happy": "开心", "neutral": "中性", "surprised": "惊讶",
            "other": "其他", "unk": "未知",
        }
        display_name = class_name_cn.get(result.class_name, result.class_name)

        self._label_class.setText(display_name)
        conf_pct = result.confidence * 100
        self._label_confidence.setText(f"置信度: {conf_pct:.1f}%")

        # 颜色 & 报警指示灯联动
        if result.is_abnormal:
            self._label_class.setStyleSheet("color: #e74c3c;")
            # 指示灯同步到异常状态
            self._alert_indicator.setText(f"⚠ 异常: {display_name}")
            self._alert_indicator.setStyleSheet(
                "color: #e67e22; padding: 8px;"
                "border: 2px solid #e67e22; border-radius: 8px;"
                "background-color: #fef5e7;"
            )
        else:
            self._label_class.setStyleSheet("color: #2ecc71;")
            self._reset_alert_indicator()

    def _on_source(self, text: str, color: str):
        """更新判断来源标签"""
        self._label_source.setText(text)
        self._label_source.setStyleSheet(f"color: {color};")

    def _on_alert(self, alert: Alert):
        """收到报警 → 更新指示灯（不再自动重置，由 _on_result 结果驱动）"""
        level_styles = {
            "critical": {
                "color": "#c0392b", "bg": "#fdecea", "border": "#c0392b",
                "text": f"!!! 危急: {alert.class_name}",
            },
            "warning": {
                "color": "#e67e22", "bg": "#fef5e7", "border": "#e67e22",
                "text": f"!! 警告: {alert.class_name}",
            },
            "pre_alert": {
                "color": "#f39c12", "bg": "#fef9e7", "border": "#f39c12",
                "text": f"! 注意: {alert.class_name}",
            },
        }
        style = level_styles.get(alert.level, level_styles["warning"])

        self._alert_indicator.setText(f"● {style['text']}")
        self._alert_indicator.setStyleSheet(
            f"color: {style['color']}; padding: 8px;"
            f"border: 2px solid {style['border']}; border-radius: 8px;"
            f"background-color: {style['bg']};"
        )

        self._label_last_alert.setText(
            f"最近报警 [{alert.level}]: {alert.class_name} | {alert.message[:60]}"
        )

        # 添加到历史
        self._history.add_alert(alert)

    def _on_omni_result(self, result: EmotionResult):
        """收到 Omni 情绪分析结果 → 更新显示"""
        # 标记来源
        self._source_is_omni = result.api_success
        if result.api_success:
            self._label_source.setText("来源: Omni 云端 ☁")
            self._label_source.setStyleSheet("color: #3498db; font-weight: bold;")
        else:
            self._label_source.setText("来源: Omni (降级)")
            self._label_source.setStyleSheet("color: #e74c3c;")

        # 报警指示灯联动 Omni 危险等级
        if result.danger_level == "危险":
            self._alert_indicator.setText(f"🔴 危险: {result.emotion_cn}")
            self._alert_indicator.setStyleSheet(
                "color: #c0392b; padding: 8px;"
                "border: 2px solid #c0392b; border-radius: 8px;"
                "background-color: #fdecea;"
            )
        elif result.danger_level == "关注":
            self._alert_indicator.setText(f"🟡 关注: {result.emotion_cn}")
            self._alert_indicator.setStyleSheet(
                "color: #e67e22; padding: 8px;"
                "border: 2px solid #e67e22; border-radius: 8px;"
                "background-color: #fef5e7;"
            )
        elif result.api_success:
            self._reset_alert_indicator()

        # 情绪标签
        if result.emotion_cn:
            self._label_omni_emotion.setText(f"情绪: {result.emotion_cn}")
            if result.emotion_confidence > 0:
                self._label_omni_emotion.setText(
                    f"情绪: {result.emotion_cn} ({result.emotion_confidence:.0%})"
                )

        # 危险等级
        danger_styles = {
            "危险": ("color: #c0392b; font-weight: bold;", f"🔴 危险 ({result.danger_score:.2f})"),
            "关注": ("color: #e67e22; font-weight: bold;", f"🟡 关注 ({result.danger_score:.2f})"),
            "正常": ("color: #2ecc71;", f"🟢 正常 ({result.danger_score:.2f})"),
        }
        style, text = danger_styles.get(result.danger_level, ("", result.danger_level))
        self._label_omni_danger.setText(f"等级: {text}")
        self._label_omni_danger.setStyleSheet(style)
        self._label_omni_emotion.setStyleSheet(style)

        # 转写文本
        transcript = result.text if result.text else "(未检测到语音)"
        if len(transcript) > 60:
            transcript = transcript[:57] + "..."
        self._label_omni_text.setText(f"转写: {transcript}")

        # 判定理由
        reason = result.reason if result.reason else "--"
        if len(reason) > 80:
            reason = reason[:77] + "..."
        self._label_omni_reason.setText(f"理由: {reason}")

        # 延迟
        if result.api_latency_ms > 0:
            self._label_omni_latency.setText(f"延迟: {result.api_latency_ms:.0f}ms")
        else:
            self._label_omni_latency.setText("延迟: --")

        # 在线状态
        if result.api_success:
            self._label_omni_online.setText("在线: ✅")
            self._label_omni_online.setStyleSheet("color: #2ecc71;")
        else:
            self._label_omni_online.setText("在线: ❌ 降级")
            self._label_omni_online.setStyleSheet("color: #e74c3c;")

        # 顶栏状态文字
        if result.api_success:
            self._omni_status.setText(
                f"上次分析: {result.emotion_cn} | {result.danger_level} | "
                f"{result.api_latency_ms:.0f}ms"
            )
            self._omni_status.setStyleSheet("color: #7f8c8d; padding: 4px;")
        else:
            self._omni_status.setText(f"⚠ API 失败: {result.error_message[:40]}")
            self._omni_status.setStyleSheet("color: #e74c3c; padding: 4px;")

    def _reset_alert_indicator(self):
        """恢复报警指示灯为正常状态"""
        self._alert_indicator.setText("● 正常")
        self._alert_indicator.setStyleSheet(
            "color: #2ecc71; padding: 8px;"
            "border: 2px solid #2ecc71; border-radius: 8px;"
            "background-color: transparent;"
        )

    def _on_status(self, text: str):
        """更新状态栏 / Omni 状态"""
        if text.startswith("__omni__"):
            # Omni 状态专用消息格式: __omni__<color>__<text>
            parts = text.split("__", 3)
            if len(parts) >= 4:
                self._omni_status.setText(parts[3])
                self._omni_status.setStyleSheet(f"color: {parts[2]}; padding: 4px;")
        else:
            self._statusbar.showMessage(text, 5000)

    def _on_connection_change(self, connected: bool):
        """连接状态变更"""
        self._connected = connected
        if connected:
            self._status_connection.setText("连接状态: ● 已连接")
            self._status_connection.setStyleSheet(
                "color: #2ecc71; padding: 2px 8px; font-weight: bold;"
            )
        else:
            self._status_connection.setText("连接状态: ● 未连接")
            self._status_connection.setStyleSheet(
                "color: #e74c3c; padding: 2px 8px;"
            )

    # ═══════════════════════════════════════════════════════════
    # 统计栏与定时刷新 — FPS / 帧数 / 异常率
    # ═══════════════════════════════════════════════════════════

    def _on_refresh(self):
        """定时刷新 (50ms)"""
        # 刷新波形
        self._waveform.update_display()

        # 更新 FPS
        now = time.time()
        # 清理1秒前的时间戳
        while self._recent_fps and now - self._recent_fps[0] > 1.0:
            self._recent_fps.popleft()

        fps = len(self._recent_fps) if self._recent_fps else 0
        self._label_fps.setText(f"帧率: {fps} fps")
        self._label_frames.setText(f"总帧数: {self._frame_count}")

        # 异常率 & 统计
        if self._current_result:
            ab_rate = (self._frame_count_abnormal / self._frame_count * 100) if self._frame_count > 0 else 0
            self._label_abnormal_rate.setText(
                f"异常率: {ab_rate:.1f}% | 当前: {self._current_result.class_name}"
            )

    # ═══════════════════════════════════════════════════════════
    # SHUTDOWN — clean resource release on window close
    # ═══════════════════════════════════════════════════════════

    def closeEvent(self, event):
        """窗口关闭事件"""
        self._refresh_timer.stop()
        logger.info("GUI 窗口已关闭")
        event.accept()
