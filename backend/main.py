"""
后端入口
启动 WebSocket 服务器 + PyQt5 GUI + HTML仪表盘数据桥接

用法:
    python main.py                           # 使用默认配置
    python main.py --host 0.0.0.0 --port 8765
    python main.py --simulate                # 模拟模式 (无需前端)
"""

import sys
import asyncio
import argparse
import json
import os
import re
import threading
import queue
import time
import tempfile
from pathlib import Path
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — handles concurrent browser requests"""
    daemon_threads = True

import numpy as np

# WebSocket 服务端 (用于仪表盘广播)
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from backend.config import BackendConfig
from backend.audio_receiver import AudioReceiver
from backend.feature_extractor import FeatureExtractor
from backend.audio_classifier import AudioClassifier
from backend.alert_manager import AlertManager
from backend.gui_main_window import MainWindow
from common.utils import setup_logging
from common.audio_config import AudioFrame, ClassifyResult, Alert

# Omni 智能体 (可选)
try:
    from asr_emotion_agent import OmniEmotionAgent, OmniConfig
    OMNI_AVAILABLE = True
except ImportError:
    OMNI_AVAILABLE = False

logger = None  # 在 main 中初始化

# 仪表盘 WebSocket 广播端口
DASHBOARD_PORT = 8766
# 浏览器客户端列表
dashboard_clients: set = set()


# ============================================================
# 预测平滑器: 消除孤立误判
# ============================================================

class PredictionSmoother:
    """
    增强预测平滑器 (v2)

    三层机制，从轻到重:
    1. 多数投票 (75% 门槛) — 类别判定，历史一致性
    2. EMA 指数移动平均 — 仅用于平滑置信度，不改类别
    3. 置信度 margin 门槛 — 异常判定需 top-1 明显领先

    关键: EMA 只平滑置信度的数值，不改变类别。类别由多数投票决定。
    """

    def __init__(self, window_size: int = 5,
                 ema_alpha: float = 0.4,
                 margin_threshold: float = 0.10):
        from collections import deque as _deque
        from collections import Counter as _Counter
        import copy as _copy
        self._deque = _deque
        self._Counter = _Counter
        self._copy = _copy
        self._history = _deque(maxlen=window_size)
        self.window_size = window_size
        self.ema_alpha = ema_alpha           # EMA 平滑系数
        self.margin_threshold = margin_threshold  # top-1 vs top-2 最小差距
        self._ema_conf: float | None = None  # 当前预测类别的 EMA 置信度

    def smooth(self, result: ClassifyResult) -> ClassifyResult:
        """
        平滑流程:
        1. 多数投票 → 决定类别
        2. EMA → 平滑该类的置信度数值
        3. margin 检查 → 异常判定是否需要打折
        """
        self._history.append(result)

        smoothed = self._copy.copy(result)

        # ---- 第1层: 多数投票决定类别 (75% 高门槛) ----
        if len(self._history) >= 3:
            votes = self._Counter(r.class_name for r in self._history)
            winner, winner_count = votes.most_common(1)[0]

            # 非常明确的多数 → 覆盖
            if (winner != smoothed.class_name and
                    winner_count >= len(self._history) * 0.75):
                smoothed.class_name = winner

        # ---- 第2层: EMA 平滑置信度 ----
        if self._ema_conf is None:
            self._ema_conf = result.confidence
        else:
            self._ema_conf = (self.ema_alpha * result.confidence +
                              (1 - self.ema_alpha) * self._ema_conf)
        smoothed.confidence = round(self._ema_conf, 4)

        # ---- 第3层: margin 检查 (用原始 all_probs) ----
        if result.all_probs:
            sorted_cls = sorted(result.all_probs.items(),
                                key=lambda x: x[1], reverse=True)
            if len(sorted_cls) >= 2:
                margin = sorted_cls[0][1] - sorted_cls[1][1]
                # margin 不足 → 模型在多个类之间犹豫
                if margin < self.margin_threshold:
                    # 如果模型自己都不确定是不是异常，折扣置信度
                    if smoothed.class_name != "normal":
                        discount = 0.6 + 0.4 * (margin / self.margin_threshold)
                        smoothed.confidence = round(smoothed.confidence * discount, 4)
            # 保留原始 all_probs 供仪表盘查看
            smoothed.all_probs = dict(result.all_probs)

        # ---- 同步 is_abnormal ----
        smoothed.is_abnormal = (smoothed.class_name != "normal")

        return smoothed

    def reset(self):
        self._history.clear()
        self._ema_conf = None


# ============================================================
# 后端流水线: 接收 → 特征提取 → 分类 → 报警 → GUI
# ============================================================

class BackendPipeline:
    """
    后端处理流水线
    串联 Receiver → FeatureExtractor → Classifier → AlertManager → GUI

    关键设计: 模型训练用 1 秒窗口，实时推理时累积音频缓冲
    到 1 秒再提取 MFCC 特征，确保与训练一致的输入分布。
    """

    def __init__(self, config: BackendConfig, gui: MainWindow):
        self.config = config
        self.gui = gui

        # 接收器 (成员B)
        self.receiver = AudioReceiver(
            host=config.host,
            port=config.port,
        )

        # 特征提取器 (成员C)
        self.extractor = FeatureExtractor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.n_mfcc,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            simulate=config.simulate,
        )

        # AI 分类器 (成员C)
        # input_dim 为 dict: {channels: 3, n_mels: 128}
        self.classifier = AudioClassifier(
            model_path=config.model_path,
            input_dim=self.extractor.get_feature_dim(),
            num_classes=config.num_classes,
            class_names=config.class_names,
            simulate=config.simulate,
        )
        self.classifier_mode = config.classifier_mode  # "nonverbal" | "emotion"

        # 预测平滑器 (消除孤立误判)
        self.smoother = PredictionSmoother(window_size=5)

        # 报警引擎 (成员D)
        self.alert_manager = AlertManager(config=config.alert)

        # 注册回调链
        self.receiver.on_frame(self._process_frame)

        # 统计
        self.frames_processed = 0

        # ---- 音频缓冲: 累积到 1 秒再做特征提取 ----
        self._window_samples = config.audio.sample_rate  # 1 秒 = 16000 样本
        self._audio_buffer = np.array([], dtype=np.float32)

        # ---- Omni 智能体 (可选) ----
        self._omni_enabled = (
            OMNI_AVAILABLE and
            getattr(config, 'omni_enabled', True)
        )
        self.omni_agent = None
        self._omni_buffer = np.array([], dtype=np.float32)
        self._omni_window_samples = int(
            getattr(config, 'omni_window_sec', 2.0) * 16000
        )
        self._omni_trigger_interval = getattr(config, 'omni_trigger_interval', 2.0)
        self._last_omni_time = 0.0

        # Omni 危险时间戳 (用于 CNN+Omni 双确认门控)
        self._omni_danger_times: deque = deque(maxlen=20)

        if self._omni_enabled:
            self._init_omni()

        # ---- 仪表盘回调 ----
        self._dashboard_callback = None

    def _init_omni(self):
        """初始化 Omni 智能体"""
        try:
            omni_config = OmniConfig()
            omni_config.load_from_env()
            # 尝试从文件加载额外配置
            try:
                omni_config.load_from_file()
            except Exception:
                pass
            # ★ 无密钥时终端交互式询问
            omni_config.prompt_for_key()
            # 应用后端配置覆盖
            if hasattr(self.config, 'omni_danger_threshold'):
                omni_config.danger_threshold = self.config.omni_danger_threshold
            omni_config.window_seconds = getattr(self.config, 'omni_window_sec', 2.0)

            self.omni_agent = OmniEmotionAgent(omni_config)

            if self.omni_agent.client.is_available:
                logger.info(
                    f"[Omni] 智能体已就绪 | model={omni_config.model} | "
                    f"窗口={omni_config.window_seconds}s | "
                    f"间隔={self._omni_trigger_interval}s"
                )
                self.gui.set_omni_status("Omni 智能体已就绪 ✅", "#2ecc71")
            else:
                logger.warning("[Omni] API Key 未设置，Omni 通路不可用")
                self.gui.set_omni_status("Omni 未配置 API Key ⚠", "#e67e22")
                self._omni_enabled = False
        except Exception as e:
            logger.warning(f"[Omni] 初始化失败: {e}")
            self.gui.set_omni_status(f"Omni 初始化失败: {e}", "#e74c3c")
            self._omni_enabled = False

    def _process_omni(self, frame: AudioFrame):
        """
        Omni 主判断流水线 (每3秒触发一次)

        1. 累积音频到 2秒缓冲区
        2. VAD 预过滤: 无人声则跳过 API 调用 (节省 ~70% 成本)
        3. 按触发间隔调用 Omni API
        4. 结果作为主判断 → GUI 主显示 + Omni 面板
        """
        # 累积音频
        self._omni_buffer = np.concatenate([self._omni_buffer, frame.data])

        # 保留最近 5 秒
        max_samples = 16000 * 5
        if len(self._omni_buffer) > max_samples:
            self._omni_buffer = self._omni_buffer[-max_samples:]

        # 触发条件检查
        now = time.time()
        buffer_ready = len(self._omni_buffer) >= self._omni_window_samples
        interval_ready = (now - self._last_omni_time) >= self._omni_trigger_interval

        if not (buffer_ready and interval_ready):
            return

        # 取最近 2 秒音频窗口
        audio_window = self._omni_buffer[-self._omni_window_samples:].copy()

        # ★ VAD 前置过滤: 无人声时跳过 API 调用，节省成本
        if not self.extractor.is_likely_speech(audio_window):
            logger.debug("[Omni] VAD 过滤: 无人声，跳过 API 调用")
            self._last_omni_time = now
            # 通知 GUI: Omni 跳过一次 (静音)
            self.gui.set_omni_status("跳过 (无人声) 💤", "#95a5a6")
            return

        # ★ 调 Omni API (主判断)
        logger.debug("[Omni] 发送音频到云端...")
        result = self.omni_agent.judge(audio_window)

        # 处理结果 → 作为主判断
        self._handle_omni_result(result)

        self._last_omni_time = now

    def _handle_omni_result(self, result):
        """
        处理 Omni 返回结果 (★ 主判断模式)

        Omni 作为主要判断来源:
          1. 更新 GUI 主显示区 (情绪分类 + 置信度)
          2. 更新 Omni 详情面板 (转写 + 理由 + 延迟)
          3. 危险时触发报警
        """
        # ---- GUI 主显示区: Omni 结果覆盖 CNN ----
        # 将 Omni 情绪映射到 CNN 5 类体系
        omni_to_cnn = {
            "fearful": "scream", "angry": "scream",
            "sad": "cry", "disgusted": "cry",
            "surprised": "normal", "happy": "laugh",
            "neutral": "normal",
        }
        cnn_class = omni_to_cnn.get(result.emotion, "normal")
        is_abnormal = cnn_class != "normal"

        # 构建 ClassifyResult 用于主显示区
        class_names = self.config.class_names
        all_probs = {c: 0.0 for c in class_names}
        all_probs[cnn_class] = result.emotion_confidence if result.api_success else 0.5
        # 将剩余概率分配给其他类别
        remaining = 1.0 - all_probs[cnn_class]
        other_classes = [c for c in class_names if c != cnn_class]
        for c in other_classes:
            all_probs[c] = remaining / len(other_classes)

        main_result = ClassifyResult(
            class_name=cnn_class,
            confidence=result.danger_score if is_abnormal else result.emotion_confidence,
            is_abnormal=is_abnormal,
            timestamp=result.timestamp,
            frame_index=int(result.timestamp * 1000) % 1000000,
            all_probs=all_probs,
        )
        # ★ 主显示区更新
        self.gui.feed_classify_result(main_result)

        # 送入 smoother 保持连续性
        self.smoother.smooth(main_result)

        # ---- Omni 详情面板 ----
        self.gui.feed_emotionresult(result)

        # ---- 报警 ----
        if result.danger_level == "危险":
            # ★ 记录 Omni 危险时间戳 (CNN 双确认门控用)
            self._omni_danger_times.append(time.time())

            alert = Alert(
                level="warning",
                class_name=result.emotion,
                message=(
                    f"[Omni] {result.emotion_cn}: {result.reason[:60]} | "
                    f"严重度: {result.danger_score:.2f} | "
                    f"转写: \"{result.text[:20]}\""
                ),
                timestamp=result.timestamp,
            )
            self.gui.feed_alert(alert)

            # 录入 AlertManager 统计
            self.alert_manager.feed(main_result)

            logger.warning(
                f"🔴 [Omni·主判断] 危险! {result.emotion_cn} "
                f"(score={result.danger_score:.2f}) | {result.reason[:80]}"
            )

        elif result.danger_level == "关注":
            logger.info(
                f"🟡 [Omni·主判断] 关注 | {result.emotion_cn} "
                f"(score={result.danger_score:.2f})"
            )
        else:
            logger.debug(
                f"🟢 [Omni·主判断] 正常 | {result.emotion_cn} "
                f"(score={result.danger_score:.2f})"
            )

    def _should_allow_cnn_alert(self, time_window: float = 5.0) -> bool:
        """
        CNN+Omni 双确认门控

        CNN 报警放行条件 (满足任一即可):
          1. Omni 不可用 (离线/未配置) → 放行，CNN 独立工作
          2. Omni 在 ±time_window 秒内也检测到危险 → 双确认通过

        Returns:
            True: 放行 CNN 报警
            False: 抑制 CNN 报警 (等待 Omni 确认)
        """
        # 条件1: Omni 不可用 → CNN 独立报警
        if not self._omni_enabled or self.omni_agent is None:
            return True
        if not self.omni_agent.client.is_available:
            return True

        # 条件2: Omni 在时间窗口内检测到危险
        now = time.time()
        # 清理过期记录
        while self._omni_danger_times and (now - self._omni_danger_times[0] > time_window * 2):
            self._omni_danger_times.popleft()

        for danger_time in self._omni_danger_times:
            if abs(now - danger_time) <= time_window:
                logger.info(
                    f"[门控] CNN+Omni 双确认通过! "
                    f"Omni危险时间: {danger_time:.0f}, 当前: {now:.0f}, "
                    f"间隔: {abs(now-danger_time):.1f}s"
                )
                return True

        return False

    def set_dashboard_callback(self, callback):
        """注册仪表盘数据推送回调"""
        self._dashboard_callback = callback

    # ---- 情绪标签映射: 9 类情绪 → 5 类非言语 ----
    EMOTION_TO_NONVERBAL = {
        "angry": "scream", "fearful": "scream",
        "sad": "cry",
        "happy": "laugh",
        "disgusted": "cry",
        "surprised": "normal", "neutral": "normal",
        "other": "normal", "unk": "normal",
    }

    def _map_emotion_to_nonverbal(self, result: ClassifyResult) -> ClassifyResult:
        """
        将 emotion2vec 9 类情绪标签映射到 5 类非言语体系。
        保留原始情绪数据到 all_probs_emotion，同时构造兼容的 all_probs。
        """
        emo_class = result.class_name
        mapped_class = self.EMOTION_TO_NONVERBAL.get(emo_class, "normal")

        # 保留原始情绪概率
        original_probs = dict(result.all_probs) if result.all_probs else {}

        # 构造 5 类非言语概率分布
        nonverbal_probs = {"normal": 0.0, "scream": 0.0, "cry": 0.0,
                           "laugh": 0.0}
        for emo, prob in original_probs.items():
            target = self.EMOTION_TO_NONVERBAL.get(emo, "normal")
            nonverbal_probs[target] += prob

        # 归一化
        total = sum(nonverbal_probs.values())
        if total > 0:
            nonverbal_probs = {k: round(v / total, 4) for k, v in nonverbal_probs.items()}

        # 置信度用主导类别的值
        confidence = round(nonverbal_probs.get(mapped_class, result.confidence), 4)

        return ClassifyResult(
            class_name=mapped_class,
            confidence=confidence,
            is_abnormal=(mapped_class != "normal"),
            timestamp=result.timestamp,
            frame_index=result.frame_index,
            all_probs=nonverbal_probs,
        )

    def _process_frame(self, frame: AudioFrame):
        """
        处理收到的音频帧（在 WebSocket 事件循环中调用）
        累积音频 → 满 1 秒 → 特征提取 → 分类 → 报警 → GUI + 仪表盘
        """
        self.frames_processed += 1

        # → GUI 波形显示 (每帧都显示，保持流畅)
        self.gui.feed_audio_frame(frame)

        # → 累积音频到缓冲区
        self._audio_buffer = np.concatenate([self._audio_buffer, frame.data])

        # 保持缓冲区不超过 2 秒 (防止内存增长)
        max_samples = self._window_samples * 2
        if len(self._audio_buffer) > max_samples:
            self._audio_buffer = self._audio_buffer[-max_samples:]

        # 缓冲区够 1 秒了才做推理
        if len(self._audio_buffer) < self._window_samples:
            return

        # 取最近 1 秒的音频用于特征提取
        audio_window = self._audio_buffer[-self._window_samples:]

        # → 语音检测: 静音/噪声直接判 normal，不送入模型
        if not self.extractor.is_likely_speech(audio_window):
            import time as _time
            result = ClassifyResult(
                class_name="normal",
                confidence=0.99,
                is_abnormal=False,
                timestamp=_time.time(),
                frame_index=frame.frame_index,
                all_probs={"normal": 0.99, "scream": 0.00, "cry": 0.01,
                           "laugh": 0.00},
            )
            self.gui.feed_classify_result(result)
            # 静音帧仍然传给 smoother 维持 normal 状态
            self.smoother.smooth(result)
            return

        # → 推理 (按 mode 分发)
        if self.classifier_mode == 'emotion':
            # 原生情绪识别模式: 愤怒/恐惧/高兴/悲伤/中性/惊讶
            result = self.classifier.predict_emotion_from_audio(
                audio_window,
                frame_index=frame.frame_index,
            )
            # ★ 映射 9 类情绪 → 5 类非言语体系 (用于 GUI/报警/仪表盘)
            result = self._map_emotion_to_nonverbal(result)
        else:
            # emotion2vec + FC 头: 非言语检测 (normal/scream/cry/laugh)
            result = self.classifier.predict_from_audio(
                audio_window,
                frame_index=frame.frame_index,
            )

        # → 预测平滑 (滑动窗口投票，消除孤立误判)
        result = self.smoother.smooth(result)

        # → GUI 分类结果显示
        self.gui.feed_classify_result(result)

        # → 报警判断 (双确认门控: CNN 报警需 Omni 也检测到危险才放行)
        alert = self.alert_manager.feed(result)
        if alert:
            if self._should_allow_cnn_alert():
                self.gui.feed_alert(alert)
            else:
                logger.debug(
                    f"[门控] CNN 报警被抑制: {alert.class_name} "
                    f"(Omni 未确认危险，等待双确认)"
                )

        # → 仪表盘 WebSocket 广播
        if self._dashboard_callback:
            try:
                self._dashboard_callback(result, alert)
            except Exception:
                logger.error("仪表盘广播失败", exc_info=True)

        # → Omni 智能体分析 (异步于 CNN，不阻塞)
        if self._omni_enabled and self.omni_agent is not None:
            try:
                self._process_omni(frame)
            except Exception:
                logger.error("Omni 分析异常", exc_info=True)

        # 日志 (每100帧输出一次)
        if self.frames_processed % 100 == 0:
            stats = self.alert_manager.get_stats()
            log_msg = (
                f"已处理 {self.frames_processed} 帧 | "
                f"缓冲: {len(self._audio_buffer)}样本 | "
                f"异常率: {stats['abnormal_rate']:.1%} | "
                f"报警: {stats['total_alerts']}次"
            )
            if self._omni_enabled and self.omni_agent is not None:
                omni_stats = self.omni_agent.get_stats()
                log_msg += (
                    f" | Omni: {omni_stats['total_judgments']}次 "
                    f"(危险{omni_stats['danger_count']})"
                )
            logger.info(log_msg)

    async def start_server(self):
        """启动 WebSocket 服务器"""
        await self.receiver.start()
        self.gui.set_connection_status(True)
        logger.info("后端流水线已就绪，等待前端连接...")

    async def stop_server(self):
        """停止服务器"""
        await self.receiver.stop()
        self.gui.set_connection_status(False)


# ============================================================
# 主程序
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="后端: 音频接收 + AI分类 + GUI显示"
    )
    parser.add_argument("--host", type=str, default=None,
                        help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None,
                        help="监听端口 (默认: 8765)")
    parser.add_argument("--simulate", action="store_true",
                        help="模拟模式 (后端自产自消，无需前端)")
    parser.add_argument("--no-omni", action="store_true",
                        help="禁用 Omni 智能体通路")
    return parser.parse_args()


def run_asyncio_loop(loop, pipeline, args):
    """在单独的线程中运行 asyncio 事件循环"""
    asyncio.set_event_loop(loop)

    # ---- 仪表盘 WebSocket 服务 ----
    async def handle_dashboard_client(websocket):
        """浏览器仪表盘客户端连接处理"""
        client_id = id(websocket)
        dashboard_clients.add(websocket)
        logger.info(f"[仪表盘] 浏览器已连接 (共 {len(dashboard_clients)} 个)")
        try:
            # 保持连接，等待客户端断开
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            dashboard_clients.discard(websocket)
            logger.info(f"[仪表盘] 浏览器已断开 (剩余 {len(dashboard_clients)} 个)")

    async def broadcast_dashboard(result: ClassifyResult, alert: Alert | None):
        """向所有连接的浏览器广播分类结果"""
        if not dashboard_clients:
            return
        payload = {
            "type": "classifyresult",
            "class_name": result.class_name,
            "confidence": round(result.confidence, 4),
            "is_abnormal": result.is_abnormal,
            "timestamp": result.timestamp,
            "frame_index": result.frame_index,
            "all_probs": getattr(result, 'all_probs', None) or {},
        }
        if alert:
            payload["alert"] = {
                "level": alert.level,
                "severity": round(alert.severity, 4),
                "class_name": alert.class_name,
                "message": alert.message,
            }
        msg = json.dumps(payload, ensure_ascii=False)
        # 并发广播给所有客户端
        dead = set()
        for client in dashboard_clients.copy():
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        dashboard_clients.difference_update(dead)

    async def run():
        await pipeline.start_server()

        # 启动仪表盘 WebSocket 服务器
        if not WEBSOCKETS_AVAILABLE:
            logger.error("[仪表盘] websockets 库未安装, 仪表盘功能不可用")
            logger.error("[仪表盘] 安装: pip install websockets")
            dash_server = None
        else:
            dash_server = await websockets.serve(
                handle_dashboard_client,
                pipeline.config.host,
                DASHBOARD_PORT,
            )
            logger.info(f"[仪表盘] WebSocket 服务: ws://{pipeline.config.host}:{DASHBOARD_PORT}")

        # 注册仪表盘回调
        pipeline.set_dashboard_callback(
            lambda result, alert: asyncio.run_coroutine_threadsafe(
                broadcast_dashboard(result, alert), loop
            )
        )

        # 保持运行
        while True:
            await asyncio.sleep(1)

    try:
        loop.run_until_complete(run())
    except asyncio.CancelledError:
        pass
    finally:
        loop.run_until_complete(pipeline.stop_server())


def _get_local_ips() -> list[str]:
    """获取本机所有局域网 IP 地址"""
    import socket
    ips = []
    try:
        hostname = socket.gethostname()
        # 获取所有 IP
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip != "127.0.0.1" and not ip.startswith("169.254"):
                ips.append(ip)
    except Exception:
        pass

    # 去重
    return list(dict.fromkeys(ips))


# ---- 文件上传分析 ----
# 模块级注册: HTTP handler 通过此字典访问 classifier / extractor
_analysis_registry: dict = {}


def _convert_audio_to_wav(file_bytes: bytes, suffix: str) -> bytes:
    """
    将浏览器上传的音频转为 WAV (16kHz/mono/16bit)，供 wave.open() 读取。
    浏览器录音默认输出 WebM，后端 wave 模块只认 WAV。
    Fallback: ffmpeg → sox → 直接返回原始字节（让下游处理）。
    """
    suffix_lower = suffix.lower()
    if suffix_lower == '.wav':
        return file_bytes  # 已是 WAV

    _http_logger = __import__('logging').getLogger("backend.http")

    # 写入临时文件
    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix_lower, delete=False)
    tmp_in.write(file_bytes)
    tmp_in.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_out.close()

    try:
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-y', '-i', tmp_in.name,
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
            tmp_out.name,
        ], capture_output=True, timeout=60)

        if result.returncode == 0 and Path(tmp_out.name).stat().st_size > 0:
            with open(tmp_out.name, 'rb') as f:
                wav_bytes = f.read()
            _http_logger.info(f"[WebM→WAV] 转换成功: {len(file_bytes)} → {len(wav_bytes)} bytes")
            return wav_bytes
        else:
            _http_logger.warning(f"[WebM->WAV] ffmpeg conversion failed: {result.stderr.decode()[:200]}")
            raise RuntimeError("ffmpeg conversion failed")
    except FileNotFoundError:
        _http_logger.warning("[WebM->WAV] ffmpeg not found; cannot convert non-WAV formats")
        raise RuntimeError("ffmpeg not installed; please use WAV files or install ffmpeg")
    except Exception:
        raise
    finally:
        os.unlink(tmp_in.name)
        if Path(tmp_out.name).exists():
            os.unlink(tmp_out.name)


def _start_http_server(port: int = 8080):
    """启动 HTTP 服务器: 静态文件 + /api/analyze 文件上传分析"""
    import logging as _logging
    _http_logger = _logging.getLogger("backend.http")

    # 切换到 design-demos 目录提供文件服务
    demo_dir = _PROJECT_ROOT / "design-demos"
    if not demo_dir.exists():
        _http_logger.warning(f"design-demos 目录不存在: {demo_dir}")
        return

    os.chdir(str(demo_dir))

    class APIHandler(SimpleHTTPRequestHandler):
        """扩展 HTTP 处理器: 静态文件 + API 端点"""

        def do_POST(self):
            """POST 路由"""
            if self.path == "/api/analyze":
                self._handle_analyze()
            elif self.path == "/api/omni":
                self._handle_omni()
            else:
                self.send_error(404, "API endpoint not found")

        def _parse_multipart(self):
            """手动解析 multipart/form-data，返回 {field_name: (filename, data_bytes)}"""
            content_type = self.headers.get("Content-Type", "")
            boundary_match = re.search(r"boundary=([^;]+)", content_type)
            if not boundary_match:
                return {}
            boundary = boundary_match.group(1).strip().strip('"')

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            boundary_b = ("--" + boundary).encode()

            parts = body.split(boundary_b)
            result = {}
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                # 分割头部和内容
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                header_section = part[:header_end].decode("utf-8", errors="replace")
                data = part[header_end + 4:]

                # 去除尾部 \r\n 和 -- 标记
                data = data.rstrip(b"\r\n").rstrip(b"--").rstrip(b"\r\n")

                # 解析 name 和 filename
                name_m = re.search(r'name="([^"]+)"', header_section)
                fname_m = re.search(r'filename="([^"]+)"', header_section)
                if name_m:
                    result[name_m.group(1)] = (
                        fname_m.group(1) if fname_m else None,
                        data,
                    )
            return result

        def _handle_analyze(self):
            """接收上传文件 → 分析 → 返回 JSON"""
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_error(400, "需要 multipart/form-data")
                    return

                form = self._parse_multipart()
                file_entry = form.get("file")
                if file_entry is None:
                    self.send_error(400, "缺少 'file' 字段")
                    return

                filename, file_data = file_entry
                if file_data is None:
                    self.send_error(400, "文件数据为空")
                    return

                # 保存临时文件
                suffix = Path(filename or "audio").suffix or ".wav"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp_path = tmp.name
                tmp.write(file_data)
                tmp.close()

                _http_logger.info(f"[API] 收到文件: {filename} -> {tmp_path}")

                # 确保后端已就绪
                classifier = _analysis_registry.get("classifier")
                extractor = _analysis_registry.get("extractor")
                if not classifier or not extractor:
                    self.send_error(503, "Analysis module not ready, please retry later")
                    os.unlink(tmp_path)
                    return

                # ★ 提取音频喂入 GUI 波形
                from backend.file_handler import extract_audio, analyze_file
                _gui = _analysis_registry.get("gui")
                if _gui:
                    try:
                        wave_audio = extract_audio(tmp_path, target_sr=16000)
                        if wave_audio is not None:
                            _gui.feed_file_audio(wave_audio)
                    except Exception:
                        pass

                # 分析文件
                classifier_mode = _analysis_registry.get("classifier_mode", "nonverbal")
                result = analyze_file(
                    tmp_path,
                    classifier=classifier,
                    extractor=extractor,
                    classifier_mode=classifier_mode,
                    window_sec=1.0,
                    step_sec=0.5,
                )

                # 清理临时文件
                os.unlink(tmp_path)

                # ★ 广播到仪表盘 WebSocket
                _broadcast = _analysis_registry.get("broadcast_dashboard")
                if _broadcast:
                    dominant = result.get("dominant_emotion", "normal")
                    dash_payload = {
                        "type": "classifyresult",
                        "class_name": dominant,
                        "confidence": result.get("confidence", 0),
                        "is_abnormal": dominant not in ("normal", "silence"),
                        "timestamp": time.time(),
                        "frame_index": 0,
                        "all_probs": result.get("emotion_distribution", {}),
                        "source": "file_cnn",
                        "file": filename,
                        "cnn": {
                            "total_windows": result.get("total_windows", 0),
                            "summary": result.get("summary", ""),
                        },
                    }
                    if dominant not in ("normal", "silence"):
                        dash_payload["alert"] = {
                            "level": "warning",
                            "severity": round(result.get("confidence", 0), 4),
                            "class_name": dominant,
                            "message": f"[CNN·文件分析] 检测到异常: {dominant} (置信度 {result.get('confidence', 0):.0%})",
                        }
                    _broadcast(dash_payload)

                # ★ 同步更新 PyQt5 GUI
                _gui = _analysis_registry.get("gui")
                if _gui:
                    dominant = result.get("dominant_emotion", "normal")
                    is_abnormal = dominant not in ("normal", "silence")
                    all_probs = result.get("emotion_distribution", {})
                    classify_result = ClassifyResult(
                        class_name=dominant,
                        confidence=result.get("confidence", 0),
                        is_abnormal=is_abnormal,
                        timestamp=time.time(),
                        frame_index=0,
                        all_probs=all_probs,
                    )
                    _gui.feed_classify_result(classify_result)
                    _gui.feed_source("CNN 文件分析 ⚡", "#7f8c8d")

                    if is_abnormal:
                        alert = Alert(
                            level="warning",
                            class_name=dominant,
                            message=f"[CNN·文件] {result.get('summary', '')[:80]}",
                            timestamp=time.time(),
                            severity=round(result.get("confidence", 0), 4),
                        )
                        _gui.feed_alert(alert)
                        # 更新报警统计
                        _alert_mgr = _analysis_registry.get("alert_manager")
                        if _alert_mgr:
                            _alert_mgr.feed(classify_result)

                # 返回 JSON
                response = json.dumps(result, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))
                _http_logger.info(
                    f"[API] 分析完成: 主导={result.get('dominant_emotion')}, "
                    f"置信度={result.get('confidence', 0):.2%}"
                )

            except Exception as e:
                _http_logger.error(f"[API] 分析失败: {e}", exc_info=True)
                self.send_error(500, f"Analysis failed: {str(e).encode('ascii','replace').decode('ascii')}")

        def _handle_omni(self):
            """接收上传文件 → Omni 智能体分析 → 返回 JSON"""
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_error(400, "需要 multipart/form-data")
                    return

                form = self._parse_multipart()
                file_entry = form.get("file")
                if file_entry is None:
                    self.send_error(400, "缺少 'file' 字段")
                    return

                filename, file_data = file_entry
                if file_data is None:
                    self.send_error(400, "文件数据为空")
                    return

                # 保存临时文件
                suffix = Path(filename or "audio").suffix or ".wav"
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp_path = tmp.name
                tmp.write(file_data)
                tmp.close()

                _http_logger.info(f"[API/Omni] 收到文件: {filename} -> {tmp_path}")

                # 确保 Omni Agent 已就绪
                omni_agent = _analysis_registry.get("omni_agent")
                if omni_agent is None:
                    self.send_error(503, "Omni agent not ready")
                    os.unlink(tmp_path)
                    return

                # 用 extract_audio 加载 (支持 WAV/MP3/M4A/FLAC/WebM 等)
                from backend.file_handler import extract_audio
                audio = extract_audio(tmp_path, target_sr=16000)
                if audio is None:
                    self.send_error(400, "Cannot read audio file; format not supported or ffmpeg missing")
                    os.unlink(tmp_path)
                    return

                # 归一化
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio / peak

                # → 截取/填充到 2 秒 (32000 samples @ 16kHz)
                target_samples = 32000
                if len(audio) > target_samples:
                    audio = audio[-target_samples:]  # 取最后 2 秒
                elif len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")

                _http_logger.info(
                    f"[API/Omni] 音频就绪: {len(audio)} samples @ 16kHz"
                )

                # ★ 喂入 GUI 波形
                _gui = _analysis_registry.get("gui")
                if _gui:
                    try:
                        _gui.feed_file_audio(audio)
                    except Exception:
                        pass

                # 调用 Omni Agent
                import time as _time
                t0 = _time.time()
                result = omni_agent.judge(audio)
                elapsed = _time.time() - t0
                _http_logger.info(
                    f"[API/Omni] 分析完成: {result.emotion_cn} | "
                    f"{result.danger_level} | {elapsed:.1f}s"
                )

                # 清理临时文件
                os.unlink(tmp_path)

                # 构建 JSON 响应
                from dataclasses import asdict as _asdict
                resp_dict = _asdict(result)
                # 移除内部调试字段
                resp_dict.pop("rawresponse", None)
                # 添加文件信息
                resp_dict["file"] = filename or "unknown"
                resp_dict["duration_sec"] = len(audio) / 16000
                # 统一 keywords 字段名
                resp_dict["keywords"] = resp_dict.pop("keywords_detected", [])

                # ★ 广播到仪表盘 WebSocket
                _broadcast = _analysis_registry.get("broadcast_dashboard")
                if _broadcast:
                    omni_to_cnn = {
                        "fearful": "scream", "angry": "scream",
                        "sad": "cry", "disgusted": "cry",
                        "surprised": "normal", "happy": "laugh",
                        "neutral": "normal",
                    }
                    cnn_class = omni_to_cnn.get(result.emotion, "normal")
                    dash_payload = {
                        "type": "classifyresult",
                        "class_name": cnn_class,
                        "confidence": result.danger_score if cnn_class != "normal" else result.emotion_confidence,
                        "is_abnormal": cnn_class != "normal",
                        "timestamp": result.timestamp,
                        "frame_index": 0,
                        "all_probs": {},
                        "source": "file_omni",
                        "file": filename,
                        "omni": {
                            "emotion_cn": result.emotion_cn,
                            "danger_level": result.danger_level,
                            "danger_score": result.danger_score,
                            "text": result.text[:100] if result.text else "",
                            "reason": result.reason[:120] if result.reason else "",
                            "api_latency_ms": result.api_latency_ms,
                        },
                    }
                    if result.danger_level == "危险":
                        dash_payload["alert"] = {
                            "level": "warning",
                            "severity": round(result.danger_score, 4),
                            "class_name": result.emotion,
                            "message": f"[Omni·文件分析] {result.emotion_cn}: {result.reason[:60] if result.reason else ''}",
                        }
                    _broadcast(dash_payload)

                # ★ 同步更新 PyQt5 GUI
                _gui = _analysis_registry.get("gui")
                if _gui:
                    cnn_class = omni_to_cnn.get(result.emotion, "normal")
                    is_abnormal = cnn_class != "normal"
                    all_probs = {c: 0.0 for c in ["normal", "scream", "cry", "laugh"]}
                    all_probs[cnn_class] = result.danger_score if is_abnormal else result.emotion_confidence
                    remaining = 1.0 - all_probs[cnn_class]
                    others = [c for c in all_probs if c != cnn_class]
                    for c in others:
                        all_probs[c] = remaining / len(others)
                    classify_result = ClassifyResult(
                        class_name=cnn_class,
                        confidence=result.danger_score if is_abnormal else result.emotion_confidence,
                        is_abnormal=is_abnormal,
                        timestamp=result.timestamp,
                        frame_index=0,
                        all_probs=all_probs,
                    )
                    _gui.feed_classify_result(classify_result)
                    _gui.feed_emotion_result(result)
                    _gui.feed_source("Omni 云端 ☁", "#3498db")

                    if result.danger_level == "危险":
                        alert = Alert(
                            level="warning",
                            class_name=result.emotion,
                            message=f"[Omni·文件] {result.emotion_cn}: {result.reason[:60] if result.reason else ''}",
                            timestamp=result.timestamp,
                            severity=round(result.danger_score, 4),
                        )
                        _gui.feed_alert(alert)
                        _alert_mgr = _analysis_registry.get("alert_manager")
                        if _alert_mgr:
                            _alert_mgr.feed(classify_result)

                response = json.dumps(resp_dict, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(response.encode("utf-8"))

            except Exception as e:
                _http_logger.error(f"[API/Omni] 分析失败: {e}", exc_info=True)
                self.send_error(500, f"Omni analysis failed: {str(e).encode('ascii','replace').decode('ascii')}")

        def do_OPTIONS(self):
            """CORS preflight"""
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def log_message(self, format, *args):
            _http_logger.debug(f"[HTTP] {args[0]}")

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
        _http_logger.info(f"HTTP 服务: http://localhost:{port}/")
        _http_logger.info(f"  后端监控面板: http://localhost:{port}/backend-dashboard.html")
        _http_logger.info(f"  前端采集面板: http://localhost:{port}/frontend-panel.html")
        _http_logger.info(f"  文件分析 (CNN): POST http://localhost:{port}/api/analyze")
        _http_logger.info(f"  文件分析 (Omni): POST http://localhost:{port}/api/omni")
        server.serve_forever()
    except OSError as e:
        _http_logger.warning(f"HTTP 服务启动失败 (端口 {port} 可能被占用): {e}")
    except Exception as e:
        _http_logger.error(f"HTTP 服务异常: {e}")


def main():
    global logger

    args = parse_args()

    # 配置
    config = BackendConfig()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.simulate:
        config.simulate = True
    if args.no_omni:
        config.omni_enabled = False

    # 日志
    logger = setup_logging("backend", config.log_level, config.log_dir)

    # 检测 ffmpeg (M4A/AAC/WebM 等格式需要)
    _has_ffmpeg = False
    try:
        import subprocess as _sp
        result = _sp.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        if result.returncode == 0:
            _has_ffmpeg = True
            logger.info("ffmpeg detected - M4A/AAC/WebM/MP4 formats fully supported")
    except Exception:
        pass
    if not _has_ffmpeg:
        logger.warning(
            "ffmpeg not found! M4A/AAC/WebM formats will fail to decode. "
            "Install: winget install ffmpeg  "
            "or https://ffmpeg.org/download.html"
        )

    # ---- 打印启动信息 ----
    logger.info("=" * 50)
    logger.info("后端启动: 双机实时音频监测系统")
    logger.info(f"监听地址: ws://{config.host}:{config.port}")
    logger.info(f"模拟模式: {'开' if config.simulate else '关'}")
    logger.info(f"Omni 智能体: {'启用' if config.omni_enabled else '禁用'}")
    logger.info("=" * 50)

    # 打印本机 IP（方便前端连接）
    local_ips = _get_local_ips()
    if local_ips:
        logger.info("本机局域网 IP (前端请用以下地址连接):")
        for ip in local_ips:
            logger.info(f"  >>> ws://{ip}:{config.port}")
        print(f"\n{'='*50}")
        print(f"  前端连接地址:")
        for ip in local_ips:
            print(f"  >>> ws://{ip}:{config.port}")
        print(f"\n  HTML 仪表盘:")
        print(f"  >>> http://localhost:8080/backend-dashboard.html  (后端监测)")
        print(f"  >>> http://localhost:8080/frontend-panel.html     (前端采集面板)")
        print(f"  仪表盘数据源: ws://localhost:{DASHBOARD_PORT}")
        print(f"{'='*50}\n")
    else:
        logger.warning("未能检测到局域网 IP，请手动运行 ipconfig 查看")

    # Qt 应用
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # GUI
    gui = MainWindow("双机实时音频监测系统 - 后端")
    gui.show()

    # 处理流水线
    pipeline = BackendPipeline(config, gui)

    # → 注册到 HTTP 文件分析接口
    _analysis_registry["classifier"] = pipeline.classifier
    _analysis_registry["extractor"] = pipeline.extractor
    _analysis_registry["omni_agent"] = pipeline.omni_agent
    _analysis_registry["classifier_mode"] = config.classifier_mode
    _analysis_registry["gui"] = gui
    _analysis_registry["alert_manager"] = pipeline.alert_manager
    logger.info("文件分析接口已就绪: POST /api/analyze")
    if pipeline.omni_agent is not None:
        logger.info("Omni 文件分析接口已就绪: POST /api/omni")

    # ★ 模型预热: 第一次推理触发 emotion2vec 懒加载, 提前完成避免用户等待
    try:
        warmup_audio = np.random.randn(16000).astype(np.float32) * 0.01  # 1秒低噪
        logger.info("正在预热 CNN 模型 (首次推理较慢, 请稍候)...")
        t_warm = time.time()
        if pipeline.classifier_mode == 'emotion':
            _ = pipeline.classifier.predict_emotion_from_audio(warmup_audio, frame_index=-1)
        else:
            _ = pipeline.classifier.predict_from_audio(warmup_audio, frame_index=-1)
        logger.info(f"CNN 模型预热完成 ({time.time()-t_warm:.1f}s)")
    except Exception as e:
        logger.warning(f"模型预热跳过: {e}")

    # Asyncio 事件循环 (在独立线程中运行)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=run_asyncio_loop,
        args=(loop, pipeline, args),
        daemon=True,
    )
    thread.start()

    # → 注册仪表盘广播函数 (供 HTTP 文件分析结果推送)
    def _broadcast_file_result(payload_dict: dict):
        """线程安全: 推送文件分析结果到仪表盘 WebSocket 客户端"""
        if not dashboard_clients:
            return
        msg = json.dumps(payload_dict, ensure_ascii=False)
        async def _send():
            dead = set()
            for client in dashboard_clients.copy():
                try:
                    await client.send(msg)
                except Exception:
                    dead.add(client)
            dashboard_clients.difference_update(dead)
        asyncio.run_coroutine_threadsafe(_send(), loop)
    _analysis_registry["broadcast_dashboard"] = _broadcast_file_result
    logger.info("仪表盘广播已注册: 文件分析结果将推送到仪表盘")

    # HTTP 静态文件服务器 (提供 HTML 仪表盘页面)
    http_thread = threading.Thread(
        target=_start_http_server,
        args=(8080,),
        daemon=True,
    )
    http_thread.start()

    # 定时器: 让 Qt 和 asyncio 协同
    def poll_asyncio():
        """让 asyncio 有机会处理回调"""
        pass

    timer = QTimer()
    timer.timeout.connect(poll_asyncio)
    timer.start(10)  # 10ms

    # Qt 主循环
    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        # 清理
        timer.stop()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=3)
        logger.info("后端已停止")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
