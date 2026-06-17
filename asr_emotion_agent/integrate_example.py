"""
后端集成示例 — 展示如何将 OmniEmotionAgent 插入现有 BackendPipeline

这是一个独立的参考代码，演示在 backend/main.py 中需要添加的最小改动。

集成步骤:
  1. 在 BackendPipeline.__init__() 中初始化 OmniEmotionAgent
  2. 在 BackendPipeline._process_frame() 中累积音频 + 触发分析
  3. 处理返回的 EmotionResult → GUI 显示 + 报警

改动量: ~30 行新代码, 不影响现有逻辑
"""

import os
import time
import logging
from collections import deque

import numpy as np

# ★ 导入 Agent 模块
from asr_emotion_agent import OmniEmotionAgent, OmniConfig, EmotionResult

logger = logging.getLogger("backend.integration")


# ============================================================
# 示例: BackendPipeline 修改版 (仅展示新增部分)
# ============================================================

class BackendPipelineWithOmni:
    """
    BackendPipeline 的 Omni 集成版本

    在原有 BackendPipeline 基础上新增:
      - self.omni_agent:  OmniEmotionAgent 实例
      - self._omni_buffer: 音频环形缓冲区
      - _process_omni(): 触发 Omni 分析的方法

    原有代码全部保留, 只添加 ★ 标记的部分。
    """

    def __init__(self, config, gui):
        """
        原始 __init__ 保持不变, 仅追加 Omni 初始化
        """
        # ===== 原有代码 (不变) =====
        self.config = config
        self.gui = gui

        # self.receiver = AudioReceiver(...)
        # self.extractor = FeatureExtractor(...)
        # self.classifier = AudioClassifier(...)
        # self.alert_manager = AlertManager(...)
        # ...

        # ===== ★ 新增: Omni 情绪智能体 =====
        self._init_omni_agent()

        # ===== ★ 新增: 音频缓冲 =====
        self._omni_buffer = np.array([], dtype=np.float32)
        self._omni_sample_rate = 16000
        self._omni_window_sec = 2.0        # 分析窗口 2 秒
        self._omni_trigger_interval = 2.0  # 触发间隔 2 秒
        self._omni_window_samples = int(
            self._omni_sample_rate * self._omni_window_sec
        )
        self._last_omni_trigger_time = 0.0

        logger.info("OmniEmotionAgent 已集成到 BackendPipeline")

    def _init_omni_agent(self):
        """初始化 Omni 情绪智能体"""
        omni_config = OmniConfig()
        omni_config.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        omni_config.window_seconds = self._omni_window_sec
        omni_config.trigger_interval = self._omni_trigger_interval

        # 可选: 从配置文件加载
        # omni_config.load_from_file("agent_config.json")

        self.omni_agent = OmniEmotionAgent(omni_config)

        if not self.omni_agent.client.is_available:
            logger.warning(
                "⚠️ Omni API 不可用! "
                "请设置环境变量 DASHSCOPE_API_KEY 并确保网络可达。"
            )

    def _process_frame(self, frame):
        """
        原始 _process_frame 方法

        在现有逻辑之后追加 Omni 分析流水线
        """
        # ===== 原有代码 (不变) =====
        # self.gui.feed_audio_frame(frame)
        # ... CNN 分类 ...
        # ... 报警 ...

        # ===== ★ 新增: Omni 情绪分析流水线 =====
        self._process_omni_pipeline(frame)

    def _process_omni_pipeline(self, frame):
        """
        Omni 情绪分析流水线 (新增方法)

        1. 累积音频到缓冲区
        2. 每 2 秒触发一次 Omni API 调用
        3. 处理结果 → GUI + 报警
        """
        # ---- 累积音频 ----
        self._omni_buffer = np.concatenate([self._omni_buffer, frame.data])

        # 保留最近 5 秒
        max_samples = self._omni_sample_rate * 5
        if len(self._omni_buffer) > max_samples:
            self._omni_buffer = self._omni_buffer[-max_samples:]

        # ---- 触发条件检查 ----
        now = time.time()
        buffer_ready = len(self._omni_buffer) >= self._omni_window_samples
        interval_ready = (
            now - self._last_omni_trigger_time >= self._omni_trigger_interval
        )

        if not (buffer_ready and interval_ready):
            return

        # ---- 取音频窗口 ----
        audio_window = self._omni_buffer[-self._omni_window_samples:].copy()

        # ---- ★ 调 Omni API ----
        result = self.omni_agent.judge(audio_window)

        # ---- 处理结果 ----
        self._handle_omni_result(result)

        self._last_omni_trigger_time = now

    def _handle_omni_result(self, result: EmotionResult):
        """
        处理 Omni 返回的结果 → GUI 显示 + 报警触发

        这个方法需要根据实际 GUI 结构调整。
        """
        # ---- GUI 显示 ----
        # 示例: 更新 GUI 中的标签
        # self.gui.feed_emotion_result(result)  # 需要在 GUI 中新增此方法
        # self.gui.set_emotion_label(result.emotion_cn, result.danger_level)

        # ---- 日志 ----
        if result.danger_level == "危险":
            logger.warning(
                f"🔴 Omni 检测到危险! | {result.emotion_cn} | {result.reason}"
            )
        elif result.danger_level == "关注":
            logger.info(
                f"🟡 Omni 关注 | {result.emotion_cn} | score={result.danger_score:.2f}"
            )

        # ---- 报警 ----
        if result.danger_level == "危险":
            # 方式1: 使用现有的 AlertManager
            # alert = Alert(
            #     level="warning",
            #     message=f"[Omni] {result.emotion_cn}: {result.reason}",
            #     class_name=result.emotion,
            #     timestamp=result.timestamp,
            # )
            # self.gui.feed_alert(alert)

            # 方式2: 直接触发 GUI 报警
            # self.gui.trigger_danger_alert(result)
            pass

        elif result.danger_level == "关注":
            # 预报警 (可选)
            # alert = Alert(
            #     level="pre_alert",
            #     message=f"[Omni] {result.emotion_cn}: {result.reason}",
            #     class_name=result.emotion,
            #     timestamp=result.timestamp,
            # )
            # self.gui.feed_alert(alert)
            pass

    # ================================================================
    # Omni 相关查询方法 (新增)
    # ================================================================

    def get_omni_stats(self) -> dict:
        """获取 Omni Agent 统计信息"""
        return self.omni_agent.get_stats()

    def get_omni_summary(self) -> str:
        """获取 Omni Agent 统计摘要"""
        return self.omni_agent.get_summary()

    def is_omni_dangerous(self) -> bool:
        """最近是否检测到危险"""
        return self.omni_agent.is_recently_dangerous(window=5)

    def get_omni_recent(self, n: int = 10) -> list:
        """获取最近 N 条 Omni 分析结果"""
        return self.omni_agent.get_recent_results(n)


# ============================================================
# GUI 新增方法 (示例)
# ============================================================

# 在 backend/gui_main_window.py 的 MainWindow 类中新增:

def feed_emotion_result_example(self, result: EmotionResult):
    """
    在 GUI 中显示 Omni 情绪分析结果

    这是一个示例实现，需要根据实际 GUI 布局调整。

    Args:
        result: EmotionResult 对象
    """
    # 更新情绪状态标签
    danger_colors = {
        "危险": "red",
        "关注": "orange",
        "正常": "green",
    }
    color = danger_colors.get(result.danger_level, "gray")

    # 示例: 如果 GUI 有 QLabel 用于显示情绪
    # self.emotion_label.setText(f"情绪: {result.emotion_cn}")
    # self.emotion_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    # 示例: 更新转写文本显示区域
    # self.transcript_text.setPlainText(result.text)

    # 示例: 更新判定理由
    # self.verdict_label.setText(result.reason)

    # 示例: 追加到日志区
    # timestamp = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
    # level_icon = {"危险": "🔴", "关注": "🟡", "正常": "🟢"}
    # self.log_area.append(
    #     f"[{timestamp}] {level_icon.get(result.danger_level, '')} "
    #     f"{result.danger_level} | {result.emotion_cn} | "
    #     f"\"{result.text[:30]}\" | {result.reason}"
    # )

    pass


# ============================================================
# 快速启动示例
# ============================================================

def quick_start_example():
    """
    最简单的使用示例 — 可直接运行测试
    """
    import numpy as np

    print("OmniEmotionAgent 快速启动示例")
    print("=" * 50)

    # 1. 创建 Agent
    agent = OmniEmotionAgent(OmniConfig())

    if not agent.client.is_available:
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY=sk-xxx")
        return

    # 2. 生成测试音频
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (3000 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # 3. 分析
    print(f"\n发送 {len(audio)} 样本 ({duration}s) 到 Omni...")
    result = agent.judge(audio)

    # 4. 输出
    print(f"\n结果:")
    print(f"  转写: {result.text or '(无语音)'}")
    print(f"  情绪: {result.emotion_cn} (置信度: {result.emotion_confidence:.2f})")
    print(f"  危险: {result.danger_level} (评分: {result.danger_score:.2f})")
    print(f"  理由: {result.reason}")
    print(f"  延迟: {result.api_latency_ms:.0f}ms")


if __name__ == "__main__":
    quick_start_example()
