

import time
import logging
from typing import Optional, List, Tuple
from collections import deque

from common.audio_config import ClassifyResult, Alert, AlertConfig

logger = logging.getLogger("backend.alert")


class AlertManager:
    """
    报警管理器（智能版 v2）

    改进点 vs v1:
    1. 滑动窗口分析（默认15帧），不再要求"连续N帧全异常"
       → 容忍偶发的正常帧穿插，减少漏报
    2. 严重度评分 = 异常占比 × 平均置信度 × 类别权重
       → 尖叫(1.0) 比大笑(0.5) 更容易触发高级别报警
    3. 三级报警: pre_alert(注意) / warning(警告) / critical(危急)
       → 不同级别不同响应的冷却时间和日志级别
    4. 趋势检测: 异常比例是上升还是下降
       → 持续上升会升级报警级别
    5. [v2] 主导类别一致性检查 — 同一异常类别需连续出现才能报警
       → 防止模型在不同异常类之间摇摆产生虚假报警
    """

    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()

        # 滑动窗口：存放最近 N 帧分类结果
        self._recent_results: deque[ClassifyResult] = deque(
            maxlen=self.config.window_size
        )

        # [v2] 主导类别历史 — 跟踪最近几次窗口的主导类别
        self._dominant_class_history: deque[str] = deque(maxlen=3)

        # [v2] 上一个窗口的主导类别 (用于趋势比较)
        self._last_dominant_class: str | None = None

        # 报警历史
        self._alert_history: List[Alert] = []

        # 冷却状态
        self._last_alert_time: float = 0.0
        self._last_alert_level: str = "normal"  # 初始化为 normal，避免空字符串

        # 趋势跟踪
        self._previous_abnormal_ratio: float = 0.0
        self._escalation_counter: int = 0       # 连续触发次数

        # 统计
        self.total_frames: int = 0
        self.total_abnormal: int = 0

    # ═══════════════════════════════════════════════════════════
    # 滑动窗口缓冲管理 + 主入口 — 分类结果入队并触发评估
    # Buffers the last N classification results in a fixed-size deque.
    # On each feed(), appends the new result and — once the minimum fill
    # threshold is met — triggers the full analysis→scoring→alert pipeline.
    # Includes v2 dominant-class consistency gating to suppress alerts
    # when the model flip-flops between different abnormal classes.
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 主入口
    # ================================================================

    def feed(self, result: ClassifyResult) -> Optional[Alert]:
        """
        喂入一个分类结果
        Returns: Alert 或 None
        """
        self.total_frames += 1
        if result.is_abnormal:
            self.total_abnormal += 1

        # 加入滑动窗口
        self._recent_results.append(result)

        # 窗口还不够大时不做判断（至少 1/3 窗口大小或 3 帧）
        min_frames = max(3, self.config.window_size // 3)
        if len(self._recent_results) < min_frames:
            return None

        # --- 1. 分析窗口 ---
        abnormal_ratio, dominant_class, avg_confidence = self._analyze_window()

        # --- [v2] 主导类别一致性检查 ---
        # 如果当前窗口主导类别变化了，重置历史
        if dominant_class != self._last_dominant_class:
            if dominant_class != "normal":
                # 新类别出现，记录它
                self._dominant_class_history.append(dominant_class)
                self._last_dominant_class = dominant_class
            else:
                # 回到 normal，清空历史
                self._dominant_class_history.clear()
                self._last_dominant_class = None
        else:
            # 同一类别持续中
            self._dominant_class_history.append(dominant_class)

        # 需要同一异常类别至少连续主导 2 个窗口才触发报警
        if dominant_class != "normal":
            consistent = (
                len(self._dominant_class_history) >= 2 and
                all(c == dominant_class for c in self._dominant_class_history)
            )
            if not consistent:
                return None  # 类别摇摆，不报警

        # --- 2. 计算严重度评分 ---
        severity_score, class_weight = self._calc_severity(
            abnormal_ratio, avg_confidence, dominant_class
        )

        # --- 3. 检测趋势 ---
        trend = self._detect_trend(abnormal_ratio)

        # --- 4. 判断是否报警 ---
        alert = self._evaluate_alert(
            severity_score, dominant_class, abnormal_ratio,
            avg_confidence, trend
        )

        return alert

    # ═══════════════════════════════════════════════════════════
    # 窗口分析 — 计算异常占比 / 主导类别 / 平均置信度
    #   abnormal_ratio  = fraction of frames marked abnormal
    #   dominant_class  = most frequent abnormal label in window
    #   avg_confidence  = mean confidence across all abnormal frames
    # Only counts frames whose confidence exceeds the configured threshold.
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 窗口分析
    # ================================================================

    def _analyze_window(self) -> Tuple[float, str, float]:
        """
        分析滑动窗口
        Returns:
            abnormal_ratio: 异常帧占比 [0, 1]
            dominant_class: 出现最多的异常类别
            avg_confidence: 异常帧平均置信度
        """
        recent = list(self._recent_results)

        # 筛选有效异常帧（置信度达标才算）
        abnormal_frames = [
            r for r in recent
            if r.is_abnormal and r.confidence >= self.config.confidence_threshold
        ]

        total = len(recent)
        abnormal_count = len(abnormal_frames)
        abnormal_ratio = abnormal_count / total if total > 0 else 0.0

        if abnormal_count == 0:
            return 0.0, "normal", 0.0

        avg_confidence = (
            sum(r.confidence for r in abnormal_frames) / abnormal_count
        )

        # 主导类别（出现最多次的异常类别）
        class_counts: dict = {}
        for r in abnormal_frames:
            class_counts[r.class_name] = class_counts.get(r.class_name, 0) + 1
        dominant_class = max(class_counts, key=class_counts.get)

        return abnormal_ratio, dominant_class, avg_confidence

    # ═══════════════════════════════════════════════════════════
    # 严重度评分 — 三维乘法公式（异常占比 × 置信度 × 类别权重）
    #   severity = abnormal_ratio × avg_confidence × class_weight
    # - abnormal_ratio:  how pervasive the abnormality is [0..1]
    # - avg_confidence:  how certain the AI model is     [0..1]
    # - class_weight:    category multiplier (scream=1.0, cry=0.8,
    #                    sob=0.6, laugh=0.5) — inherent risk ranking
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 严重度评分
    # ================================================================

    def _calc_severity(self, abnormal_ratio: float, avg_confidence: float,
                       class_name: str) -> Tuple[float, float]:
        """
        计算严重度评分

        公式: severity = abnormal_ratio × avg_confidence × class_weight

        三个因子:
        - abnormal_ratio: 窗口内异常占比（越高越严重）
        - avg_confidence: AI 模型的置信度（越确定越可信）
        - class_weight: 类别权重（尖叫1.0 > 大哭0.8 > 抽泣0.6 > 大笑0.5）

        Returns:
            severity_score: 严重度 [0, 1]
            class_weight: 类别权重
        """
        class_weight = self.config.class_severity.get(class_name, 0.5)
        severity = abnormal_ratio * avg_confidence * class_weight
        return severity, class_weight

    # ═══════════════════════════════════════════════════════════
    # 趋势检测 — 上升 / 下降 / 平稳（12% 滞后带）
    #   rising  ← increase > 12 percentage points
    #   falling ← decrease > 12 percentage points
    #   stable  ← change within ±12pp band
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 趋势检测
    # ================================================================

    def _detect_trend(self, current_ratio: float) -> str:
        """
        检测异常比例趋势
        Returns: "rising" | "falling" | "stable"
        """
        prev = self._previous_abnormal_ratio
        self._previous_abnormal_ratio = current_ratio

        if prev <= 0:
            return "stable"

        diff = current_ratio - prev
        if diff > 0.12:
            return "rising"
        elif diff < -0.12:
            return "falling"
        else:
            return "stable"

    # ═══════════════════════════════════════════════════════════
    # 报警评估 — 三级阈值阶梯 + 分级冷却 + 趋势升级
    #   severity >= 0.75 → critical  (longest cooldown)
    #   severity >= 0.50 → warning
    #   severity >= 0.30 → pre_alert (shortest cooldown)
    # Additionally: if abnormal_ratio < min_abnormal_ratio → suppress.
    # Rising trend + 3 consecutive triggers → automatic escalation.
    # Each alert level has its own cooldown period to avoid flooding.
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 报警判断
    # ================================================================

    def _evaluate_alert(self, severity_score: float, dominant_class: str,
                        abnormal_ratio: float, avg_confidence: float,
                        trend: str) -> Optional[Alert]:
        """
        根据严重度评分 + 趋势决定是否报警以及报警级别

        三级阈值:
        - severity >= critical_threshold (0.75) → 危急
        - severity >= warning_threshold  (0.50) → 警告
        - severity >= pre_alert_threshold(0.30) → 预报警
        """

        # 异常占比太低 → 不报警
        if abnormal_ratio < self.config.min_abnormal_ratio:
            self._escalation_counter = max(0, self._escalation_counter - 1)
            return None

        # --- 冷却检查 ---
        now = time.time()
        if self._last_alert_time > 0:
            # 不同级别不同冷却时间
            if self._last_alert_level == "critical":
                cooldown = self.config.critical_cooldown_seconds
            elif self._last_alert_level == "warning":
                cooldown = self.config.cooldown_seconds * 0.7
            else:
                cooldown = self.config.cooldown_seconds

            if now - self._last_alert_time < cooldown:
                return None

        # --- 判定级别 ---
        if severity_score >= self.config.critical_threshold:
            level = "critical"
            self._escalation_counter += 1
        elif severity_score >= self.config.warning_threshold:
            level = "warning"
            self._escalation_counter += 1
        elif severity_score >= self.config.pre_alert_threshold:
            level = "pre_alert"
            self._escalation_counter += 1
        else:
            self._escalation_counter = max(0, self._escalation_counter - 1)
            return None

        # --- 趋势加成：持续恶化则升级 ---
        if trend == "rising" and self._escalation_counter >= 3:
            if level == "pre_alert":
                level = "warning"
                logger.info("趋势恶化: 预报警升级为警告")
            elif level == "warning":
                level = "critical"
                logger.warning("趋势恶化: 警告升级为危急!")

        # --- 创建报警 ---
        alert = self._create_alert(
            level, dominant_class, abnormal_ratio, avg_confidence,
            severity_score, trend
        )

        self._alert_history.append(alert)
        self._last_alert_time = now
        self._last_alert_level = level

        # 分级日志
        if level == "critical":
            logger.error(f"!!! 危急报警: {alert.message}")
        elif level == "warning":
            logger.warning(f"!! 警告: {alert.message}")
        else:
            logger.info(f"! 预报警: {alert.message}")

        return alert

    # ═══════════════════════════════════════════════════════════
    # 报警记录构造 — 中文显示字符串含完整上下文
    # with Chinese-language display strings for level, class, and trend.
    # The message template includes all diagnostic fields inline so the
    # GUI / dashboard can render it without further translation.
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 报警记录构造
    # ================================================================

    def _create_alert(self, level: str, class_name: str,
                      abnormal_ratio: float, avg_confidence: float,
                      severity_score: float, trend: str) -> Alert:
        """创建含详细信息的报警记录"""

        level_cn = {"pre_alert": "预报警", "warning": "警告", "critical": "危急"}
        class_cn = {
            "normal": "正常", "scream": "尖叫",
            "cry": "大哭", "laugh": "大笑",
        }
        trend_cn = {"rising": "↑恶化", "falling": "↓好转", "stable": "→平稳"}

        cn_class = class_cn.get(class_name, class_name)
        cn_level = level_cn.get(level, level)
        cn_trend = trend_cn.get(trend, trend)

        message = (
            f"[{cn_level}] 异常声音: {cn_class} | "
            f"异常占比: {abnormal_ratio:.0%} | "
            f"置信度: {avg_confidence:.2f} | "
            f"严重度: {severity_score:.2f} | "
            f"趋势: {cn_trend}"
        )

        return Alert(
            level=level,
            message=message,
            class_name=class_name,
            timestamp=time.time(),
            frame_index=0,
            severity=round(severity_score, 4),
        )

    # ═══════════════════════════════════════════════════════════
    # 查询与统计接口 — GUI 和仪表盘的只读数据访问
    #   get_recent_alerts()     → last N alert records
    #   get_latest_alert()      → most recent alert or None
    #   get_recent_results()    → current sliding-window snapshot
    #   get_window_summary()    → full diagnostic dict for dashboard
    #   get_stats()             → global counters + cooldown state
    #   reset()                 → clear all state
    # ═══════════════════════════════════════════════════════════
    # ================================================================
    # 查询接口
    # ================================================================

    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        """获取最近 N 条报警"""
        return self._alert_history[-n:]

    def get_latest_alert(self) -> Optional[Alert]:
        """获取最近一条报警"""
        return self._alert_history[-1] if self._alert_history else None

    def get_recent_results(self) -> List[ClassifyResult]:
        """获取滑动窗口快照"""
        return list(self._recent_results)

    def get_window_summary(self) -> dict:
        """
        获取当前窗口摘要（供 GUI 显示详细状态）
        """
        abnormal_ratio, dominant_class, avg_confidence = self._analyze_window()
        severity_score, class_weight = self._calc_severity(
            abnormal_ratio, avg_confidence, dominant_class
        )
        trend = self._detect_trend(abnormal_ratio)

        # 当前风险等级
        if severity_score >= self.config.critical_threshold:
            risk_level = "critical"
        elif severity_score >= self.config.warning_threshold:
            risk_level = "warning"
        elif severity_score >= self.config.pre_alert_threshold:
            risk_level = "pre_alert"
        else:
            risk_level = "normal"

        return {
            "window_size": self.config.window_size,
            "abnormal_ratio": round(abnormal_ratio, 3),
            "dominant_class": dominant_class,
            "avg_confidence": round(avg_confidence, 3),
            "severity_score": round(severity_score, 3),
            "class_weight": class_weight,
            "trend": trend,
            "risk_level": risk_level,
            "escalation_counter": self._escalation_counter,
        }

    def get_stats(self) -> dict:
        """获取全局统计"""
        return {
            "total_frames": self.total_frames,
            "total_abnormal": self.total_abnormal,
            "abnormal_rate": (
                self.total_abnormal / self.total_frames
                if self.total_frames > 0 else 0
            ),
            "total_alerts": len(self._alert_history),
            "last_level": self._last_alert_level,
            "in_cooldown": bool(
                self._last_alert_time > 0 and
                time.time() - self._last_alert_time < self.config.cooldown_seconds
            ),
            "window_summary": self.get_window_summary(),
        }

    def reset(self):
        """重置所有状态"""
        self._recent_results.clear()
        self._dominant_class_history.clear()
        self._last_dominant_class = None
        self._alert_history.clear()
        self._last_alert_time = 0.0
        self._last_alert_level = "normal"
        self._previous_abnormal_ratio = 0.0
        self._escalation_counter = 0
        self.total_frames = 0
        self.total_abnormal = 0
        logger.info("报警管理器已重置")
