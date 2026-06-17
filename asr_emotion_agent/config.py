"""
OmniConfig — Qwen3.5-Omni-Flash 情绪判断智能体配置

配置优先级:
  1. 代码默认值
  2. 配置文件 agent_config.json
  3. 环境变量 (DASHSCOPE_API_KEY 等)
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("asr-emotion-agent.config")


# ============================================================
# System Prompt — 告诉 Omni 模型如何分析音频
# ============================================================

SYSTEM_PROMPT = """你是一个专业的安全监控语音分析系统。你的任务是对输入的音频片段进行分析，判断是否存在危险情况。

## 分析维度
1. **转写**: 将音频中的语音内容转写为文字（如果没有人声则标注为空字符串）
2. **情绪识别**: 判断说话者的情绪状态，从以下类别中选择最匹配的一个：
   - fearful (恐惧) — 呼救、害怕、受到威胁
   - angry (愤怒) — 争吵、威胁、暴力倾向
   - sad (悲伤) — 哭泣、抽泣、难过
   - disgusted (厌恶) — 反感、被骚扰
   - surprised (惊讶) — 突然惊吓、意外
   - happy (开心) — 笑声、愉快交谈
   - neutral (中性) — 日常对话、平静、无语音
3. **语气描述**: 简要描述说话者的语气特征（如"声音颤抖、音量高、语速急促"、"语气平静"等）
4. **危险判定**: 综合语音内容、情绪和语气，判断是否存在危险：
   - "危险" — 听到呼救、暴力威胁、极度恐惧、激烈争吵、有人身安全风险
   - "关注" — 哭泣抽泣、异常语气、可能有异常但不确定
   - "正常" — 日常对话、笑声、安静无声
5. **判定理由**: 用一句话解释判定依据

## 输出格式（必须严格遵守，只输出JSON，不要有任何前缀或后缀）
{
  "text": "转写的文字内容",
  "emotion": "fearful",
  "emotion_confidence": 0.92,
  "tone_description": "声音颤抖，音量高，语速急促",
  "danger_level": "危险",
  "danger_score": 0.90,
  "reason": "检测到恐惧呼救声，语音内容包含求救信号"
}

## 重要
- 如果音频中没有人声，text 设为 ""，emotion 设为 "neutral"，danger_level 设为 "正常"
- danger_score 为 0.0 到 1.0 之间的数值，越高越危险
- emotion_confidence 为情绪判断的置信度 0.0 到 1.0
- 只输出 JSON，不要有任何其他内容"""


# ============================================================
# 情绪 → 危险评分 本地映射（模型返回 JSON 解析失败时的降级方案）
# ============================================================

EMOTION_DANGER_MAP: dict[str, float] = {
    "fearful":    0.90,
    "angry":      0.80,
    "disgusted":  0.50,
    "surprised":  0.45,
    "sad":        0.40,
    "neutral":    0.08,
    "happy":      0.05,
}

# 情绪英文 → 中文
EMOTION_CN_MAP: dict[str, str] = {
    "fearful":    "恐惧",
    "angry":      "愤怒",
    "sad":        "悲伤",
    "disgusted":  "厌恶",
    "surprised":  "惊讶",
    "happy":      "开心",
    "neutral":    "中性",
}

# 有效的情绪标签集合
VALID_EMOTIONS = set(EMOTION_DANGER_MAP.keys())


@dataclass
class OmniConfig:
    """Qwen3.5-Omni-Flash 情绪判断智能体配置"""

    # ---- 模型 ----
    model: str = "qwen3.5-omni-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_api_key: str = ""

    # ---- 音频 ----
    sample_rate: int = 16000
    window_seconds: float = 2.0           # 每次分析的音频窗口 (秒)
    trigger_interval: float = 2.0         # 触发间隔 (秒)

    # ---- API 参数 ----
    timeout: float = 15.0                 # HTTP 超时 (秒), Omni 模型推理较慢
    retry_count: int = 1                  # 失败重试次数
    max_tokens: int = 500                 # 最大输出 token 数
    temperature: float = 0.1              # 温度参数 (越低越确定)

    # ---- 判定阈值 ----
    danger_threshold: float = 0.60        # >= 此值 → "危险"
    attention_threshold: float = 0.30     # >= 此值 → "关注", < 此值 → "正常"

    # ---- 降级 ----
    fallback_on_failure: bool = True      # API 失败时降级到"正常"

    # ---- System Prompt ----
    system_prompt: str = SYSTEM_PROMPT

    # ---- 日志 ----
    log_level: str = "INFO"

    # ---- 历史记录 ----
    max_history: int = 50                 # 保留最近 N 条结果

    # ================================================================
    # 方法
    # ================================================================

    def prompt_for_key(self):
        """终端交互式输入 API Key（无密钥时弹出询问）"""
        if self.dashscope_api_key and not self.dashscope_api_key.startswith("sk-"):
            self.dashscope_api_key = ""
        if self.dashscope_api_key:
            return
        print()
        print("  ╔══════════════════════════════════════╗")
        print("  ║   Omni 智能体需要 DashScope API Key  ║")
        print("  ║   获取: https://dashscope.console.aliyun.com/apiKey  ║")
        print("  ╚══════════════════════════════════════╝")
        print()
        try:
            key = input("  请输入 API Key (回车跳过则禁用 Omni): ").strip()
            if key:
                self.dashscope_api_key = key
                self._save_key_to_file(key)
            else:
                print("  Omni 智能体已禁用（无 API Key）")
                print()
        except (EOFError, KeyboardInterrupt):
            print()
            print("  Omni 智能体已禁用（输入被取消）")
            print()

    def _save_key_to_file(self, key: str):
        """保存 API Key 到 agent_config.json"""
        import json
        config_path = Path(__file__).parent / "agent_config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"dashscope_api_key": key}, f, ensure_ascii=False, indent=2)
            logger.info("API Key 已保存到 agent_config.json（下次启动无需重新输入）")
        except Exception as e:
            logger.debug(f"无法保存 API Key 到文件: {e}")

    def load_from_env(self):
        """从环境变量加载敏感配置"""
        if key := os.getenv("DASHSCOPE_API_KEY"):
            self.dashscope_api_key = key
        if model := os.getenv("OMNI_MODEL"):
            self.model = model
        if url := os.getenv("OMNI_API_BASE_URL"):
            self.api_base_url = url
        if timeout := os.getenv("OMNI_TIMEOUT"):
            try:
                self.timeout = float(timeout)
            except ValueError:
                pass

    def load_from_file(self, filepath: str = None):
        """从 JSON 配置文件加载"""
        if filepath is None:
            filepath = Path(__file__).parent / "agent_config.json"
        path = Path(filepath)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            logger.info(f"配置已从 {filepath} 加载")
        except Exception as e:
            logger.warning(f"配置文件加载失败: {filepath} — {e}")

    def validate(self) -> list[str]:
        """校验配置，返回问题列表"""
        issues = []
        if not self.dashscope_api_key:
            issues.append("dashscope_api_key 未设置（请设置环境变量 DASHSCOPE_API_KEY）")
        if self.window_seconds < 0.5:
            issues.append(f"window_seconds={self.window_seconds} 太小，建议 >= 1.0")
        if self.window_seconds > 300:
            issues.append(f"window_seconds={self.window_seconds} 太大，建议 <= 10.0")
        if self.danger_threshold <= self.attention_threshold:
            issues.append(
                f"danger_threshold({self.danger_threshold}) 必须大于 "
                f"attention_threshold({self.attention_threshold})"
            )
        if self.timeout < 3.0:
            issues.append(f"timeout={self.timeout}s 太短，Omni 推理需要较长时间，建议 >= 10s")
        return issues

    def to_dict(self, hide_key: bool = True) -> dict:
        """导出为字典"""
        d = {}
        for k, v in self.__dict__.items():
            if k == "system_prompt":
                d[k] = v[:80] + "..." if len(v) > 80 else v
            elif k == "dashscope_api_key" and hide_key and v:
                d[k] = "***" + v[-4:]
            else:
                d[k] = v
        return d
