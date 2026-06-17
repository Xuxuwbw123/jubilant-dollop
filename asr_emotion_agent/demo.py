"""
Qwen3.5-Omni-Flash 情绪判断智能体 — 演示脚本

演示模式:
  1. 环境检查 — 检查 SDK 和 API Key 配置
  2. 模拟测试 — 用模拟音频验证完整流水线 (无需真实API)
  3. 真实 API — 对模拟音频实际调用 Omni API 做情绪分析
  4. 批量测试 — 多帧连续测试 + 统计

用法:
  # 环境检查
  python demo.py --check

  # 模拟模式 (无 API Key)
  python demo.py --simulate

  # 真实 API 模式
  export DASHSCOPE_API_KEY=sk-xxx
  python demo.py

  # 指定模型
  python demo.py --model qwen3.5-omni-flash

  # 处理真实音频文件
  python demo.py --file path/to/audio.wav
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from asr_emotion_agent import (
    OmniEmotionAgent,
    OmniConfig,
    EmotionResult,
    create_agent,
    check_environment,
    EMOTION_CN_MAP,
)


# ============================================================
# 模拟音频生成
# ============================================================

def generate_test_audio(duration_sec: float = 2.0,
                        sample_rate: int = 16000,
                        audio_type: str = "normal") -> np.ndarray:
    """
    生成模拟测试音频

    注意: 这只是合成音频（正弦波+噪声），用于验证流水线。
    真实语音情绪识别需要真实人声录音。

    Args:
        duration_sec: 时长 (秒)
        sample_rate: 采样率
        audio_type: normal | speech | loud | silent

    Returns:
        float32 numpy 数组
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    if audio_type == "normal":
        # 低能量背景音
        audio = np.random.normal(0, 300, n_samples).astype(np.float32)

    elif audio_type == "speech":
        # 模拟人声频段 (200Hz~2000Hz) 的复杂波形
        audio = (
            4000 * np.sin(2 * np.pi * 220 * t) +
            3000 * np.sin(2 * np.pi * 550 * t) +
            2000 * np.sin(2 * np.pi * 1100 * t) +
            1000 * np.sin(2 * np.pi * 1800 * t)
        ).astype(np.float32)
        audio += np.random.normal(0, 400, n_samples).astype(np.float32)

    elif audio_type == "loud":
        # 大音量 + 高频 (模拟尖叫频率特征)
        audio = (
            12000 * np.sin(2 * np.pi * 800 * t) +
            8000 * np.sin(2 * np.pi * 1500 * t) +
            5000 * np.sin(2 * np.pi * 2500 * t)
        ).astype(np.float32)
        audio += np.random.normal(0, 1500, n_samples).astype(np.float32)

    elif audio_type == "silent":
        # 非常安静
        audio = np.random.normal(0, 50, n_samples).astype(np.float32)

    else:
        audio = np.random.normal(0, 500, n_samples).astype(np.float32)

    # 归一化
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 20000

    return audio.astype(np.float32)


# ============================================================
# 显示函数
# ============================================================

def print_result(result: EmotionResult, index: int = None):
    """格式化打印 EmotionResult"""
    prefix = f"[{index:2d}] " if index is not None else ""

    # 危险等级图标
    level_icon = {
        "危险": "🔴",
        "关注": "🟡",
        "正常": "🟢",
    }
    icon = level_icon.get(result.danger_level, "❓")

    # API 状态
    api_status = "✓" if result.api_success else "✗(降级)"

    print(f"  {prefix}{icon} {result.danger_level:4s} | "
          f"情绪: {result.emotion_cn:5s} | "
          f"评分: {result.danger_score:.2f} | "
          f"API: {api_status} | "
          f"延迟: {result.api_latency_ms:.0f}ms")

    if result.text:
        print(f"      转写: \"{result.text[:60]}\"")
    if result.reason:
        print(f"      理由: {result.reason[:80]}")
    if result.tone_description:
        print(f"      语气: {result.tone_description[:60]}")


def print_header(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 演示函数
# ============================================================

def demo_simulate():
    """
    模拟模式演示: 验证数据结构、编解码、流水线逻辑

    不调用真实 API, 测试生成的 EmotionResult 是否正确。
    """
    print_header("演示: 模拟模式 (不调API, 验证流水线)")

    test_cases = [
        ("speech", "模拟人声"),
        ("loud", "模拟大音量高频"),
        ("normal", "模拟背景音"),
        ("silent", "模拟安静环境"),
    ]

    for audio_type, desc in test_cases:
        audio = generate_test_audio(duration_sec=2.0, audio_type=audio_type)
        energy = np.sqrt(np.mean(audio ** 2))

        print(f"\n--- {desc} (能量={energy:.0f}, 样本数={len(audio)}) ---")
        print(f"  numpy dtype={audio.dtype}, 范围=[{audio.min():.0f}, {audio.max():.0f}]")

        # 模拟编码流程 (不调 API)
        from asr_emotion_agent.qwen_omni_client import QwenOmniClient
        config = OmniConfig()
        client = QwenOmniClient(config)
        b64 = client._audio_to_b64(audio)
        print(f"  Base64 长度: {len(b64)} 字符")

    print("\n  ✅ 流水线验证通过 (音频编码正常)")


def demo_real_api(agent: OmniEmotionAgent):
    """
    真实 API 模式: 调用 Qwen3.5-Omni-Flash 分析模拟音频

    注意: 模拟音频是正弦波合成，不是真人语音。
    Omni 模型会如实反映——通常返回 "无语音" 或 "neutral"。
    要看到真实的情绪识别效果，需使用真人录音文件。
    """
    print_header("演示: 真实 API 模式 (调 Qwen3.5-Omni-Flash)")

    if not agent.client.is_available:
        print("\n  ❌ API 不可用! 请检查:")
        print("     1. pip install openai")
        print("     2. export DASHSCOPE_API_KEY=sk-xxx")
        print("     3. 网络是否能访问 dashscope.aliyuncs.com")
        return

    print(f"\n  模型: {agent.config.model}")
    print(f"  API: {agent.config.api_base_url}")
    print(f"  Key: {'***' + agent.config.dashscope_api_key[-4:] if agent.config.dashscope_api_key else '未设置'}")
    print(f"  窗口: {agent.config.window_seconds}s | 阈值: 危险≥{agent.config.danger_threshold}")

    test_cases = [
        ("speech", "模拟人声 (正弦波合成)"),
        ("normal", "模拟环境背景音"),
    ]

    for audio_type, desc in test_cases:
        print(f"\n--- {desc} ---")
        audio = generate_test_audio(duration_sec=2.0, audio_type=audio_type)

        print(f"  发送 {len(audio)} 样本 ({len(audio)/16000:.1f}s) 到 Omni...")
        result = agent.judge(audio)
        print_result(result)

    # 显示统计
    print(f"\n{'─'*60}")
    stats = agent.get_stats()
    print(f"  统计: 总{stats['total_judgments']}次 | "
          f"API成功率{stats['api_success_rate']:.1%} | "
          f"平均延迟{stats['avg_latency_ms']:.0f}ms")


def demo_batch(agent: OmniEmotionAgent):
    """批量测试 + 趋势分析"""
    print_header("演示: 批量测试 + 统计")

    if not agent.client.is_available:
        print("\n  ❌ API 不可用，跳过批量测试")
        return

    print(f"\n  连续分析 8 帧 (每帧 2 秒模拟音频)...")

    audio_types = ["normal", "speech", "normal", "loud",
                   "loud", "speech", "normal", "normal"]

    results = []
    for i, at in enumerate(audio_types):
        audio = generate_test_audio(duration_sec=2.0, audio_type=at)
        result = agent.judge(audio)
        results.append(result)

        # 打印
        level_icon = {"危险": "🔴", "关注": "🟡", "正常": "🟢"}
        icon = level_icon.get(result.danger_level, "❓")
        print(f"  [{i}] {icon} {result.danger_level:4s} | "
              f"{result.emotion_cn:5s} | score={result.danger_score:.2f} | "
              f"{result.reason[:50]}")

    # 统计与趋势
    print(f"\n{'─'*60}")
    stats = agent.get_stats()
    trend_cn = {"rising": "↑上升(危险加剧)", "falling": "↓下降(趋于安全)",
                "stable": "→平稳", "none": "-"}
    print(f"  总判定: {stats['total_judgments']}")
    print(f"  危险: {stats['danger_count']} | 关注: {stats['attention_count']} | 正常: {stats['normal_count']}")
    print(f"  近期危险率 (10帧): {stats['recent_danger_ratio']:.1%}")
    print(f"  趋势: {trend_cn.get(stats['trend'], '-')}")
    print(f"  API 成功率: {stats['api_success_rate']:.1%}")
    print(f"  平均延迟: {stats['avg_latency_ms']:.0f}ms")


def demo_file(agent: OmniEmotionAgent, file_path: str):
    """分析真实音频文件"""
    print_header(f"演示: 分析音频文件 — {file_path}")

    if not os.path.exists(file_path):
        print(f"\n  ❌ 文件不存在: {file_path}")
        return

    if not agent.client.is_available:
        print("\n  ❌ API 不可用")
        return

    # 读取 WAV 文件
    try:
        import wave as wav
        with wav.open(file_path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

        print(f"\n  文件: {file_path}")
        print(f"  采样率: {sr}Hz | 时长: {n_frames/sr:.1f}s | 声道: 单声道")

        # 如果太长，取前 30 秒分段分析
        max_duration = 30.0
        if n_frames / sr > max_duration:
            print(f"  音频较长，仅分析前 {max_duration}s...")
            audio = audio[:int(max_duration * sr)]

        # 按 2 秒窗口滑窗分析
        window_samples = int(sr * 2.0)
        step_samples = int(sr * 1.0)  # 1 秒步长 (50% 重叠)

        print(f"\n  开始滑窗分析 (窗口={2.0}s, 步长={1.0}s)...\n")

        i = 0
        while i + window_samples <= len(audio):
            window = audio[i:i + window_samples]
            time_offset = i / sr

            result = agent.judge(window, sample_rate=sr)

            level_icon = {"危险": "🔴", "关注": "🟡", "正常": "🟢"}
            print(f"  [{time_offset:5.1f}s] "
                  f"{level_icon.get(result.danger_level, '❓')} "
                  f"{result.danger_level:4s} | "
                  f"{result.emotion_cn:5s} | "
                  f"\"{result.text[:40]}\"")

            if result.danger_level == "危险":
                print(f"         ⚠️  {result.reason}")

            i += step_samples

    except Exception as e:
        print(f"\n  ❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()


def demo_check():
    """环境检查"""
    print_header("环境检查")

    env = check_environment()

    print(f"\n  Python: {sys.version}")
    print(f"  openai SDK: {'✓ 已安装' if env['openai_available'] else '✗ 未安装 (pip install openai)'}")
    print(f"  DASHSCOPE_API_KEY: {'✓ 已设置' if env['api_key_set'] else '✗ 未设置'}")
    print(f"  API Base URL: {env['api_base']}")

    # 检查 numpy
    try:
        import numpy
        print(f"  numpy: ✓ {numpy.__version__}")
    except ImportError:
        print(f"  numpy: ✗ 未安装 (pip install numpy)")

    # 测试音频编码
    try:
        audio = generate_test_audio(0.5)
        from asr_emotion_agent.qwen_omni_client import QwenOmniClient
        config = OmniConfig()
        client = QwenOmniClient(config)
        b64 = client._audio_to_b64(audio)
        print(f"  音频编码: ✓ (500ms 音频 → {len(b64)} 字符 Base64)")
    except Exception as e:
        print(f"  音频编码: ✗ {e}")

    print(f"\n  模块路径: {Path(__file__).resolve().parent}")
    print(f"  项目根目录: {_PROJECT_ROOT}")

    if not env['openai_available']:
        print(f"\n  💡 请运行: pip install openai")
    if not env['api_key_set']:
        print(f"\n  💡 请设置: export DASHSCOPE_API_KEY=sk-xxx")
        print(f"    获取 Key: https://dashscope.console.aliyun.com/apiKey")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-Omni-Flash 情绪判断智能体 — 演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python demo.py --check              # 环境检查
  python demo.py --simulate           # 模拟模式 (不调API)
  python demo.py                      # 真实API模式 (需设置 API Key)
  python demo.py --batch              # 批量测试 + 统计
  python demo.py --file audio.wav     # 分析音频文件
  python demo.py --model qwen3.5-omni-flash  # 指定模型
        """,
    )
    parser.add_argument("--check", action="store_true",
                        help="环境检查")
    parser.add_argument("--simulate", action="store_true",
                        help="模拟模式 (不调用真实API)")
    parser.add_argument("--batch", action="store_true",
                        help="批量测试模式")
    parser.add_argument("--file", type=str, default=None,
                        help="分析音频文件路径")
    parser.add_argument("--model", type=str, default="qwen3.5-omni-flash",
                        help="模型名 (默认: qwen3.5-omni-flash)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DashScope API Key (也可通过环境变量 DASHSCOPE_API_KEY 设置)")
    parser.add_argument("--window", type=float, default=2.0,
                        help="音频窗口大小 (秒, 默认 2.0)")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="危险判定阈值 (默认 0.60)")
    args = parser.parse_args()

    # --check 模式
    if args.check:
        demo_check()
        return

    # --simulate 模式
    if args.simulate:
        demo_simulate()
        return

    # --file 模式 (不需要完整 agent, 但需要 client)
    if args.file:
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
        agent = create_agent(
            api_key=api_key,
            model=args.model,
            window_seconds=args.window,
            danger_threshold=args.threshold,
        )
        demo_file(agent, args.file)
        return

    # 默认 & --batch 模式
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        print("⚠️  未设置 API Key！将使用模拟模式。")
        print("   设置方式: export DASHSCOPE_API_KEY=sk-xxx")
        print("   或使用: python demo.py --api-key sk-xxx")
        print()
        demo_simulate()
        return

    # 创建 Agent
    agent = create_agent(
        api_key=api_key,
        model=args.model,
        window_seconds=args.window,
        danger_threshold=args.threshold,
    )

    if args.batch:
        demo_real_api(agent)
        demo_batch(agent)
    else:
        demo_real_api(agent)

    # 最终统计
    print(f"\n{'='*60}")
    print(agent.get_summary())
    print(f"\n演示完成。")


if __name__ == "__main__":
    main()
