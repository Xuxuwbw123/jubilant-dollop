# -*- coding: utf-8 -*-
"""
API Key 验证脚本 - 测试 Qwen3.5-Omni-Flash 连通性和情绪分析能力

Key 从 agent_config.json 读取，不会出现在命令行中，安全。
"""

import sys
import os
import io
import time
import numpy as np

# 将项目根目录加入 Python 路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from asr_emotion_agent import OmniConfig, OmniEmotionAgent, EmotionResult

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
def main():
    print("=" * 55)
    print("  Qwen3.5-Omni-Flash 连通性验证")
    print("=" * 55)

    # 1. 加载配置 (从 agent_config.json 读 API Key)
    config = OmniConfig()
    config.load_from_env()
    config.load_from_file()  # 读 agent_config.json

    if not config.dashscope_api_key:
        print("\n[FAIL] API Key 未找到!")
        print("   请确保 agent_config.json 中已设置 dashscope_api_key")
        return

    print(f"\n[OK] API Key: ***{config.dashscope_api_key[-4:]}")
    print(f"[OK] 模型: {config.model}")
    print(f"[OK] API: {config.api_base_url}")

    # 2. 创建 Agent
    print("\n初始化 OmniEmotionAgent...")
    agent = OmniEmotionAgent(config)

    if not agent.client.is_available:
        print("[FAIL] API 客户端初始化失败")
        return
    print("[OK] 客户端就绪")

    # 3. 生成一段模拟语音音频 (2秒)
    sr = 16000
    duration = 2.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # 混合多频段模拟人声
    audio = (
        5000 * np.sin(2 * np.pi * 250 * t) +
        3000 * np.sin(2 * np.pi * 600 * t) +
        2000 * np.sin(2 * np.pi * 1200 * t)
    ).astype(np.float32)
    audio += np.random.normal(0, 500, n).astype(np.float32)
    audio = audio / np.max(np.abs(audio)) * 15000

    print(f"\n测试音频: {duration}s, {n}样本, 16kHz, float32")

    # 4. 调 Omni API
    print("\n发送到 Qwen3.5-Omni-Flash...")
    t0 = time.time()
    result = agent.judge(audio)
    elapsed = time.time() - t0

    # 5. 结果
    print(f"\n{'─' * 55}")
    print(f"  结果 (耗时 {elapsed:.1f}s)")
    print(f"{'─' * 55}")

    if result.api_success:
        print(f"  API状态:  [OK] 成功")
        print(f"  API延迟:  {result.api_latency_ms:.0f}ms")
    else:
        print(f"  API状态:  [FAIL] 失败")
        print(f"  错误信息: {result.error_message[:100]}")

    print(f"  转写文本: \"{result.text or '(未检测到语音)'}\"")
    print(f"  情绪标签: {result.emotion} ({result.emotion_cn})")
    print(f"  情绪置信度: {result.emotion_confidence:.2f}")
    print(f"  危险等级: {result.danger_level}")
    print(f"  危险评分: {result.danger_score:.2f}")
    print(f"  语气描述: {result.tone_description or '(无)'}")
    print(f"  判定理由: {result.reason or '(无)'}")
    print(f"  匹配敏感词: {result.keywords_detected or '无'}")

    print(f"\n{'─' * 55}")
    if result.api_success:
        print("  [PASS] 验证通过! Qwen3.5-Omni-Flash 连通正常")
    else:
        print("  [FAIL] 验证失败! 请检查网络和 API Key")

    if result.raw_response:
        print(f"  原始响应(前200字): {result.raw_response[:200]}")
    print(f"{'─' * 55}")


if __name__ == "__main__":
    main()
