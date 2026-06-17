# -*- coding: utf-8 -*-
"""真实录音批量测试 - 结果输出到 测试结果.txt"""
import sys, os, io, wave, time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.signal import resample_poly
from asr_emotion_agent import OmniConfig, OmniEmotionAgent

# 读 Key
key_file = os.path.join(os.path.dirname(__file__), '.key')
with open(key_file) as f:
    api_key = f.read().strip()

config = OmniConfig()
config.dashscope_api_key = api_key
agent = OmniEmotionAgent(config)

test_dir = os.path.join(_PROJECT_ROOT, '测试用例')
out_path = os.path.join(_PROJECT_ROOT, '测试结果.txt')
files = sorted([f for f in os.listdir(test_dir) if f.endswith('.wav')])

with open(out_path, 'w', encoding='utf-8') as out:
    def p(text=''):
        print(text)
        out.write(text + '\n')

    p('=' * 65)
    p(f'  Qwen3.5-Omni-Flash 真实录音情绪分析')
    p(f'  模型: qwen3.5-omni-flash')
    p(f'  文件数: {len(files)}')
    p(f'  时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    p('=' * 65)

    total = danger = attention = normal = 0

    for idx, fname in enumerate(files):
        fpath = os.path.join(test_dir, fname)
        with wave.open(fpath, 'rb') as wf:
            sr_orig = wf.getframerate()
            n_frames = wf.getnframes()
            audio_raw = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32)

        if sr_orig != 16000:
            audio = resample_poly(audio_raw, 16000, sr_orig).astype(np.float32)
        else:
            audio = audio_raw

        target = 32000
        if len(audio) > target:
            start = max(0, (len(audio) - target) // 2)
            audio = audio[start:start + target]
        elif len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))

        dur = n_frames / sr_orig
        p(f'\n{"─"*65}')
        p(f'  [{idx+1}/{len(files)}] {fname} ({dur:.1f}s)')
        p(f'{"─"*65}')

        result = agent.judge(audio)
        total += 1

        icon = {'危险': '!!', '关注': '! ', '正常': '  '}
        if result.danger_level == '危险':
            danger += 1
        elif result.danger_level == '关注':
            attention += 1
        else:
            normal += 1

        p(f'  {icon.get(result.danger_level,"?")} [{result.danger_level}]  '
          f'情绪={result.emotion_cn}  评分={result.danger_score:.2f}  '
          f'延迟={result.api_latency_ms:.0f}ms')
        p(f'  转写: "{result.text or "(无)"}"')
        p(f'  语气: {result.tone_description}')
        p(f'  置信: {result.emotion_confidence:.2f}')
        p(f'  理由: {result.reason}')

    p(f'\n{"="*65}')
    p(f'  测试汇总')
    p(f'{"="*65}')
    p(f'  总计: {total}  |  !!危险: {danger}  |  !关注: {attention}  |  正常: {normal}')
    if total > 0:
        p(f'  危险率: {danger/total*100:.0f}%')
    stats = agent.get_stats()
    p(f'  API成功率: {stats["api_success_rate"]:.0%}')
    p(f'  平均延迟: {stats["avg_latency_ms"]:.0f}ms')
    p(f'{"="*65}')

print(f'\n结果已保存到: {out_path}')
