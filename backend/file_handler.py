

import json
import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger("backend.file_handler")

# 支持的格式
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
SUPPORTED_VIDEO = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".wmv"}


# ═══════════════════════════════════════════════════════════
# 音频提取 — 三级降级解码（librosa → soundfile → ffmpeg 子进程）
#   1. librosa  (handles most formats incl. video via ffmpeg backend)
#   2. soundfile  (pure-audio fallback for WAV/FLAC/OGG)
#   3. ffmpeg subprocess  (last resort for video containers;
#      pipes decoded WAV through stdin, no temp file needed)
# Resamples everything to target_sr (default 16000 Hz) in mono.
# ═══════════════════════════════════════════════════════════

def extract_audio(filepath: str | Path, target_sr: int = 16000) -> np.ndarray | None:
    """
    从音视频文件中提取音频，重采样到 target_sr，返回 float32 数组。
    优先用 librosa（支持 ffmpeg 后端），失败则尝试 soundfile。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        return None

    suffix = filepath.suffix.lower()

    # ---- 尝试 librosa (支持视频, 需要 ffmpeg) ----
    try:
        import librosa
        audio, sr = librosa.load(str(filepath), sr=target_sr, mono=True)
        logger.info(f"librosa 加载成功: {len(audio)/sr:.1f}s, sr={sr}")
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"librosa 加载失败 ({e})，尝试 soundfile...")

    # ---- 纯音频回退: soundfile ----
    if suffix in SUPPORTED_AUDIO:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(filepath), dtype="float32", always_2d=False)
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            logger.info(f"soundfile 加载成功: {len(audio)/target_sr:.1f}s")
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"soundfile 也失败: {e}")
            # M4A/AAC/MP4 等格式需要 ffmpeg 解码，soundfile 不支持
            if suffix in {'.m4a', '.aac', '.mp4', '.m4b', '.webm'}:
                logger.error(
                    f"Cannot decode {suffix}: this format requires ffmpeg. "
                    f"Install ffmpeg and add to PATH, or use WAV/FLAC instead. "
                    f"Download: https://ffmpeg.org/download.html"
                )
            return None

    # ---- 视频回退: ffmpeg 子进程 ----
    if suffix in SUPPORTED_VIDEO:
        try:
            import subprocess
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-i", str(filepath),
                "-ar", str(target_sr), "-ac", "1",
                "-f", "wav", tmp.name,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            import soundfile as sf
            audio, sr = sf.read(tmp.name, dtype="float32")
            os.unlink(tmp.name)
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            logger.info(f"ffmpeg 提取成功: {len(audio)/target_sr:.1f}s")
            return audio.astype(np.float32)
        except Exception as e:
            logger.error(f"ffmpeg 提取失败: {e}")
            return None

    logger.error(f"不支持的格式: {suffix}")
    return None


def analyze_file(
    filepath: str | Path,
    classifier,       # AudioClassifier
    extractor,        # FeatureExtractor
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    classifier_mode: str = "nonverbal",  # "nonverbal" | "emotion"
) -> dict:
    """
    分析整个文件，返回综合结果。

    流程:
    1. 提取音频
    2. 滑动窗口 (window_sec 秒, 步长 step_sec)
    3. 逐窗口分类
    4. 汇总统计

    Returns:
        {
            "file": str,
            "duration_sec": float,
            "total_windows": int,
            "dominant_emotion": str,
            "confidence": float,
            "emotion_distribution": {emotion: ratio},
            "timeline": [{start_s, end_s, emotion, confidence}, ...],
            "summary": str,
        }
    """
    filepath = Path(filepath)
    audio = extract_audio(filepath)
    if audio is None:
        suffix = filepath.suffix.lower()
        hint = ""
        if suffix in {'.m4a', '.aac', '.mp4', '.m4b', '.webm'}:
            hint = (
                " (此格式需要 ffmpeg 解码。"
                "请在终端运行: winget install ffmpeg  "
                "或从 https://ffmpeg.org/download.html 下载并添加到 PATH)"
            )
        return {"error": f"无法加载文件: {filepath}{hint}"}

    sr = 16000
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)
    total_samples = len(audio)

    if total_samples < window_samples:
        # 文件太短: 补零
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total_samples] = audio
        audio = padded
        total_samples = window_samples

    # 计算窗口数
    n_windows = max(1, (total_samples - window_samples) // step_samples + 1)

    # ═══════════════════════════════════════════════════════════
    # 文件分析流水线 — 滑动窗口分类 + 情绪标签映射
    # Steps through the audio with window_sec / step_sec stride.
    # Each window: RMS-gate silence (< 0.003), then run the classifier
    # (CNN 3-channel mel or emotion2vec depending on mode/model).
    # Emotion mode applies a 9→5 class mapping (angry/fearful→scream, etc.).
    # ═══════════════════════════════════════════════════════════

    timeline = []
    emotions_counter = Counter()
    confidence_sum = {}

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        window = audio[start:end].astype(np.float32)

        # 静音跳过
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < 0.003:
            timeline.append({
                "start_s": round(start / sr, 1),
                "end_s": round(end / sr, 1),
                "emotion": "silence",
                "confidence": 1.0,
            })
            continue

        # 推理 (按 mode 分发)
        if classifier_mode == 'emotion':
            result = classifier.predict_emotion_from_audio(window, frame_index=i)
        elif hasattr(classifier, '_detected_type') and 'emotion2vec' in classifier._detected_type:
            result = classifier.predict_from_audio(window, frame_index=i)
        else:
            # CNN: 特征提取 + 时间维度对齐
            features = extractor.extract_3channel_mel(window)
            target_frames = 32
            current_frames = features.shape[2]
            if current_frames < target_frames:
                features = np.pad(features, ((0,0),(0,0),(0,target_frames-current_frames)), mode='edge')
            elif current_frames > target_frames:
                features = features[:, :, :target_frames]
            result = classifier.predict(features, frame_index=i)

        raw_emotion = result.class_name
        conf = result.confidence

        # ★ 情绪模式: 9 类 → 5 类映射 (与 BackendPipeline 保持一致)
        if classifier_mode == 'emotion':
            _EMO_MAP = {
                "angry": "scream", "fearful": "scream",
                "sad": "cry", "happy": "laugh",
                "disgusted": "cry",
                "surprised": "normal", "neutral": "normal",
                "other": "normal", "unk": "normal",
            }
            emotion = _EMO_MAP.get(raw_emotion, "normal")
        else:
            emotion = raw_emotion

        timeline.append({
            "start_s": round(start / sr, 1),
            "end_s": round(end / sr, 1),
            "emotion": emotion,
            "confidence": round(conf, 4),
            "raw_emotion": raw_emotion if classifier_mode == 'emotion' else None,
        })
        emotions_counter[emotion] += 1
        if emotion not in confidence_sum:
            confidence_sum[emotion] = []
        confidence_sum[emotion].append(conf)

    # ═══════════════════════════════════════════════════════════
    # 结果聚合 — 汇总分布 / 时间线 / 中文摘要
    #   distribution  = per-class fraction across all windows
    #   timeline      = per-window {start_s, end_s, emotion, confidence}
    #   summary       = Chinese natural-language summary string
    # Dominant emotion determined by majority vote (excluding silence).
    # ═══════════════════════════════════════════════════════════

    # 汇总
    total_classified = sum(emotions_counter.values())
    if total_classified == 0:
        return {
            "file": str(filepath),
            "duration_sec": round(total_samples / sr, 1),
            "total_windows": n_windows,
            "dominant_emotion": "silence",
            "confidence": 1.0,
            "emotion_distribution": {"silence": 1.0},
            "timeline": timeline,
            "summary": "整个文件为静音/极低音量。",
        }

    # 主导情绪 (出现次数最多的异常情绪，或 normal)
    # 去掉 silence
    classified = [(e, c) for e, c in emotions_counter.items() if e != "silence"]
    if not classified:
        dominant = "silence"
        dom_conf = 1.0
    else:
        # 如果有异常情绪占比 > 20%, 取最频繁的
        dominant, dom_count = max(classified, key=lambda x: x[1])
        dom_ratio = dom_count / total_classified
        avg_conf = np.mean(confidence_sum.get(dominant, [0.5]))
        dom_conf = round(float(avg_conf), 4)

    # 情绪分布
    distribution = {}
    for emotion in ["normal", "scream", "cry", "laugh", "silence"]:
        count = emotions_counter.get(emotion, 0)
        distribution[emotion] = round(count / n_windows, 4) if n_windows > 0 else 0.0

    # 生成摘要
    dom_ratio = distribution.get(dominant, 0)
    if dominant == "normal" or dominant == "silence":
        summary = f"该文件主要为正常声音/静音 ({dom_ratio:.0%})，未检测到异常情绪。"
    elif dom_ratio >= 0.5:
        emo_cn = {"scream": "尖叫/愤怒", "cry": "大哭/悲伤", "laugh": "大笑/开心"}
        summary = f"该文件主要情绪为【{emo_cn.get(dominant, dominant)}】，占比 {dom_ratio:.0%}，平均置信度 {dom_conf:.2%}。"
    elif dom_ratio >= 0.2:
        emo_cn = {"scream": "尖叫/愤怒", "cry": "大哭/悲伤", "laugh": "大笑/开心"}
        summary = f"该文件包含部分【{emo_cn.get(dominant, dominant)}】(占比 {dom_ratio:.0%})，但整体情绪较为混合。"
    else:
        summary = f"该文件情绪较为混合，无单一主导情绪。分布: {distribution}"

    return {
        "file": str(filepath),
        "duration_sec": round(total_samples / sr, 1),
        "total_windows": n_windows,
        "dominant_emotion": dominant,
        "confidence": dom_conf,
        "emotion_distribution": distribution,
        "timeline": timeline,
        "summary": summary,
    }
