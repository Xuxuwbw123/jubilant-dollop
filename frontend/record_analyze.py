"""
前端分析工具 — 支持 CNN 和 Omni 两种分析模式
用法:
    # 模式1: 实时录音分析 (默认 CNN)
    python frontend/record_analyze.py                     # 录音3秒
    python frontend/record_analyze.py --record --duration 5  # 录音5秒
    python frontend/record_analyze.py --record --loop        # 循环录音
    python frontend/record_analyze.py --record --device 15   # 指定麦克风

    # 模式2: 直接传文件分析
    python frontend/record_analyze.py --file recording.wav
    python frontend/record_analyze.py --file ../测试用例/大笑2.wav
    python frontend/record_analyze.py --file audio.mp3 --server http://192.168.1.100:8080

    # Omni 模式: 云端智能体情绪分析 (Qwen3.5-Omni-Flash)
    python frontend/record_analyze.py --file test.wav --omni
    python frontend/record_analyze.py --file 尖叫1.wav --omni
    python frontend/record_analyze.py --record --duration 5 --omni
    python frontend/record_analyze.py --record --omni --loop
"""

import sys
import os
import time
import wave
import tempfile
import argparse
from pathlib import Path

import numpy as np
import requests

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.audio_config import AudioConfig, VADConfig


def record_audio(duration_sec: float, device_index: int = None) -> np.ndarray:
    """录音指定时长，返回完整的 float32 音频数组"""
    from frontend.audio_capture import AudioCapture

    audio_config = AudioConfig(
        sample_rate=16000, channels=1,
        sample_width=2, frame_duration_ms=30,
    )
    vad_config = VADConfig(enabled=False)

    capture = AudioCapture(
        audio_config=audio_config,
        vad_config=vad_config,
        simulate=False,
        input_device_index=device_index,
    )

    frame_duration = capture.config.frame_duration_ms / 1000.0
    total_frames = int(duration_sec / frame_duration)
    frames = []

    if not capture.start():
        print("[ERROR] Failed to start microphone")
        return None

    # 显示设备信息
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        idx = device_index or pa.get_default_input_device_info()["index"]
        info = pa.get_device_info_by_index(idx)
        print(f"\n  Device:    [{idx}] {info['name']}")
        pa.terminate()
    except Exception:
        pass

    print(f"  [Recording] {duration_sec}s...")

    try:
        for i in range(total_frames):
            frame = capture.get_frame()
            if frame is None:
                break
            data = frame.data.copy()
            frames.append(data)

            # 实时音量条
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            vol = min(int(rms / 200), 20)
            elapsed = (i + 1) * frame_duration
            pct = min(elapsed / duration_sec, 1.0)
            bar_len = 15
            filled = int(bar_len * pct)
            bar = "#" * filled + "-" * (bar_len - filled)
            vol_bar = "|" * vol
            print(f"\r  [{bar}] {elapsed:.1f}s  vol:{vol_bar:<20s} {rms:6.0f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n  Recording interrupted")
    finally:
        capture.stop()

    if not frames:
        print("\n[ERROR] No audio captured")
        return None

    audio = np.concatenate(frames)
    rms_total = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.abs(audio).max())
    print(f"\n  Captured: {len(audio)/16000:.1f}s | RMS={rms_total:.0f} | Peak={peak:.0f}")
    if rms_total < 50:
        print(f"  [WARNING] Volume very low (RMS={rms_total:.0f}) — check microphone!")
    return audio


def save_wav(audio: np.ndarray, filepath: str, sample_rate: int = 16000):
    """将 float32 音频保存为 16-bit WAV 文件"""
    peak = np.abs(audio).max()
    if peak > 0 and peak < 0.01:
        audio = audio * (0.8 / peak)
    elif peak > 32767:
        audio = audio * (32767 / peak)

    audio_int16 = np.clip(audio, -32768, 32767).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def send_to_backend(filepath: str, server_url: str) -> dict:
    """发送文件到后端 /api/analyze，返回分析结果"""
    url = f"{server_url}/api/analyze"
    try:
        with open(filepath, "rb") as f:
            files = {"file": (Path(filepath).name, f)}
            resp = requests.post(url, files=files, timeout=120)
        if resp.status_code != 200:
            return {"error": f"Server returned {resp.status_code}: {resp.text[:200]}"}
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend: {server_url}"}
    except Exception as e:
        return {"error": str(e)}


def print_result(result: dict):
    """格式化打印分析结果"""
    if "error" in result:
        print(f"\n  [ERROR] {result['error']}\n")
        return

    print(f"\n  {'='*55}")
    print(f"  Analysis Result")
    print(f"  {'='*55}")
    print(f"  File:       {Path(result.get('file', '?')).name}")
    print(f"  Duration:   {result.get('duration_sec', 0):.1f}s")
    print(f"  Windows:    {result.get('total_windows', 0)}")
    print(f"  {'-'*45}")
    print(f"  Dominant:   {result.get('dominant_emotion', '?')}")
    print(f"  Confidence: {result.get('confidence', 0):.2%}")
    print(f"  {'-'*45}")

    # 情绪分布
    print(f"\n  Distribution:")
    distribution = result.get("emotion_distribution", {})
    for emo in ["normal", "scream", "cry", "laugh", "silence"]:
        ratio = distribution.get(emo, 0)
        bar_len = int(ratio * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"    {emo:>8s}  {bar}  {ratio:.0%}")

    # 摘要
    summary = result.get("summary", "")
    if summary:
        print(f"\n  >> {summary}")

    # 时间线（只显示异常段）
    timeline = result.get("timeline", [])
    abnormal = [t for t in timeline if t.get("emotion", "normal") not in ("normal", "silence")]
    if abnormal:
        print(f"\n  Abnormal Segments:")
        for t in abnormal[:12]:
            emo = t["emotion"]
            conf = t["confidence"]
            print(f"    {t['start_s']:>5.1f}s - {t['end_s']:>5.1f}s  {emo:>8s}  ({conf:.0%})")
        if len(abnormal) > 12:
            print(f"    ... and {len(abnormal) - 12} more segments")

    print(f"\n  {'='*55}\n")


# ============================================================
# Omni 模式: 调用 POST /api/omni (Omni 智能体情绪分析)
# ============================================================

def send_to_omni_api(filepath: str, server_url: str) -> dict:
    """发送文件到 POST /api/omni，返回 Omni 情绪分析结果"""
    url = f"{server_url}/api/omni"
    try:
        with open(filepath, "rb") as f:
            files = {"file": (Path(filepath).name, f)}
            resp = requests.post(url, files=files, timeout=120)
        if resp.status_code != 200:
            return {"error": f"Server returned {resp.status_code}: {resp.text[:200]}"}
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend: {server_url}"}
    except Exception as e:
        return {"error": str(e)}


def print_omni_result(result: dict):
    """格式化打印 Omni 情绪分析结果"""
    if "error" in result:
        print(f"\n  [ERROR] {result['error']}\n")
        return

    danger_icon = {"危险": "🔴", "关注": "🟡", "正常": "🟢"}
    icon = danger_icon.get(result.get("danger_level", ""), "⚪")

    print(f"\n  {'='*55}")
    print(f"  Omni 情绪分析结果")
    print(f"  {'='*55}")
    print(f"  文件:       {Path(result.get('file','?')).name}")
    print(f"  时长:       {result.get('duration_sec', 0):.1f}s")
    print(f"  {'-'*45}")
    print(f"  情绪:       {result.get('emotion_cn','?')} ({result.get('emotion_confidence',0):.0%})")
    print(f"  危险等级:   {icon} {result.get('danger_level','?')} ({result.get('danger_score',0):.2f})")
    print(f"  转写:       \"{result.get('text','(无语音)')}\"")
    print(f"  语气:       {result.get('tone_description','--')}")
    print(f"  理由:       {result.get('reason','--')}")
    if result.get("api_latency_ms", 0) > 0:
        print(f"  API延迟:    {result['api_latency_ms']:.0f}ms")
    if result.get("keywords"):
        print(f"  敏感词:     {', '.join(result['keywords'])}")
    if result.get("error_message"):
        print(f"  [WARNING]   {result['error_message']}")
    print(f"  {'='*55}\n")


# ============================================================
# 模式1: 传文件分析
# ============================================================

def mode_file(file_path: str, server_url: str):
    """直接传文件到后端分析"""
    fpath = Path(file_path)
    if not fpath.exists():
        print(f"\n  [ERROR] File not found: {file_path}\n")
        return

    print("=" * 55)
    print("  Mode: File Analysis")
    print("=" * 55)
    print(f"  File:      {fpath.name}")
    print(f"  Size:      {fpath.stat().st_size / 1024:.0f} KB")
    print(f"  Backend:   {server_url}")

    print(f"\n  Sending to backend...")
    result = send_to_backend(str(fpath), server_url)
    print_result(result)


# ============================================================
# 模式2: 录音分析
# ============================================================

def mode_record(duration_sec: float, server_url: str,
                loop: bool, device_index: int):
    """录音后传后端分析"""
    print("=" * 55)
    print("  Mode: Record & Analyze")
    print("=" * 55)
    print(f"  Duration:  {duration_sec}s")
    print(f"  Backend:   {server_url}")
    print(f"  Loop:      {'ON' if loop else 'OFF'}")
    if device_index is not None:
        print(f"  Device:    #{device_index}")

    save_dir = _PROJECT_ROOT / "recordings"
    save_dir.mkdir(exist_ok=True)
    round_num = 0

    while True:
        round_num += 1

        # 1. 录音
        audio = record_audio(duration_sec, device_index)
        if audio is None:
            break

        # 2. 保存 WAV（带时间戳，分析完保留不删）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = save_dir / f"rec_{timestamp}_{round_num:02d}.wav"
        save_wav(audio, str(wav_path))

        # 3. 发送到后端
        print(f"  Sending to backend...")
        result = send_to_backend(str(wav_path), server_url)

        # 4. 显示结果
        print_result(result)

        print(f"  Saved: {wav_path}")

        if not loop:
            break

        print(f"  {'-'*55}")
        print(f"  Next recording in 2s... (Ctrl+C to stop)")
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break

    print(f"\n  Recordings saved in: {save_dir}")
    print("  Done.\n")


# ============================================================
# Omni 模式: Omni 文件分析
# ============================================================

def mode_omni_file(file_path: str, server_url: str):
    """Omni 智能体文件分析模式 — 调用 POST /api/omni"""
    fpath = Path(file_path)
    if not fpath.exists():
        print(f"\n  [ERROR] File not found: {file_path}\n")
        return

    print("=" * 55)
    print("  Mode: Omni File Analysis (云端智能体)")
    print("=" * 55)
    print(f"  File:      {fpath.name}")
    print(f"  Size:      {fpath.stat().st_size / 1024:.0f} KB")
    print(f"  Backend:   {server_url}")

    print(f"\n  Sending to Omni API...")
    result = send_to_omni_api(str(fpath), server_url)
    print_omni_result(result)


def mode_omni_record(duration_sec: float, server_url: str,
                     loop: bool, device_index: int):
    """Omni 智能体录音分析模式 — 录音后调 POST /api/omni"""
    print("=" * 55)
    print("  Mode: Omni Record & Analyze (云端智能体)")
    print("=" * 55)
    print(f"  Duration:  {duration_sec}s")
    print(f"  Backend:   {server_url}")
    print(f"  Loop:      {'ON' if loop else 'OFF'}")
    if device_index is not None:
        print(f"  Device:    #{device_index}")

    save_dir = _PROJECT_ROOT / "recordings"
    save_dir.mkdir(exist_ok=True)
    round_num = 0

    while True:
        round_num += 1

        # 1. 录音
        audio = record_audio(duration_sec, device_index)
        if audio is None:
            break

        # 2. 保存 WAV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        wav_path = save_dir / f"rec_omni_{timestamp}_{round_num:02d}.wav"
        save_wav(audio, str(wav_path))

        # 3. 发送到 Omni API
        print(f"  Sending to Omni API...")
        result = send_to_omni_api(str(wav_path), server_url)

        # 4. 显示结果
        print_omni_result(result)

        print(f"  Saved: {wav_path}")

        if not loop:
            break

        print(f"  {'-'*55}")
        print(f"  Next recording in 2s... (Ctrl+C to stop)")
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break

    print(f"\n  Recordings saved in: {save_dir}")
    print("  Done.\n")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Audio Analysis Tool — Record or File mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File mode (send audio file to backend for analysis)
  python frontend/record_analyze.py --file recording.wav
  python frontend/record_analyze.py --file ../测试用例/大笑2.wav

  # Record mode (record from mic then analyze)
  python frontend/record_analyze.py                        # 3s default
  python frontend/record_analyze.py --record --duration 5  # 5 seconds
  python frontend/record_analyze.py --record --loop        # continuous loop
  python frontend/record_analyze.py --record --device 15   # specific mic

  # Omni mode (cloud agent: Qwen3.5-Omni-Flash)
  python frontend/record_analyze.py --file test.wav --omni
  python frontend/record_analyze.py --record --duration 5 --omni
        """,
    )
    # 文件模式
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Audio file path (wav/mp3/mp4 etc.) — file analysis mode")

    # 录音模式
    parser.add_argument("--record", "-r", action="store_true",
                        help="Record from microphone — recording mode (default)")
    parser.add_argument("--duration", "-d", type=float, default=3.0,
                        help="Recording duration in seconds (default: 3)")
    parser.add_argument("--loop", "-l", action="store_true",
                        help="Loop: keep recording and analyzing")
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index (use --list-devices to see)")

    # Omni 模式
    parser.add_argument("--omni", action="store_true",
                        help="Use Omni agent (Qwen3.5-Omni-Flash) for analysis instead of CNN")

    # 通用
    parser.add_argument("--server", "-s", default="http://localhost:8080",
                        help="Backend server URL (default: http://localhost:8080)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")

    args = parser.parse_args()

    # 列出设备
    if args.list_devices:
        from frontend.audio_capture import AudioCapture
        print("\n  Available audio input devices:\n")
        print("-" * 60)
        for d in AudioCapture.list_devices():
            print(f"  [{d['index']:>2d}] {d['name']}")
            print(f"        channels={d['channels']}, sr={d['default_sample_rate']}Hz")
        print("-" * 60)
        print("\n  Usage: python frontend/record_analyze.py --record --device <index>\n")
        return

    # 选模式
    if args.file:
        # 模式1: 直接传文件
        if args.omni:
            mode_omni_file(args.file, args.server)
        else:
            mode_file(args.file, args.server)
    else:
        # 模式2: 录音分析 (默认)
        if args.omni:
            mode_omni_record(args.duration, args.server, args.loop, args.device)
        else:
            mode_record(args.duration, args.server, args.loop, args.device)


if __name__ == "__main__":
    main()
