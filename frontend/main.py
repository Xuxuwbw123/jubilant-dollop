"""
前端入口
启动音频采集 → WebSocket 发送 流水线

用法:
    python main.py                              # 默认配置
    python main.py --server ws://192.168.1.100:8765
    python main.py --simulate                   # 模拟模式 (无需麦克风)
    python main.py --list-devices               # 列出音频设备
    python main.py --device 1                   # 指定音频设备
"""

import sys
import asyncio
import argparse
import json
import time
import re
from pathlib import Path
from urllib.parse import urlparse

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from frontend.config import FrontendConfig
from frontend.audio_capture import AudioCapture
from frontend.audio_sender import AudioSender
from common.utils import setup_logging

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="前端: 音频采集与网络发送"
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="WebSocket 服务器地址 (例: ws://192.168.1.100:8765)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="模拟模式 (无需麦克风，用假音频测试)"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出可用音频输入设备并退出"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="指定音频输入设备索引 (先用 --list-devices 查看)"
    )
    return parser.parse_args()


def _derive_dashboard_url(server_url: str) -> str:
    """从音频 WebSocket URL 推导仪表盘 WebSocket URL"""
    parsed = urlparse(server_url)
    # 仪表盘默认在 8766 端口
    dashboard_port = 8766
    return f"ws://{parsed.hostname}:{dashboard_port}"


async def receive_dashboard_results(dashboard_url: str,
                                     stop_event: asyncio.Event):
    """连接仪表盘 WebSocket，实时显示 Omni/CNN 分析结果"""
    if not WEBSOCKETS_AVAILABLE:
        return

    # 颜色映射
    CLASS_COLORS = {
        "normal": "\033[32m",    # 绿色
        "scream": "\033[31m",    # 红色
        "cry": "\033[35m",       # 紫色
        "laugh": "\033[33m",     # 黄色
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    while not stop_event.is_set():
        try:
            async with websockets.connect(
                dashboard_url,
                ping_interval=None,
                close_timeout=3,
            ) as ws:
                print(f"\n  [仪表盘] 已连接: {dashboard_url}")
                print(f"  {'─'*50}")

                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        msg = json.loads(raw)

                        if msg.get("type") != "classify_result":
                            continue

                        class_name = msg.get("class_name", "?")
                        confidence = msg.get("confidence", 0)
                        all_probs = msg.get("all_probs", {})
                        alert = msg.get("alert")

                        # 构建实时显示行
                        ts = time.strftime("%H:%M:%S")
                        color = CLASS_COLORS.get(class_name, "")
                        class_cn = {
                            "normal": "正常", "scream": "尖叫",
                            "cry": "大哭", "laugh": "大笑",
                        }.get(class_name, class_name)

                        line = f"  [{ts}] {color}{class_cn}{RESET} ({confidence:.0%})"

                        # 显示 Omni 来源标记 (all_probs 均匀分布 → Omni)
                        probs = list(all_probs.values())
                        if probs and max(probs) - min(probs) < 0.01 and max(probs) > 0:
                            line += f" {DIM}☁Omni{RESET}"

                        # 报警标记
                        if alert:
                            level_icon = {"critical": "🔴", "warning": "🟠", "pre_alert": "🟡"}
                            line += f"  {level_icon.get(alert.get('level',''),'⚠')} {alert.get('message','')[:50]}"

                        print(f"\r\033[K{line}", end="", flush=True)

                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

        except (OSError, ConnectionRefusedError):
            # 仪表盘未就绪，等一会重试
            await asyncio.sleep(5)
        except Exception:
            await asyncio.sleep(5)

        if not stop_event.is_set():
            print(f"\n  [仪表盘] 断开，重连中...", end="", flush=True)
            await asyncio.sleep(3)


async def main():
    args = parse_args()

    # ---- 列出设备 ----
    if args.list_devices:
        print("\n正在检测音频输入设备...\n")
        devices = AudioCapture.list_devices()
        if devices:
            print("可用音频输入设备:")
            print("-" * 60)
            for d in devices:
                print(f"  [{d['index']}] {d['name']}")
                print(f"       声道: {d['channels']}, "
                      f"默认采样率: {d['default_sample_rate']}Hz")
            print("-" * 60)
            print("\n用法: python main.py --device <索引号>")
        else:
            print("未检测到音频输入设备！")
            print("请检查麦克风是否已连接，或使用 --simulate 模式测试")
        return

    # ---- 加载配置 ----
    config = FrontendConfig()

    # 命令行覆盖
    if args.server:
        config.server_url = args.server
    if args.simulate:
        config.simulate = True
    if args.device is not None:
        config.input_device_index = args.device

    # ---- 日志 ----
    logger = setup_logging("frontend", config.log_level, config.log_dir)

    # ---- 打印启动信息 ----
    logger.info("=" * 50)
    logger.info("前端启动: 双机实时音频监测系统")
    logger.info(f"服务器地址: {config.server_url}")
    logger.info(f"运行模式: {'模拟模式' if config.simulate else '真实麦克风模式'}")
    if not config.simulate:
        if config.input_device_index is not None:
            logger.info(f"音频设备: 索引 #{config.input_device_index}")
        else:
            logger.info("音频设备: 系统默认麦克风")
    logger.info(f"音频参数: {config.audio.sample_rate}Hz, "
                f"{config.audio.frame_duration_ms}ms/帧, "
                f"{'单声道' if config.audio.channels == 1 else '多声道'}")
    logger.info(f"VAD静音检测: {'启用' if config.vad.enabled else '禁用'}")
    logger.info("=" * 50)

    # ---- 初始化模块 ----
    capture = AudioCapture(
        audio_config=config.audio,
        vad_config=config.vad,
        simulate=config.simulate,
        input_device_index=config.input_device_index,
    )

    sender = AudioSender(
        server_url=config.server_url,
        reconnect_interval=config.reconnect_interval,
        heartbeat_interval=config.heartbeat_interval,
    )

    # ---- 开始采集 ----
    if not capture.start():
        logger.error("音频采集启动失败！")
        if not config.simulate:
            logger.info("提示: 试试 --list-devices 查看可用设备")
            logger.info("提示: 或使用 --simulate 进入模拟模式测试")
        return

    # ---- 仪表盘结果监听 (并行) ----
    dashboard_url = _derive_dashboard_url(config.server_url)
    stop_event = asyncio.Event()
    dashboard_task = asyncio.create_task(
        receive_dashboard_results(dashboard_url, stop_event)
    )

    # ---- 运行发送循环 ----
    try:
        await sender.run_with_reconnect(capture)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        stop_event.set()
        capture.stop()
        await sender.disconnect()
        dashboard_task.cancel()
        try:
            await dashboard_task
        except asyncio.CancelledError:
            pass
        print()  # 换行，避免 shell prompt 覆盖
        logger.info(f"前端已停止。共发送 {sender.frames_sent} 帧。")


if __name__ == "__main__":
    asyncio.run(main())
