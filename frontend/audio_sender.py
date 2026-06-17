
import asyncio
import json
import logging
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    ConnectionClosed = Exception
    WebSocketException = Exception
    WEBSOCKETS_AVAILABLE = False

from common.protocol import (
    encode_audio_frame, encode_heartbeat, encode_control_ack,
    parse_message_type, MessageType,
)
from common.audio_config import AudioFrame

logger = logging.getLogger("frontend.sender")


class AudioSender:
    """
    WebSocket 音频发送客户端
    - 连接到后端服务器
    - 发送音频帧
    - 断线自动重连
    - 心跳保活
    """

    def __init__(self, server_url: str,
                 reconnect_interval: float = 3.0,
                 heartbeat_interval: float = 5.0):
        """
        Args:
            server_url: WebSocket 服务器地址 (e.g. ws://192.168.1.100:8765)
            reconnect_interval: 断线重连间隔 (秒)
            heartbeat_interval: 心跳间隔 (秒)
        """
        self.server_url = server_url
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False

        # 统计
        self.frames_sent: int = 0
        self.bytes_sent: int = 0

    # ═══════════════════════════════════════════════════════════
    # WebSocket 连接管理 — 带错误处理的连接与断开
    # ═══════════════════════════════════════════════════════════

    async def connect(self) -> bool:
        """连接到服务器"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error(
                "websockets 包未安装！请运行: pip install websockets"
            )
            return False

        try:
            logger.info(f"正在连接服务器: {self.server_url}")

            # 设置连接超时
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.server_url,
                    ping_interval=None,
                    close_timeout=3,
                    max_size=2 ** 24,    # 16MB
                ),
                timeout=5.0,  # 5 秒连接超时
            )
            self._connected = True
            self._running = True
            logger.info(f"[OK] 已连接到服务器: {self.server_url}")
            return True

        except asyncio.TimeoutError:
            logger.error("=" * 50)
            logger.error("连接超时！可能原因:")
            logger.error("  1. 后端没有启动")
            logger.error("  2. 后端 IP 地址填错了")
            logger.error("  3. 两台电脑不在同一网络")
            logger.error("  4. Windows 防火墙拦截了端口 8765")
            logger.error(f"  目标地址: {self.server_url}")
            logger.error("=" * 50)
            self._connected = False
            return False

        except ConnectionRefusedError:
            logger.error("=" * 50)
            logger.error("连接被拒绝！可能原因:")
            logger.error("  1. 后端还没启动（先在后端电脑上运行 start_backend.bat）")
            logger.error("  2. 端口号不对（默认是 8765）")
            logger.error(f"  目标地址: {self.server_url}")
            logger.error("=" * 50)
            self._connected = False
            return False

        except OSError as e:
            # 分析具体的 OSError
            errmsg = str(e)
            if "10060" in errmsg or "10061" in errmsg or "1225" in errmsg:
                logger.error("=" * 50)
                logger.error("无法连接到后端！")
                logger.error(f"  错误: {e}")
                logger.error("  可能原因:")
                logger.error("  1. 后端未启动或 IP 地址错误")
                logger.error("  2. 防火墙拦截（Windows 防火墙默认拦截外来连接）")
                logger.error("  3. 两台电脑不在同一局域网")
                logger.error("  解决方法:")
                logger.error("  - 在后端电脑 cmd 中运行:")
                logger.error("    netsh advfirewall firewall add rule name=\"AudioMonitor\" dir=in action=allow protocol=tcp localport=8765")
                logger.error(f"  目标地址: {self.server_url}")
                logger.error("=" * 50)
            else:
                logger.error(f"网络错误: {e}")
            self._connected = False
            return False

        except WebSocketException as e:
            logger.error(f"WebSocket 连接失败: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """断开连接"""
        self._running = False
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("已断开连接")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ═══════════════════════════════════════════════════════════
    # 帧编码与发送 — AudioFrame→JSON→WebSocket 传输
    # ═══════════════════════════════════════════════════════════

    async def send_frame(self, frame: AudioFrame) -> bool:
        """
        发送一个音频帧
        Returns: True=成功, False=失败
        """
        if not self._connected or not self._ws:
            return False

        try:
            msg = encode_audio_frame(frame)
            await self._ws.send(msg)

            self.frames_sent += 1
            self.bytes_sent += len(frame.data) * frame.data.itemsize
            return True
        except (ConnectionClosed, WebSocketException) as e:
            logger.warning(f"发送失败: {e}")
            self._connected = False
            return False

    async def send_heartbeat(self) -> bool:
        """发送心跳"""
        if not self._connected or not self._ws:
            return False

        try:
            msg = encode_heartbeat()
            await self._ws.send(msg)
            return True
        except Exception:
            self._connected = False
            return False

    # ═══════════════════════════════════════════════════════════
    # 控制指令接收 — 监听后端下发的启停命令
    # ═══════════════════════════════════════════════════════════

    async def receive_control(self) -> Optional[str]:
        """
        接收来自后端的控制消息
        Returns: "start" | "stop" | None
        """
        if not self._ws:
            return None

        try:
            # 非阻塞尝试接收
            raw = await asyncio.wait_for(self._ws.recv(), timeout=0.1)
            msg_type = parse_message_type(raw)

            if msg_type == MessageType.CONTROL_START:
                await self._ws.send(encode_control_ack())
                return "start"
            elif msg_type == MessageType.CONTROL_STOP:
                await self._ws.send(encode_control_ack())
                return "stop"
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

        return None

    # ═══════════════════════════════════════════════════════════
    # 心跳保活 — 定期发送 Ping 维持连接
    # ═══════════════════════════════════════════════════════════

    async def heartbeat_loop(self):
        """心跳发送循环"""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval)
            if self._connected:
                success = await self.send_heartbeat()
                if not success:
                    logger.warning("心跳发送失败，连接可能断开")

    # ═══════════════════════════════════════════════════════════
    # 自动重连循环 — 断线后指数退避重连
    # ═══════════════════════════════════════════════════════════

    async def run_with_reconnect(self, capture):
        """
        持续运行：连接 → 采集发送 → 断开重连
        Args:
            capture: AudioCapture 实例，提供 get_frame() 方法
        """
        from frontend.audio_capture import AudioCapture

        self._running = True  # 启动运行标志

        while self._running:
            # 连接
            if not self._connected:
                success = await self.connect()
                if not success:
                    logger.info(f"{self.reconnect_interval}s 后重试连接...")
                    await asyncio.sleep(self.reconnect_interval)
                    continue

            # 开始采集
            if not capture.is_running:
                capture.start()

            # 心跳协程
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())

            try:
                # 主发送循环
                while self._connected and capture.is_running:
                    # 获取音频帧 (在线程池中执行阻塞调用)
                    frame = await asyncio.get_event_loop().run_in_executor(
                        None, capture.get_frame
                    )

                    if frame is None:
                        break

                    # 只发送有声音的帧（降低带宽）
                    # 也可以选择全发，让后端决定
                    success = await self.send_frame(frame)
                    if not success:
                        logger.warning("发送失败，准备重连...")
                        break

                    # 检查控制消息
                    ctrl = await self.receive_control()
                    if ctrl == "stop":
                        logger.info("收到停止指令")
                        self._running = False
                        break

                    # 控制发送速率
                    frame_duration = frame.duration_ms / 1000.0
                    await asyncio.sleep(frame_duration * 0.8)  # 略小于帧长

            except Exception as e:
                logger.error(f"发送循环异常: {e}")
            finally:
                heartbeat_task.cancel()
                capture.stop()

            # 断开
            self._connected = False
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

            if self._running:
                logger.info(f"将在 {self.reconnect_interval}s 后重连...")
                await asyncio.sleep(self.reconnect_interval)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "connected": self._connected,
            "server_url": self.server_url,
        }
