
import asyncio
import json
import logging
from typing import Callable, Optional, Set

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WebSocketServerProtocol = None
    ConnectionClosed = Exception
    WEBSOCKETS_AVAILABLE = False

from common.protocol import (
    encode_audio_frame, decode_audio_frame,
    parse_message_type, MessageType,
)
from common.audio_config import AudioFrame

logger = logging.getLogger("backend.receiver")

# 回调类型: 收到 AudioFrame 时调用
FrameHandler = Callable[[AudioFrame], None]


class AudioReceiver:
    """
    WebSocket 音频接收服务器
    - 监听前端连接
    - 接收音频帧并解码
    - 通过回调分发给下游模块 (特征提取 + 分类)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._server = None
        self._running = False
        self._clients: Set[WebSocketServerProtocol] = set()

        # 回调链
        self._frame_handlers: list[FrameHandler] = []

        # 统计
        self.frames_received: int = 0
        self.clients_connected: int = 0

    # ---- 注册回调 ----

    def on_frame(self, handler: FrameHandler):
        """注册音频帧处理回调（可链式注册多个）"""
        self._frame_handlers.append(handler)
        return handler  # 可用作装饰器

    # ---- 服务控制 ----

    async def start(self):
        """启动 WebSocket 服务器"""
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError(
                "websockets 包未安装！请运行: pip install websockets"
            )

        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            max_size=2 ** 24,  # 16MB
        )
        logger.info(f"WebSocket 服务器已启动: ws://{self.host}:{self.port}")

    async def stop(self):
        """停止服务器"""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # 断开所有客户端
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        logger.info("WebSocket 服务器已停止")

    # ---- 客户端处理 ----

    async def _handle_client(self, ws: WebSocketServerProtocol, path: str = None):
        """处理单个客户端连接"""
        client_addr = ws.remote_address
        logger.info(f"客户端连接: {client_addr}")
        self._clients.add(ws)
        self.clients_connected += 1

        try:
            async for raw_message in ws:
                if not self._running:
                    break

                await self._process_message(raw_message, client_addr)

        except ConnectionClosed:
            logger.info(f"客户端断开: {client_addr}")
        except Exception as e:
            logger.error(f"客户端异常 [{client_addr}]: {e}")
        finally:
            self._clients.discard(ws)

    async def _process_message(self, raw_message: str, client_addr):
        """处理收到的消息"""
        msg_type = parse_message_type(raw_message)

        if msg_type == MessageType.AUDIO_FRAME:
            try:
                frame = decode_audio_frame(raw_message)
                self.frames_received += 1

                # 分发给所有注册的处理器
                for handler in self._frame_handlers:
                    try:
                        handler(frame)
                    except Exception as e:
                        logger.error(f"帧处理器异常: {e}")

            except Exception as e:
                logger.error(f"解码音频帧失败: {e}")

        elif msg_type == MessageType.HEARTBEAT:
            logger.debug(f"收到心跳: {client_addr}")

        else:
            logger.debug(f"未知消息类型: {msg_type}")

    # ---- 向客户端发送控制指令 ----

    async def send_control(self, command: str) -> int:
        """
        向所有连接的客户端发送控制指令
        Args:
            command: "start" | "stop"
        Returns:
            成功发送的客户端数
        """
        from common.protocol import encode_control
        msg = encode_control(command)
        count = 0

        for client in list(self._clients):
            try:
                await client.send(msg)
                count += 1
            except Exception:
                pass

        return count

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "frames_received": self.frames_received,
            "clients_count": len(self._clients),
            "total_connections": self.clients_connected,
            "host": self.host,
            "port": self.port,
        }
