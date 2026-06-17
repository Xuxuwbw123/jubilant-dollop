"""
前端专属配置
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.audio_config import AudioConfig, VADConfig
from common.utils import load_config


class FrontendConfig:
    """前端配置聚合"""

    def __init__(self, config_path: str = None):
        cfg = load_config(config_path)

        # 音频参数
        self.audio = AudioConfig.from_yaml(cfg)

        # VAD 参数
        self.vad = VADConfig.from_yaml(cfg)

        # 网络参数
        network = cfg.get("network", {})
        self.server_url: str = network.get("server_url", "ws://localhost:8765")
        self.reconnect_interval: float = network.get("reconnect_interval", 3.0)
        self.heartbeat_interval: float = network.get("heartbeat_interval", 5.0)

        # 模拟模式
        simulation = cfg.get("simulation", {})
        self.simulate: bool = simulation.get("enabled", False)

        # 日志
        log_cfg = cfg.get("logging", {})
        self.log_level: str = log_cfg.get("level", "INFO")
        self.log_dir: str = log_cfg.get("log_dir", "logs")

        # 音频设备
        self.input_device_index: int = cfg.get("input_device_index", None)  # None=默认设备
