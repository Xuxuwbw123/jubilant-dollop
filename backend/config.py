"""
后端专属配置
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.audio_config import AudioConfig, AlertConfig
from common.utils import load_config


class BackendConfig:
    """后端配置聚合"""

    def __init__(self, config_path: str = None):
        cfg = load_config(config_path)

        # 音频参数
        self.audio = AudioConfig.from_yaml(cfg)

        # 网络参数
        network_cfg = cfg.get("network", {})
        self.host: str = network_cfg.get("host", "0.0.0.0")
        self.port: int = network_cfg.get("port", 8765)

        # AI 模型参数
        model_cfg = cfg.get("model", {})
        self.model_path: str = model_cfg.get("model_path", "models/best_model.pt")
        self.num_classes: int = model_cfg.get("num_classes", 5)
        self.classifier_mode: str = model_cfg.get("classifier_mode", "nonverbal")
        self.class_names: list = model_cfg.get("class_names",
                                                ["normal", "scream", "cry", "laugh"])

        # 特征提取参数
        feat_cfg = model_cfg.get("feature", {})
        self.n_mfcc: int = feat_cfg.get("n_mfcc", 40)
        self.n_mels: int = feat_cfg.get("n_mels", 128)
        self.n_fft: int = feat_cfg.get("n_fft", 2048)
        self.hop_length: int = feat_cfg.get("hop_length", 512)

        # 报警规则
        self.alert = AlertConfig.from_yaml(cfg)

        # 模拟模式
        simulation_cfg = cfg.get("simulation", {})
        self.simulate: bool = simulation_cfg.get("enabled", False)

        # 日志
        log_cfg = cfg.get("logging", {})
        self.log_level: str = log_cfg.get("level", "INFO")
        self.log_dir: str = log_cfg.get("log_dir", "logs")

        # Omni 智能体配置
        omni_cfg = cfg.get("omni", {})
        self.omni_enabled: bool = omni_cfg.get("enabled", True)
        self.omni_window_sec: float = omni_cfg.get("window_seconds", 2.0)
        self.omni_trigger_interval: float = omni_cfg.get("trigger_interval", 3.0)
        self.omni_danger_threshold: float = omni_cfg.get("danger_threshold", 0.60)
