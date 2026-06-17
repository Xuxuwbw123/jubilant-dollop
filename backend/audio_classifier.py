
import os
import time
import random
import logging
from typing import List, Optional

import numpy as np

from common.audio_config import ClassifyResult

logger = logging.getLogger("backend.classifier")

# PyTorch 可选导入
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 未安装，分类器将使用模拟模式")


# ============================================================
# PyTorch 模型定义 (简易 CNN，可替换为 LSTM/YAMNet/VGGish)
# ============================================================

class AudioCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    2D-CNN 音频分类模型 (v3 — 保留时间维度)

    输入: (batch, 3, n_mels, n_frames)  3 通道频谱图
    - 4 个 Conv2d block + BatchNorm2d + MaxPool2d
    - AdaptiveAvgPool2d → 固定长度分类头
    - 参数量 ~380K

    与 v2(1D) 的本质区别:
    - v2: 时间轴取均值 → 120维向量 → 只能看"平均音色"
    - v3: 2D频谱图 → 保留时间维度 → 能看"声音怎么样随时间变化"
    """

    def __init__(self, input_dim: int | dict = 40, num_classes: int = 5):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法创建模型")
        super().__init__()

        # 解析 input_dim: dict={channels, n_mels} 或 旧版 int
        if isinstance(input_dim, dict):
            in_channels = input_dim.get("channels", 3)
        else:
            in_channels = 3  # 默认 3 通道

        # Block 1: (in, H, W) → (32, H, W)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2: (32, H, W) → (64, H/2, W/2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        # Block 3: (64, H/2, W/2) → (128, H/4, W/4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Block 4: (128, H/4, W/4) → (256, H/8, W/8)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)

        # 全局池化 → 固定维度
        self.gap = nn.AdaptiveAvgPool2d(1)  # → (256, 1, 1)

        # 分类头
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, n_mels, n_frames)
        # 若输入是 2D (batch, n_mels, n_frames)，自动补通道维
        if x.dim() == 3:
            x = x.unsqueeze(1)  # → (batch, 1, n_mels, n_frames)
        if x.size(1) == 1 and hasattr(self, 'conv1') and self.conv1.in_channels > 1:
            x = x.repeat(1, self.conv1.in_channels, 1, 1)  # 兼容旧单通道输入

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.gap(x).flatten(1)  # (batch, 256)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# AudioClassifier
# ============================================================

class AudioClassifier:
    """
    AI 人声分类器
    - 加载训练好的 PyTorch 模型
    - 对特征向量进行分类
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 input_dim: int = 40,
                 num_classes: int = 5,
                 class_names: List[str] = None,
                 simulate: bool = False,
                 model_type: str = 'auto'):
        """
        Args:
            model_path: 模型文件路径 (.pt)
            input_dim: 输入特征维度 (CNN 用)
            num_classes: 分类数
            class_names: 类别名列表
            simulate: 模拟模式
            model_type: 'v3' | 'v4' | 'emotion2vec' | 'auto' (自动从 checkpoint 检测)
        """
        self.model_path = model_path
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_names = class_names or ["normal", "scream", "cry", "laugh"]
        self.simulate = simulate or not TORCH_AVAILABLE
        self.model_type = model_type

        self.model = None
        self._loaded = False
        self._e2v_model = None  # lazy-loaded funasr emotion2vec model

        if not self.simulate:
            self.load_model(model_path)

    @staticmethod
    def _detect_model_type(state_dict: dict) -> str:
        """从 checkpoint state_dict 检测模型架构"""
        keys = list(state_dict.keys())
        # v4: 包含 temporal_attn / stage1_ms / se1 等特有模块
        for key in keys:
            if 'temporal_attn' in key or 'stage1_ms' in key or 'se1' in key:
                return 'v4'
        # emotion2vec FC head: 有 fc1/fc3 但无任何 conv 层
        has_conv = any('conv' in k for k in keys)
        if not has_conv:
            return 'emotion2vec'
        return 'v3'

    def _create_model(self, model_type: str):
        """创建对应架构的模型实例"""
        if model_type and 'emotion2vec' in model_type:
            return Emotion2VecWrapper(num_classes=self.num_classes)
        elif model_type == 'v4':
            return AudioCNNv4(num_classes=self.num_classes)
        else:
            return AudioCNN(self.input_dim, self.num_classes)

    def load_model(self, model_path: Optional[str] = None):
        """加载训练好的模型 (自动检测 v3/v4/emotion2vec 架构)"""
        if model_path:
            self.model_path = model_path

        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"模型文件不存在: {self.model_path}，使用模拟模式")
            self.simulate = True
            return

        if not TORCH_AVAILABLE:
            self.simulate = True
            return

        try:
            # ★ 自动选择设备
            if torch.cuda.is_available():
                self._device = "cuda"
                map_loc = "cuda"
                logger.info("CUDA 可用，将使用 GPU 加速")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
                map_loc = "cpu"
                logger.info("Apple MPS 可用，将使用 GPU 加速")
            else:
                self._device = "cpu"
                map_loc = "cpu"
                logger.info("使用 CPU 推理")

            checkpoint = torch.load(self.model_path, map_location=map_loc,
                                    weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # 优先用 checkpoint 中的 model_type 标记
            ckpt_type = checkpoint.get("model_type", None)

            if self.model_type == 'auto':
                if ckpt_type:
                    detected = ckpt_type
                else:
                    detected = self._detect_model_type(state_dict)
                logger.info(f"检测到模型架构: {detected}")
            else:
                detected = self.model_type

            self.model = self._create_model(detected)
            self.model.load_state_dict(state_dict)
            self.model.to(self._device)
            self.model.eval()
            self._loaded = True
            self._detected_type = detected
            logger.info(f"模型加载成功 ({detected}, device={self._device}): {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}，使用模拟模式")
            self.simulate = True

    # ---- 推理 ----

    def predict(self, features: np.ndarray,
                frame_index: int = 0) -> ClassifyResult:
        """
        单帧分类
        Args:
            features: 特征 (3, n_mels, n_frames) — 3通道2D频谱图
            frame_index: 帧序号
        Returns:
            ClassifyResult
        """
        if self.simulate:
            return self._simulate_predict(frame_index)

        try:
            with torch.no_grad():
                # numpy → tensor: (3, H, W) → (1, 3, H, W)
                x = torch.from_numpy(features).float().unsqueeze(0)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                conf, pred_idx = probs.max(dim=0)
                pred_class = self.class_names[pred_idx.item()]
                confidence = conf.item()

                # 所有类别的概率
                all_probs = {
                    self.class_names[i]: round(probs[i].item(), 4)
                    for i in range(len(self.class_names))
                }

            return ClassifyResult(
                class_name=pred_class,
                confidence=confidence,
                is_abnormal=(pred_class != "normal"),
                timestamp=time.time(),
                frame_index=frame_index,
                all_probs=all_probs,
            )
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return self._simulate_predict(frame_index)

    # ---- emotion2vec 推理 (原始音频 → embedding → FC) ----

    def predict_from_audio(self, audio: np.ndarray,
                           frame_index: int = 0) -> ClassifyResult:
        """
        从原始音频直接分类 (emotion2vec 路径)

        Args:
            audio: 1D float32 数组, 16kHz 单声道原始音频
            frame_index: 帧序号
        Returns:
            ClassifyResult
        """
        if self.simulate:
            return self._simulate_predict(frame_index)

        try:
            # Lazy-load funasr emotion2vec model (只加载一次)
            if self._e2v_model is None:
                from funasr import AutoModel
                import os as _os
                _os.environ.setdefault('FUNASR_DISABLE_UPDATE', '1')
                e2v_device = getattr(self, '_device', 'cpu')
                self._e2v_model = AutoModel(
                    model="iic/emotion2vec_plus_base",
                    hub="ms",
                    device=e2v_device if e2v_device in ("cpu", "cuda") else "cpu",
                    disable_update=True,
                )
                logger.info(f"emotion2vec 模型加载成功 (device={e2v_device})")

            # 确保 float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 提取 768-dim embedding
            result = self._e2v_model.generate(
                audio, granularity="utterance", extract_embedding=True
            )[0]
            embedding = np.array(result["feats"], dtype=np.float32)

            # FC 分类头推理
            dev = getattr(self, '_device', 'cpu')
            with torch.no_grad():
                x = torch.from_numpy(embedding).float().unsqueeze(0).to(dev)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                conf, pred_idx = probs.max(dim=0)
                pred_class = self.class_names[pred_idx.item()]
                confidence = conf.item()

                all_probs = {
                    self.class_names[i]: round(probs[i].item(), 4)
                    for i in range(len(self.class_names))
                }

            return ClassifyResult(
                class_name=pred_class,
                confidence=confidence,
                is_abnormal=(pred_class != "normal"),
                timestamp=time.time(),
                frame_index=frame_index,
                all_probs=all_probs,
            )
        except Exception as e:
            logger.error(f"emotion2vec 推理失败: {e}")
            return self._simulate_predict(frame_index)

    # ---- emotion2vec 原生情绪识别 (带情绪说话) ----

    EMOTION_LABELS = ['angry', 'disgusted', 'fearful', 'happy',
                      'neutral', 'other', 'sad', 'surprised', 'unk']
    EMOTION_CN = {
        'angry': '愤怒', 'disgusted': '厌恶', 'fearful': '恐惧',
        'happy': '高兴', 'neutral': '中性', 'other': '其他',
        'sad': '悲伤', 'surprised': '惊讶', 'unk': '未知',
    }
    # 哪些情绪算"异常"（需要报警）
    EMOTION_ABNORMAL = {'angry', 'fearful', 'sad', 'surprised'}

    def predict_emotion_from_audio(self, audio: np.ndarray,
                                   frame_index: int = 0) -> ClassifyResult:
        """
        原生情绪识别: 直接使用 emotion2vec 的 9 类标签
        适用于带情绪的自然说话 (angry/happy/sad/fearful/...)

        Args:
            audio: 1D float32, 16kHz 原始音频
            frame_index: 帧序号
        """
        if self.simulate:
            return self._simulate_predict(frame_index)

        try:
            if self._e2v_model is None:
                from funasr import AutoModel
                import os as _os
                _os.environ.setdefault('FUNASR_DISABLE_UPDATE', '1')
                e2v_device = getattr(self, '_device', 'cpu')
                self._e2v_model = AutoModel(
                    model="iic/emotion2vec_plus_base",
                    hub="ms", device=e2v_device if e2v_device in ("cpu", "cuda") else "cpu",
                    disable_update=True,
                )
                logger.info(f"emotion2vec 原生情绪模型加载成功 (device={e2v_device})")

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            result = self._e2v_model.generate(
                audio, granularity="utterance", extract_embedding=False,
            )[0]

            scores = result["scores"]
            labels = result.get("labels", self.EMOTION_LABELS)

            # 找最高分类
            best_idx = int(np.argmax(scores))
            best_label = labels[best_idx] if isinstance(labels, list) else self.EMOTION_LABELS[best_idx]
            confidence = float(scores[best_idx])

            # ★ 标签归一化: 处理 "中文/English" 组合格式 (emotion2vec 实际输出)
            def _normalize_label(raw_label: str) -> str:
                """将 emotion2vec 标签归一化为英文 short name"""
                if raw_label in self.EMOTION_LABELS:
                    return raw_label
                if raw_label in self.EMOTION_CN:
                    rev = {v: k for k, v in self.EMOTION_CN.items()}
                    return rev.get(raw_label, raw_label)
                # 处理 "中文/English" 组合格式, 取 '/' 后的英文部分
                if '/' in raw_label:
                    parts = raw_label.split('/')
                    for part in parts:
                        part = part.strip()
                        if part in self.EMOTION_LABELS:
                            return part
                    # 斜杠分割后仍无匹配, 取最后一段
                    return parts[-1].strip()
                return raw_label

            class_name = _normalize_label(best_label)

            all_probs = {}
            for i, label in enumerate(labels if isinstance(labels, list) else self.EMOTION_LABELS):
                short = _normalize_label(label)
                if short not in self.EMOTION_LABELS:
                    short = self.EMOTION_LABELS[i] if i < len(self.EMOTION_LABELS) else f'cls{i}'
                all_probs[short] = round(float(scores[i]), 4)

            is_abnormal = class_name in self.EMOTION_ABNORMAL

            return ClassifyResult(
                class_name=class_name,
                confidence=round(confidence, 4),
                is_abnormal=is_abnormal,
                timestamp=time.time(),
                frame_index=frame_index,
                all_probs=all_probs,
            )
        except Exception as e:
            logger.error(f"emotion2vec 原生情绪识别失败: {e}")
            return self._simulate_predict(frame_index)

    def _init_sim_state(self):
        """初始化模拟状态机"""
        self._sim_current_class = "normal"
        self._sim_frames_in_state = 0
        self._sim_state_duration = random.randint(30, 80)  # normal 持续 0.9~2.4 秒
        self._sim_base_confidence = random.uniform(0.80, 0.98)

    def _transition_sim_state(self):
        """
        状态转移：模拟真实场景中声音的变化

        规则:
        - 当前是 normal → 80% 继续 normal，20% 进入异常
        - 当前是异常 → 70% 回 normal（异常结束），30% 切换异常类型
        - 异常事件持续 3~30 帧（0.1~0.9 秒）
        """
        abnormal_classes = ["scream", "cry", "laugh"]

        if self._sim_current_class == "normal":
            # 正常状态：20% 概率进入异常
            if random.random() < 0.20:
                self._sim_current_class = random.choice(abnormal_classes)
                self._sim_state_duration = random.randint(3, 30)
                self._sim_base_confidence = random.uniform(0.65, 0.92)
            else:
                # 保持正常
                self._sim_current_class = "normal"
                self._sim_state_duration = random.randint(30, 100)
                self._sim_base_confidence = random.uniform(0.80, 0.98)
        else:
            # 异常状态：大概率结束后回到正常
            if random.random() < 0.70:
                self._sim_current_class = "normal"
                self._sim_state_duration = random.randint(20, 80)
                self._sim_base_confidence = random.uniform(0.80, 0.98)
            else:
                # 切换到另一个异常类型
                others = [c for c in abnormal_classes if c != self._sim_current_class]
                self._sim_current_class = random.choice(others)
                self._sim_state_duration = random.randint(3, 25)
                self._sim_base_confidence = random.uniform(0.60, 0.90)

        self._sim_frames_in_state = 0

    def _sim_confidence(self) -> float:
        """
        模拟置信度变化曲线

        在状态中间时置信度最高（模拟模型"看清"了特征），
        在状态开头和结尾置信度较低（模拟模型还"不确定"）。
        """
        duration = max(1, self._sim_state_duration)
        position = self._sim_frames_in_state / duration  # 0.0 ~ 1.0

        # 用正弦曲线模拟置信度变化：中间高，两端低
        # sin(pi * position) 在 0→0→1 时从 0→1→0
        curve = np.sin(np.pi * position) if hasattr(np, 'sin') else 1.0

        # 基础置信度 ± 0.15 的波动
        variance = 0.15 * curve
        confidence = self._sim_base_confidence + random.uniform(-0.08, 0.08)

        # 加一些随机噪声
        confidence += random.uniform(-0.03, 0.03)

        return max(0.50, min(0.99, confidence))

    def _simulate_predict(self, frame_index: int = 0) -> ClassifyResult:
        """
        模拟分类结果（带状态机的真实感模拟）

        与旧版纯随机的区别:
        - 状态有持续性：一旦开始尖叫，会持续 3~30 帧（0.1~0.9 秒）
        - 置信度有曲线：事件中间高、两端低，带随机噪声
        - 转换概率合理：normal→异常 15%，异常→normal 70%
        - 不会再出现"每隔一帧换一个类别"的假感
        """
        # 首次调用时初始化状态机
        if not hasattr(self, '_sim_current_class'):
            self._init_sim_state()

        # 帧计数 + 状态持续时间检查
        self._sim_frames_in_state += 1
        if self._sim_frames_in_state >= self._sim_state_duration:
            self._transition_sim_state()

        # 当前状态
        class_name = self._sim_current_class
        confidence = self._sim_confidence()

        return ClassifyResult(
            class_name=class_name,
            confidence=round(confidence, 4),
            is_abnormal=(class_name != "normal"),
            timestamp=time.time(),
            frame_index=frame_index,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ============================================================
# AudioCNN v4 — 方案B 架构改进
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.GELU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时间卷积: 并行多个不同时间跨度的卷积核

    - (3,1): 短时局部纹理 (~30ms)
    - (3,5): 中时模式 (~150ms) — 颤音、短促爆发
    - (3,11): 长时模式 (~350ms) — 节奏性爆发-衰减
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        c_per_branch = out_c // 3
        c_remainder = out_c - 3 * c_per_branch

        self.conv_short = nn.Conv2d(in_c, c_per_branch, kernel_size=(3, 1),
                                     padding=(1, 0), bias=False)
        self.conv_mid = nn.Conv2d(in_c, c_per_branch, kernel_size=(3, 5),
                                   padding=(1, 2), bias=False)
        self.conv_long = nn.Conv2d(in_c, c_per_branch + c_remainder,
                                    kernel_size=(3, 11), padding=(1, 5), bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        short = self.conv_short(x)
        mid = self.conv_mid(x)
        long_out = self.conv_long(x)
        out = torch.cat([short, mid, long_out], dim=1)
        return self.act(self.bn(out))


class TemporalAttentionPool(nn.Module):
    """
    注意力时间池化 (替代 GAP)

    GAP 把每个特征图的所有时间帧均匀平均 → 丢失位置信息
    注意力池化 → 让模型学习"哪些帧更重要"，加权求和
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn_conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)  H=freq, W=time
        B, C, H, W = x.shape
        # 频率方向平均 → (B, C, W)
        x_freq_avg = x.mean(dim=2)
        # 注意力权重 → (B, 1, W)
        attn = self.attn_conv(x_freq_avg)
        attn = torch.softmax(attn, dim=-1)
        # 加权时间池化 → (B, C)
        x_pooled = (x_freq_avg * attn).sum(dim=-1)
        return x_pooled


class AudioCNNv4(nn.Module):
    """
    AudioCNN v4 — 方案B 轻量架构改进

    相比 v3 的核心改动:
    1. SE 通道注意力 — 自适应学习特征图重要性
    2. 残差连接 — 每 stage 的跳跃连接，缓解梯度稀释
    3. 多尺度时间卷积 (3×1, 3×5, 3×11) — 覆盖短/中/长时间模式
    4. 注意力时间池化 — 替代 GAP，关注关键帧
    5. GeLU 激活 + 分层 Dropout — 更平滑的梯度

    输入: (B, 3, 128, 32) — 3通道 mel spectrogram
    参数量: ~500K
    """

    def __init__(self, input_dim=None, num_classes: int = 5):
        super().__init__()

        # ---- Stem: 3→32, 保持分辨率 ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # ---- Stage 1: 32→64, pool(2,2) ----
        self.stage1_ms = MultiScaleTemporalConv(32, 64)
        self.se1 = SEBlock(64)
        self.stage1_skip = nn.Conv2d(32, 64, kernel_size=1, bias=False)  # 1×1 投影
        self.pool1 = nn.MaxPool2d((2, 2))

        # ---- Stage 2: 64→128, pool(2,1) 只降频率 ----
        self.stage2_ms = MultiScaleTemporalConv(64, 128)
        self.se2 = SEBlock(128)
        self.stage2_skip = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.pool2 = nn.MaxPool2d((2, 1))

        # ---- Stage 3: 128→256, 不下采样 ----
        self.stage3_ms = MultiScaleTemporalConv(128, 256)
        self.se3 = SEBlock(256)
        self.stage3_skip = nn.Conv2d(128, 256, kernel_size=1, bias=False)

        # ---- 注意力时间池化 (替代 GAP) ----
        self.temporal_attn = TemporalAttentionPool(256)

        # ---- 分类头 ----
        self.dropout_conv = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 兼容旧输入格式
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Stem
        x = self.stem(x)  # (B, 32, 128, 32)

        # Stage 1 (残差)
        identity = self.stage1_skip(x)  # (B, 64, 128, 32)
        x = self.stage1_ms(x)            # (B, 64, 128, 32)
        x = self.se1(x)
        x = x + identity
        x = self.pool1(x)               # (B, 64, 64, 16)

        # Stage 2 (残差, 只降频率)
        identity = self.stage2_skip(x)   # (B, 128, 64, 16)
        x = self.stage2_ms(x)            # (B, 128, 64, 16)
        x = self.se2(x)
        x = x + identity
        x = self.pool2(x)               # (B, 128, 32, 16) 时间维度不变

        # Stage 3 (残差, 不下采样)
        identity = self.stage3_skip(x)   # (B, 256, 32, 16)
        x = self.stage3_ms(x)            # (B, 256, 32, 16)
        x = self.se3(x)
        x = x + identity                # (B, 256, 32, 16)

        # 注意力时间池化
        x = self.temporal_attn(x)       # (B, 256)

        # 分类头
        x = self.dropout_conv(x)
        x = self.fc(x)                  # (B, 5)
        return x


# ============================================================
# Emotion2Vec FC Classifier (轻量分类头)
# ============================================================

class Emotion2VecWrapper(nn.Module):
    """
    emotion2vec 分类器包装器 (仅 FC 分类头)

    embedding 提取由 funasr 完成，本模块只负责:
    768 → 256 → 128 → 5

    配合 AudioClassifier.predict_from_audio() 使用
    """

    def __init__(self, input_dim=768, num_classes=5, hidden1=256, hidden2=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        """x: (batch, 768) emotion2vec embeddings"""
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ---- 模型保存辅助 ----

def save_model(model, filepath: str, metadata: dict = None):
    """保存模型和元数据"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch 未安装，无法保存模型")
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    save_dict = {"model_state_dict": model.state_dict()}
    if metadata:
        save_dict["metadata"] = metadata
    torch.save(save_dict, filepath)
    logger.info(f"模型已保存: {filepath}")
