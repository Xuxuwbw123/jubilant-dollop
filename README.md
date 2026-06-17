# 双机实时音频远程监测与异常分析系统

## 项目简介

基于深度学习的实时人声监测系统，通过前端麦克风采集音频，经 WebSocket 传输至后端进行 **双通路分析**：**emotion2vec**（预训练语音情绪模型 + FC 分类头，4 类本地推理）和 **Qwen3.5-Omni-Flash**（全模态大模型直接分析语音情绪），实现对异常人声的实时监测和三级报警。

## 系统架构

```
[前端: 浏览器麦克风采集] --HTTP POST :8080--> [后端: AI分析+报警] --WebSocket :8766--> [仪表盘]
                                                     |
                          ┌──────────────────────────┼──────────────────────────┐
                          ↓                          ↓                          ↓
                   emotion2vec 通路              Omni 智能体通路           HTTP :8080
                emotion2vec_plus_base        Qwen3.5-Omni-Flash       /api/analyze
                768维 → FC(256→128→4)        云端情绪+危险判断        文件上传分析
                          ↓                          ↓
                          └────────────┬────────────┘
                                       ↓
                          报警引擎 (滑动窗口 + 三级报警)
                                       ↓
                          ws://:8766 → HTML 仪表盘
```

### 数据流

```
麦克风 (16kHz) → 30ms分帧 → WebSocket → 后端缓冲1秒
                                            ↓
                              频谱平坦度检测 (过滤静音/噪声)
                                            ↓
                                ┌───────────┴───────────┐
                                ↓                       ↓
                       emotion2vec 通路 (本地)    Omni 智能体 (云端)
                       emotion2vec_plus_base      Qwen3.5-Omni-Flash
                       768维 embedding             2秒音频窗口 → API
                               ↓                       ↓
                       FC 分类头 (4类, 230K)    情绪+危险+转写+理由
                               ↓                       ↓
                                └───────────┬───────────┘
                                            ↓
                                  融合判定 (CNN + Omni)
                                            ↓
                                ┌───────────┴───────────┐
                                ↓                       ↓
                         报警判断 (滑动窗口)     WebSocket 广播
                         预测平滑 + 一致性检查   ws://:8766 → 浏览器仪表盘
                                ↓
                          PyQt5 GUI 显示
```

## 目录结构

```
├── common/              # 共享模块（协议、音频配置、工具函数）
├── backend/             # 后端：WebSocket接收 + emotion2vec分类 + 报警 + GUI
│   ├── main.py              # 完整后端入口 (PyQt5 GUI + WebSocket + HTTP)
│   ├── server_headless.py   # 轻量后端入口 (HTTP only, 无GUI依赖)
│   ├── audio_classifier.py  # 模型推理 (Emotion2VecWrapper + AudioCNN v3/v4)
│   ├── feature_extractor.py # 音频特征提取 + 语音检测
│   ├── audio_receiver.py    # WebSocket 服务端
│   ├── file_handler.py      # 文件分析 (音频提取 + 滑动窗口分类)
│   ├── alert_manager.py     # 报警引擎
│   ├── gui_main_window.py   # PyQt5 主窗口
│   ├── gui_waveform.py      # 实时波形显示
│   ├── gui_history.py       # 报警历史列表
│   └── config.py            # 后端配置加载
├── frontend/            # Python前端：音频采集 + WebSocket发送
│   ├── main.py              # 实时流模式
│   ├── audio_capture.py     # PyAudio 麦克风采集 + VAD
│   └── audio_sender.py      # WebSocket 客户端
├── asr_emotion_agent/   # Omni 智能体（Qwen3.5-Omni-Flash 情绪+危险判断）
│   ├── config.py             # OmniConfig + System Prompt
│   ├── qwen_omni_client.py   # QwenOmniClient (OpenAI兼容协议)
│   └── omni_emotion_agent.py # OmniEmotionAgent (顶层智能体)
├── scripts/             # 训练与工具脚本
│   ├── train_emotion2vec_classifier.py  # emotion2vec FC分类头训练
│   ├── extract_emotion2vec_embeddings.py # 提取768维嵌入
│   ├── train_model.py          # AudioCNN v3 训练
│   ├── train_model_v4.py       # AudioCNN v4 训练
│   ├── rebuild_data.py         # 从现有数据重建划分
│   ├── prepare_data.py         # 完整数据预处理
│   ├── evaluate_model.py       # 测试集评估
│   ├── batch_test_real.py      # 批量真实录音测试
│   ├── quick_test_e2v.py       # emotion2vec 快速验证
│   └── augment_data.py         # 数据增强
├── design-demos/        # HTML 前端面板
│   ├── frontend-panel.html     # 前端采集面板 (录音+文件上传+结果)
│   ├── backend-dashboard.html  # 后端实时监控仪表盘
│   └── emotion-panel.html      # 情绪分析面板
├── models/              # 训练好的模型文件
│   ├── best_model_e2v.pt      # ★ 当前最佳模型 (emotion2vec, Val F1=0.867)
│   ├── best_model_v4_20e_f1_0593.pt  # CNN v4 最佳模型 (保留备选)
│   └── test_report_e2v.json   # emotion2vec 测试集评估报告
├── config.yaml          # 全局配置
├── requirements.txt     # Python 依赖
├── start_backend.bat    # 后端一键启动
└── start_frontend.bat   # 前端一键启动
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动后端

```bash
# 完整后端 (PyQt5 GUI + WebSocket + HTTP)
python backend/main.py

# 轻量后端 (HTTP only, 不需要 PyQt5)
python backend/server_headless.py --port 8080
```

### 3. 启动前端

浏览器访问：**http://localhost:8080/frontend-panel.html**

支持两种模式：
- **文件上传**: 上传 wav/mp3/m4a/webm → 滑动窗口分析
- **实时录音**: 浏览器麦克风 → 每2秒自动发送 → 实时结果

### 4. 打开仪表盘

浏览器访问：**http://localhost:8080/backend-dashboard.html**

### 5. Omni 智能体 (Qwen3.5-Omni-Flash)

```bash
# API 连通性验证
python asr_emotion_agent/verify.py

# 批量真实录音测试
python asr_emotion_agent/test_real.py
```

> **注意**: Omni 智能体需要 DashScope API Key。设置环境变量 `DASHSCOPE_API_KEY=sk-xxx`。

## AI 模型

### 主模型: emotion2vec + FC 分类头

| 项目 | 详情 |
|------|------|
| 架构 | emotion2vec_plus_base (冻结) → FC (768→256→128→4) |
| 参数量 | ~230K (仅 FC 分类头) |
| 输入 | 原始音频 16kHz → 768维 utterance embedding |
| 输出 | normal / scream / cry / laugh (4类) |
| 验证集 F1 | **0.867** (Epoch 72) |
| 训练数据 | 7,562 文件 (normal: 1336, scream: 3191, cry: 1514, laugh: 1521) |
| 推理延迟 | < 50ms (GPU 上 embedding + FC) |

### 备选模型: AudioCNN v4

| 项目 | 详情 |
|------|------|
| 架构 | 2D-CNN + SE注意力 + 残差连接 + 多尺度时间卷积 |
| 参数量 | ~500K |
| 输入 | 3通道梅尔频谱图 (3, 128, 32): static + Δ + ΔΔ |
| 输出 | normal / scream / cry / laugh (4类) |
| 验证集 F1 | 0.593 (20 epochs) |

### Omni 智能体 (云端)

| 项目 | 详情 |
|------|------|
| 模型 | Qwen3.5-Omni-Flash (阿里云 DashScope) |
| 类型 | 全模态大模型 (文本+图像+音频+视频) |
| 输入 | 2秒音频 (16kHz, mono, float32 → WAV base64) |
| 输出 | 转写文本 + 情绪标签 + 危险等级 + 判定理由 |
| 情绪 | 7类: fearful/angry/sad/disgusted/surprised/happy/neutral |
| 危险等级 | 3级: 危险 (≥0.60) / 关注 (≥0.30) / 正常 (<0.30) |
| 平均延迟 | ~800ms (2秒音频) |

### 双通路对比

| 对比维度 | emotion2vec (本地) | Omni (云端) |
|----------|:---:|:---:|
| 延迟 | < 50ms | ~800ms |
| 成本 | 免费 | ~¥7.2/小时 |
| 深度理解 | 声学特征 + 情绪嵌入 | 语义+情绪+语境 |
| 环境危险声 | 不支持 | ✅ 爆炸/撞击→危险 |
| 离线可用 | ✅ | ❌ 需网络 |
| F1 分数 | **0.867** | 待评测 |

### 训练模型

```bash
# 1. 从现有 data/ 重建 train/val/test 划分
python scripts/rebuild_data.py

# 2. 提取 emotion2vec 768维嵌入 (GPU)
python scripts/extract_emotion2vec_embeddings.py --device cuda

# 3. 训练 FC 分类头
python scripts/train_emotion2vec_classifier.py --epochs 150

# CNN 训练 (备选)
python scripts/train_model_v4.py --epochs 100
```

## 五人分工

| 成员 | 模块 | 文件 |
|------|------|------|
| **A** | 音频采集与预处理 | `frontend/audio_capture.py`, `design-demos/frontend-panel.html` |
| **B** | 网络传输 | `frontend/audio_sender.py`, `backend/audio_receiver.py` |
| **C** | 特征提取与AI模型 | `backend/feature_extractor.py`, `backend/audio_classifier.py`, `scripts/train_emotion2vec_classifier.py`, `asr_emotion_agent/` |
| **D** | 报警逻辑与GUI | `backend/alert_manager.py`, `backend/gui_main_window.py` |
| **E** | 文件分析与集成 | `backend/file_handler.py`, `common/audio_config.py` |

## 技术栈

| 层级 | 技术 |
|------|------|
| 音频采集 | 浏览器 MediaRecorder API / PyAudio + webrtcvad |
| 网络传输 | WebSocket (websockets + asyncio) + HTTP multipart |
| 特征提取 | emotion2vec_plus_base (funasr) / librosa (mel spectrogram) |
| AI 模型 (本地) | emotion2vec FC 分类头 (230K 参数) / AudioCNN v4 (500K) |
| AI 模型 (云端) | Qwen3.5-Omni-Flash (OpenAI 兼容 API, DashScope) |
| 报警引擎 | 预测平滑 + 滑动窗口 + 三级报警 |
| 后端 GUI | PyQt5 |
| 前端面板 | HTML5 + CSS3 + JavaScript (Web Components) |
| 仪表盘 | HTML5 Canvas + WebSocket |

## 报警规则

- **预测平滑器 v2**: EMA 置信度平滑 (α=0.4) + 75% 多数投票 + margin 折扣
- **滑动窗口**: 15 帧窗口 (~0.45s)，异常占比 ≥ 40% 触发
- **三级阈值**: 危急 ≥0.75 | 警告 ≥0.55 | 预报警 ≥0.35
- **冷却时间**: 8秒 (普通) / 3秒 (危急)
- **语音检测**: 频谱平坦度 >0.65 视为噪声/静音，跳过模型直接判 normal

## 数据来源

| 数据集 | 用途 | 样本量 |
|--------|------|--------|
| CREMA-D | 情绪语音 (ANG/FEA/HAP/SAD) | ~7,442 |
| ESC-50 | 环境音 (crying_baby/laughing) | 80 |
| CASIA | 中文情绪语音 | 部分 |

## 配置说明

核心配置见 `config.yaml`：

```yaml
model:
  model_path: "models/best_model_e2v.pt"  # 当前最佳模型
  num_classes: 4                           # normal, scream, cry, laugh
  classifier_mode: "emotion"               # emotion2vec 模式

network:
  host: "0.0.0.0"
  port: 8765              # WebSocket 音频流
  server_url: "ws://localhost:8765"
```

## API 接口

### POST /api/analyze (文件分析)

```bash
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@recording.wav"
```

### POST /api/omni (Omni 智能体分析)

```bash
curl -X POST http://localhost:8080/api/omni \
  -F "file=@recording.wav"
```

## 双机部署

```
┌─────────────────────┐          ┌──────────────────────────┐
│   采集机 (前端)      │          │   推理机 (后端)           │
│  浏览器访问 :8080   │─────────▶│   GPU 服务器 / 高配 PC   │
│  USB 麦克风          │   HTTP   │   emotion2vec 推理        │
│                      │          │   报警引擎                │
└─────────────────────┘          └──────────────────────────┘
```

将前端机的 `config.yaml` 中 `server_url` 改为后端 IP 即可。
