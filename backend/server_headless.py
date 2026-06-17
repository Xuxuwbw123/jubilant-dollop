"""
Minimal HTTP backend — serves static files + emotion2vec inference API.
No PyQt5 dependency, perfect for testing.

Usage:
    python backend/server_headless.py
    python backend/server_headless.py --port 8080
"""
import os, sys, re, json, time, argparse, logging, tempfile
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('FUNASR_DISABLE_UPDATE', '1')

import numpy as np
import torch

from common.audio_config import ClassifyResult

# ── Logging ──
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("headless")

# ── Config ──
from backend.config import BackendConfig
config = BackendConfig()
CLASS_NAMES = config.class_names  # ["normal", "scream", "cry", "laugh"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info(f"Device: {DEVICE}, Classes: {CLASS_NAMES}")


# ═══════════════════════════════════════════════════════════
# 模型加载 — emotion2vec 特征提取器 + FC 分类头
# Emotion2VecWrapper: a 3-layer MLP head (768→256→128→4) with
# BatchNorm+Dropout, trained on frozen emotion2vec embeddings.
# Checkpoint contains model weights + metadata (val_f1, hyperparams).
# emotion2vec itself is lazy-loaded via funasr AutoModel on first use.
# ═══════════════════════════════════════════════════════════

# ── Load classifier ──
from backend.audio_classifier import Emotion2VecWrapper

model_path = ROOT / config.model_path
log.info(f"Loading: {model_path}")
ckpt = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
n_classes = len(CLASS_NAMES)
classifier = Emotion2VecWrapper(num_classes=n_classes).to(DEVICE)
classifier.load_state_dict(ckpt["model_state_dict"])
classifier.eval()
log.info(f"Classifier loaded (val_f1={ckpt['metadata']['val_f1']:.4f})")


# ── Lazy-load emotion2vec extractor ──
_e2v_model = None

def get_e2v():
    global _e2v_model
    if _e2v_model is None:
        from funasr import AutoModel
        _e2v_model = AutoModel(
            model="iic/emotion2vec_plus_base", hub="ms",
            device=DEVICE, disable_update=True)
        log.info("emotion2vec extractor loaded")
    return _e2v_model


# ═══════════════════════════════════════════════════════════
# 音频分类 — 单窗口推理 + 滑动窗口推理
# classify_audio: audio → emotion2vec embedding → FC head → softmax probs
# sliding_window_classify: sweeps a window across the full audio,
# aggregates per-class mean probabilities, returns timeline + summary.
# ═══════════════════════════════════════════════════════════

# ── Inference ──
def classify_audio(audio: np.ndarray, sr: int = 16000) -> dict:
    """Full pipeline: audio → embedding → FC → 4-class probabilities"""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Resample if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    e2v = get_e2v()
    result = e2v.generate(audio, granularity="utterance", extract_embedding=True)[0]
    embedding = np.array(result["feats"], dtype=np.float32)

    with torch.no_grad():
        x = torch.from_numpy(embedding).float().unsqueeze(0).to(DEVICE)
        logits = classifier(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        "dominant_emotion": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "emotion_distribution": {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        },
    }


def sliding_window_classify(audio: np.ndarray, sr: int = 16000,
                            window_sec: float = 1.0,
                            hop_sec: float = 0.5) -> dict:
    """Sliding-window classification over full audio."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    total_samples = len(audio)
    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    duration = total_samples / sr

    timeline = []
    all_probs = {c: [] for c in CLASS_NAMES}

    for start in range(0, max(1, total_samples - win_samples + 1), hop_samples):
        end = min(start + win_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < sr * 0.1:
            continue
        result = classify_audio(chunk, sr)
        result["start_s"] = round(start / sr, 1)
        result["end_s"] = round(end / sr, 1)
        timeline.append(result)
        for c in CLASS_NAMES:
            all_probs[c].append(result["emotion_distribution"].get(c, 0))

    # Aggregate
    if all_probs[CLASS_NAMES[0]]:
        avg_dist = {
            c: round(float(np.mean(all_probs[c])), 4)
            for c in CLASS_NAMES
        }
        best = max(avg_dist, key=avg_dist.get)
    else:
        avg_dist = {c: 0.0 for c in CLASS_NAMES}
        best = "normal"

    # Add silence
    avg_dist["silence"] = 0.0

    # Dominant based on max average
    return {
        "file": "",
        "duration_sec": round(duration, 1),
        "total_windows": len(timeline),
        "dominant_emotion": best,
        "confidence": round(avg_dist[best], 4),
        "emotion_distribution": avg_dist,
        "timeline": timeline,
    }


# ═══════════════════════════════════════════════════════════
# 音频解码 — ffmpeg + soundfile 多格式支持
# ffmpeg: decodes compressed formats (M4A/AAC/MP3/OGG/FLAC/Opus,
#         MP4/WEBM) to raw PCM via pipe → numpy int16 → float32
# soundfile: fallback for formats the system codec already supports
# Returns (float32 array, sample_rate); raises RuntimeError on failure.
# ═══════════════════════════════════════════════════════════

# ── Audio decoding ──
from backend.file_handler import extract_audio as decode_audio


# ═══════════════════════════════════════════════════════════
# HTTP API 处理器 — multipart 解析 + 临时文件 + CORS + /api/analyze
# ThreadingMixIn for concurrent request handling.
# APIHandler.do_POST: parses multipart/form-data, extracts the
# uploaded file, writes it to a temp file, runs decode_audio →
# sliding_window_classify, returns JSON with timeline + distribution.
# CORS headers set on all responses (OPTIONS handled explicitly).
# ═══════════════════════════════════════════════════════════

# ── HTTP Server ──
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_handler(serve_dir: Path):
    class APIHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir))

        def do_POST(self):
            if self.path == "/api/analyze":
                self._handle_analyze()
            elif self.path == "/api/omni":
                self._send_json({"error": "Omni not available in headless mode"})
            else:
                self.send_error(404, "Not found")

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _handle_analyze(self):
            content_type = self.headers.get("Content-Type", "")
            boundary_match = re.search(r"boundary=([^;]+)", content_type)
            if not boundary_match:
                self._send_json({"error": "multipart required"})
                return

            boundary = boundary_match.group(1).strip().strip('"')
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse multipart
            parts = body.split(f'--{boundary}'.encode())
            file_data = None
            filename = "unknown"
            for part in parts:
                if b'Content-Disposition' in part and b'filename=' in part:
                    header_end = part.find(b'\r\n\r\n')
                    if header_end == -1:
                        header_end = part.find(b'\n\n')
                    if header_end == -1:
                        continue
                    file_data = part[header_end:].strip()
                    # Extract filename
                    fn_match = re.search(rb'filename="([^"]*)"', part)
                    if fn_match:
                        filename = fn_match.group(1).decode('utf-8', errors='replace')

            if not file_data:
                self._send_json({"error": "no file"})
                return

            # Save temp
            suffix = Path(filename).suffix or '.wav'
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file_data)
            tmp.close()

            try:
                t0 = time.time()
                audio = decode_audio(tmp.name)
                if audio is None:
                    self._send_json({"error": f"Cannot decode: {filename}"})
                    return
                result = sliding_window_classify(audio, 16000)
                result["file"] = filename
                result["api_latency_ms"] = round((time.time() - t0) * 1000)
                self._send_json(result)
                log.info(f"[API] {filename} → {result['dominant_emotion']} "
                         f"({result['confidence']:.0%}) {result['api_latency_ms']:.0f}ms")
            except Exception as e:
                self._send_json({"error": str(e)})
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        def _send_json(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            log.debug(f"[HTTP] {args[0]}")

    return APIHandler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--dir", default=None, help="Static files directory")
    args = p.parse_args()

    # Serve directory
    if args.dir:
        serve_dir = Path(args.dir)
    else:
        serve_dir = ROOT / "design-demos"
    if not serve_dir.exists():
        serve_dir = ROOT / "frontend" / "html"

    log.info(f"Serving static files from: {serve_dir}")

    # Warmup
    log.info("Warmup inference...")
    t0 = time.time()
    warmup = np.random.randn(16000).astype(np.float32) * 0.01
    classify_audio(warmup)
    log.info(f"Warmup done ({time.time()-t0:.1f}s)")

    handler = make_handler(serve_dir)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)

    log.info(f"=" * 55)
    log.info(f"  Headless backend ready")
    log.info(f"  API:    http://localhost:{args.port}/api/analyze")
    log.info(f"  Panel:  http://localhost:{args.port}/frontend-panel.html")
    log.info(f"  Dashboard: http://localhost:{args.port}/backend-dashboard.html")
    log.info(f"=" * 55)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping...")
        server.shutdown()


if __name__ == "__main__":
    main()
