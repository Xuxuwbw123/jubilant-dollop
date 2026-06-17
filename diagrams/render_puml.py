"""
渲染 PlantUML .puml 文件为 PNG 图片
使用 Kroki API (https://kroki.io) — 免费开源图表渲染服务
"""
import sys
from pathlib import Path
import requests
import zlib
import base64

ROOT = Path(__file__).resolve().parent
DIAGRAMS_DIR = ROOT
OUTPUT_DIR = ROOT / "png"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kroki API endpoint
KROKI_URL = "https://kroki.io/plantuml/png"

DIAGRAMS = {
    "system_architecture": "系统整体架构图",
    "data_flow": "音频数据流详解",
    "cnn_architecture": "CNN模型结构",
    "alert_engine": "报警引擎流程",
    "file_analysis": "文件分析流水线",
    "omni_agent": "Omni智能体分析流程",
    "deployment": "双机部署架构",
}

def plantuml_to_kroki(text: str) -> str:
    """PlantUML → Kroki PNG URL via deflate + base64url"""
    # Kroki expects deflate-compressed + base64url-encoded PlantUML
    compressed = zlib.compress(text.encode('utf-8'), level=9)
    # Strip the 2-byte zlib header for raw deflate
    encoded = base64.urlsafe_b64encode(compressed[2:-4]).decode('ascii')
    return encoded

def render_diagram(name: str, title: str):
    puml_path = DIAGRAMS_DIR / f"{name}.puml"
    if not puml_path.exists():
        print(f"  [SKIP] {puml_path} not found")
        return None

    text = puml_path.read_text(encoding='utf-8')

    try:
        encoded = plantuml_to_kroki(text)
        url = f"{KROKI_URL}/{encoded}"

        print(f"  Rendering: {title}...", end=" ", flush=True)
        resp = requests.get(url, timeout=30)

        if resp.status_code == 200:
            output_path = OUTPUT_DIR / f"{name}.png"
            output_path.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            return output_path
        else:
            print(f"FAIL (HTTP {resp.status_code})")
            # Try alternative: plantuml.com public server
            return render_via_plantuml_server(name, title, text)
    except Exception as e:
        print(f"ERROR: {e}")
        return render_via_plantuml_server(name, title, text)


def render_via_plantuml_server(name: str, title: str, text: str):
    """Fallback: 使用 PlantUML 官方服务器"""
    try:
        url = "https://www.plantuml.com/plantuml/png"
        # PlantUML uses a different encoding (hex + deflate)
        import zlib
        compressed = zlib.compress(text.encode('utf-8'), level=9)[2:-4]

        # Encode in plantuml's custom format
        def encode64(data):
            res = ""
            for i in range(0, len(data), 3):
                if i + 2 < len(data):
                    n = (data[i] << 16) | (data[i+1] << 8) | data[i+2]
                elif i + 1 < len(data):
                    n = (data[i] << 16) | (data[i+1] << 8)
                else:
                    n = (data[i] << 16)
                res += _plantuml_chars[(n >> 18) & 0x3F]
                res += _plantuml_chars[(n >> 12) & 0x3F]
                res += _plantuml_chars[(n >> 6) & 0x3F] if i + 2 < len(data) or i + 1 < len(data) else ''
                res += _plantuml_chars[n & 0x3F] if i + 2 < len(data) else ''
            return res

        encoded = encode64(compressed)
        resp = requests.get(f"{url}/{encoded}", timeout=30)

        if resp.status_code == 200:
            output_path = OUTPUT_DIR / f"{name}.png"
            output_path.write_bytes(resp.content)
            print(f"OK via plantuml.com ({len(resp.content)/1024:.0f} KB)")
            return output_path
    except Exception as e:
        print(f"FAIL: {e}")
    return None


# PlantUML 自定义编码表
_plantuml_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def main():
    print("=" * 50)
    print("  PlantUML -> PNG Render")
    print("=" * 50)
    print(f"  Source: {DIAGRAMS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Diagrams: {len(DIAGRAMS)}\n")

    results = {}
    for name, title in DIAGRAMS.items():
        result = render_diagram(name, title)
        results[name] = result

    print(f"\n{'='*50}")
    success = sum(1 for r in results.values() if r is not None)
    print(f"  Done: {success}/{len(DIAGRAMS)} diagrams")
    print(f"  Output: {OUTPUT_DIR}")
    for name, path in results.items():
        if path:
            print(f"    [OK] {name}.png")
        else:
            print(f"    [FAIL] {name} - render failed, use .puml manually")
    print(f"\n  Sources: {DIAGRAMS_DIR}")
    print(f"  Online editor: https://www.plantuml.com/plantuml/uml/")


if __name__ == "__main__":
    main()
