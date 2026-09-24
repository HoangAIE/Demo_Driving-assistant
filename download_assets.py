"""
Script tải và chuẩn bị các model, công cụ phục vụ Test Bench:
1. Piper TTS Windows x64 binary (piper.exe)
2. Model giọng đọc tiếng Việt Piper (vi_VN-vais1000-medium.onnx & .json)
3. Model LLM GGUF (Qwen2.5-3B-Instruct-Q4_K_M.gguf)
4. Tải trước model STT PhoWhisper (vinai/phowhisper-base) về local cache
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PIPER_DIR = BASE_DIR / "piper"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPER_DIR.mkdir(parents=True, exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, output_path: Path, desc: str):
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"✅ Đã có sẵn: {output_path.name} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
        return
    print(f"📥 Đang tải {desc}: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"✨ Hoàn tất: {output_path.name}")

def setup_piper():
    print("\n" + "="*50)
    print("1. THIẾT LẬP PIPER TTS (WINDOWS x64)")
    print("="*50)
    piper_exe = PIPER_DIR / "piper.exe"
    if piper_exe.exists():
        print(f"✅ Đã có sẵn: {piper_exe}")
    else:
        piper_zip_url = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
        piper_zip = PIPER_DIR / "piper_windows_amd64.zip"
        download_file(piper_zip_url, piper_zip, "Piper Windows x64 Release")
        print("📦 Đang giải nén Piper...")
        with zipfile.ZipFile(piper_zip, 'r') as zip_ref:
            zip_ref.extractall(PIPER_DIR)
        
        # Nếu giải nén ra thư mục con piper/piper.exe, dời lên thư mục chính nếu cần
        nested_exe = PIPER_DIR / "piper" / "piper.exe"
        if nested_exe.exists() and not (PIPER_DIR / "piper.exe").exists():
            for f in (PIPER_DIR / "piper").glob("*"):
                f.rename(PIPER_DIR / f.name)
        print("✅ Thiết lập piper.exe thành công!")

    # Tải voice model tiếng Việt
    print("\n--- Tải Piper Vietnamese Voice Model (vais1000) ---")
    onnx_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
    json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"
    download_file(onnx_url, MODELS_DIR / "vi_VN-vais1000-medium.onnx", "Vietnamese TTS Model (ONNX)")
    download_file(json_url, MODELS_DIR / "vi_VN-vais1000-medium.onnx.json", "Vietnamese TTS Config (JSON)")

def setup_llm():
    print("\n" + "="*50)
    print("2. THIẾT LẬP CORE LLM GGUF (QWEN2.5-3B-INSTRUCT Q4_K_M)")
    print("="*50)
    gguf_url = "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    gguf_path = MODELS_DIR / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    print("Model GGUF (~1.9 GB) tối ưu cho GPU VRAM 4GB-8GB trở lên.")
    choice = input("Bạn có muốn tải model GGUF này ngay bây giờ? (y/n) [mặc định y]: ").strip().lower()
    if choice in ("", "y", "yes"):
        download_file(gguf_url, gguf_path, "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    else:
        print(f"ℹ️ Bạn có thể tải thủ công và đặt vào: {gguf_path}")

def setup_stt():
    print("\n" + "="*50)
    print("3. TẢI CACHE CHO STT PHOWHISPER (vinai/phowhisper-base)")
    print("="*50)
    choice = input("Bạn có muốn tải trước PhoWhisper qua HuggingFace transformers? (y/n) [mặc định y]: ").strip().lower()
    if choice in ("", "y", "yes"):
        try:
            from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq
            model_id = "vinai/phowhisper-base"
            print(f"📥 Đang tải {model_id}...")
            AutoTokenizer.from_pretrained(model_id)
            AutoFeatureExtractor.from_pretrained(model_id)
            AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            print("✅ Đã lưu cache PhoWhisper thành công!")
        except Exception as e:
            print(f"⚠️ Lỗi khi tải PhoWhisper: {e}")
            print("Hệ thống sẽ tự động tải khi lần đầu khởi chạy app.py nếu có kết nối mạng.")
    else:
        print("ℹ️ Bỏ qua tải trước STT.")

if __name__ == "__main__":
    print("=== TIỆN ÍCH TẢI ASSETS CHO IN-CAR VOICE ASSISTANT ===")
    setup_piper()
    setup_llm()
    setup_stt()
    print("\n🎉 Hoàn thành thiết lập assets! Bây giờ bạn có thể khởi chạy: python app.py")
