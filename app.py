#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IN-CAR VOICE ASSISTANT TEST BENCH (OFFLINE & CUDA)
==================================================
Mô phỏng trợ lý giọng nói trên xe hơi với luồng hội thoại thực tế:

  1. Tin nhắn đến xe → LLM phân tích & Piper TTS đọc to tóm tắt.
  2. Trợ lý hỏi: "Bạn có muốn trả lời không?"
  3. Tài xế nói Có / Không (hoặc bấm nút mô phỏng giọng nói).
  4. Nếu Có:
     - Trường hợp 4A: Tài xế đọc câu trả lời → Trợ lý đọc lại và hỏi xác nhận.
     - Trường hợp 4B: Tài xế nghĩ một lúc mà chưa nói (im lặng) →
       Trợ lý tự động kích hoạt "Chế độ gợi ý", đọc to 3 lựa chọn phản hồi nhanh
       và tài xế có thể chọn bằng giọng nói (Số 1, Số 2, Số 3) hoặc bấm chọn.
  5. Xác nhận gửi: Trợ lý đọc lại nội dung phản hồi và hỏi: "Xác nhận gửi không?"
  6. Tài xế đồng ý → Gửi thành công và nhắc nhở lái xe an toàn.

Engines: PhoWhisper STT (CUDA) | Qwen2.5-3B GGUF (GPU) | Piper TTS (Offline)
"""

import os
import sys
import time
import json
import re
import atexit
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

# Reconfigure console encoding for Windows UTF-8 support
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import gradio as gr

# Register PyTorch DLL directory for llama-cpp on Windows
try:
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(_torch_lib):
        os.add_dll_directory(_torch_lib)
except Exception:
    pass

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THƯ MỤC
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PIPER_DIR = BASE_DIR / "piper"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPER_DIR.mkdir(parents=True, exist_ok=True)
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GGUF_PATH = str(MODELS_DIR / "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
DEFAULT_PIPER_EXE = str(PIPER_DIR / "piper.exe")
DEFAULT_TTS_ONNX = str(MODELS_DIR / "vi_VN-vais1000-medium.onnx")
DEFAULT_STT_MODEL = "vinai/phowhisper-base"

# ==============================================================================
# 2. DỌN DẸP FILE TẠM & KIỂM TRA PHẦN CỨNG
# ==============================================================================
def cleanup_temp_files(max_age_seconds: int = 1800):
    """Chỉ xóa các file âm thanh tạm cũ hơn max_age_seconds (không xóa file đang dùng)."""
    try:
        now = time.time()
        for f in TEMP_AUDIO_DIR.glob("*.wav"):
            if now - f.stat().st_mtime > max_age_seconds:
                try:
                    f.unlink()
                except Exception:
                    pass
    except Exception:
        pass

atexit.register(lambda: cleanup_temp_files(max_age_seconds=1800))

def get_hardware_status() -> Dict[str, Any]:
    cuda_ok = torch.cuda.is_available()
    status = {
        "cuda_available": cuda_ok,
        "device_name": torch.cuda.get_device_name(0) if cuda_ok else "CPU Mode",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "cuda_version": torch.version.cuda if cuda_ok else "N/A",
    }
    if cuda_ok:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            status["vram_total_gb"] = round(total_b / (1024**3), 2)
            status["vram_free_gb"] = round(free_b / (1024**3), 2)
        except Exception:
            pass
    return status

HW = get_hardware_status()

# ==============================================================================
# 3. MODEL MANAGER (Singleton loaders STT & LLM)
# ==============================================================================
class ModelManager:
    _stt_pipe = None
    _llm = None

    @classmethod
    def get_stt(cls):
        if cls._stt_pipe is None:
            from transformers import pipeline
            dev = 0 if HW["cuda_available"] else -1
            print(f"[STT] Loading PhoWhisper on device={dev}...")
            cls._stt_pipe = pipeline(
                "automatic-speech-recognition",
                model=DEFAULT_STT_MODEL,
                device=dev,
                dtype=torch.float16 if HW["cuda_available"] else torch.float32,
            )
            print("[STT] Ready!")
        return cls._stt_pipe

    @classmethod
    def get_llm(cls):
        if cls._llm is None:
            if not os.path.exists(DEFAULT_GGUF_PATH):
                raise FileNotFoundError(f"GGUF not found: {DEFAULT_GGUF_PATH}")
            from llama_cpp import Llama
            layers = -1 if HW["cuda_available"] else 0
            print(f"[LLM] Loading GGUF (n_gpu_layers={layers})...")
            cls._llm = Llama(
                model_path=DEFAULT_GGUF_PATH,
                n_gpu_layers=layers,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )
            print("[LLM] Ready!")
        return cls._llm

# ==============================================================================
# 4. ENGINE WRAPPERS: STT / LLM / TTS
# ==============================================================================
def engine_stt(audio_path: Optional[str]) -> Tuple[str, float]:
    """Chuyển giọng nói tài xế thành văn bản (không cần ffmpeg hệ thống)."""
    if not audio_path or not os.path.exists(audio_path):
        return "", 0.0
    t0 = time.perf_counter()
    try:
        pipe = ModelManager.get_stt()

        # Tự giải mã file âm thanh bằng soundfile / librosa để tránh lỗi thiếu ffmpeg
        import soundfile as sf
        import librosa

        try:
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Chuyển stereo sang mono
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
        except Exception:
            audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

        inputs = {"raw": audio_data, "sampling_rate": sample_rate}
        result = pipe(inputs, generate_kwargs={"language": "vi", "task": "transcribe"})
        text = result.get("text", "").strip()
    except Exception as e:
        print(f"[STT Error] {e}")
        text = ""
    return text, round((time.perf_counter() - t0) * 1000, 1)

def engine_llm_analyze(message: str) -> Tuple[Dict, float]:
    """LLM phân tích tin nhắn đến: phân loại, tóm tắt, 3 gợi ý ngắn và câu nói của trợ lý."""
    t0 = time.perf_counter()
    system = (
        "Bạn là Trợ lý AI trên ô tô, phân tích tin nhắn gửi đến cho tài xế đang lái xe.\n"
        "Hãy xuất DUY NHẤT một chuỗi JSON hợp lệ với cấu trúc sau:\n"
        "{\n"
        '  "category": "khan_cap" | "cong_viec" | "ban_be" | "quang_cao",\n'
        '  "summary": "Tóm tắt ngắn gọn dưới 15 từ",\n'
        '  "suggested_replies": [\n'
        '    "Gợi ý trả lời 1 (dưới 10 từ)",\n'
        '    "Gợi ý trả lời 2 (dưới 10 từ)",\n'
        '    "Gợi ý trả lời 3 (dưới 10 từ)"\n'
        '  ],\n'
        '  "assistant_speech": "Câu trợ lý đọc to: tóm tắt nội dung tin nhắn và hỏi tài xế có muốn trả lời không"\n'
        "}\n"
        "KHÔNG xuất markdown, KHÔNG thêm lời giải thích nào khác."
    )
    try:
        llm = ModelManager.get_llm()
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"TIN NHẮN ĐẾN:\n{message}"}
            ],
            temperature=0.2,
            max_tokens=450,
            response_format={"type": "json_object"},
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0) if m else raw)

        # Chuẩn hóa dữ liệu
        if data.get("category") not in ("khan_cap", "cong_viec", "ban_be", "quang_cao"):
            data["category"] = "cong_viec"
        if not data.get("summary"):
            data["summary"] = "Có tin nhắn mới gửi đến."
        if not isinstance(data.get("suggested_replies"), list) or len(data["suggested_replies"]) < 3:
            data["suggested_replies"] = [
                "Tôi đang lái xe, sẽ liên hệ lại sau.",
                "Tôi đã nhận được tin nhắn, đồng ý nhé.",
                "Để tôi kiểm tra lại rồi báo sau."
            ]
        if not data.get("assistant_speech"):
            data["assistant_speech"] = f"Bạn có tin nhắn mới: {data['summary']}. Bạn có muốn trả lời không?"
    except Exception as e:
        print(f"[LLM Error] {e}")
        data = {
            "category": "cong_viec",
            "summary": "Có tin nhắn mới",
            "suggested_replies": [
                "Tôi đang lái xe, sẽ gọi lại sau.",
                "OK, tôi đã nhận được thông tin.",
                "Để tôi xem lại rồi báo sau."
            ],
            "assistant_speech": "Bạn có tin nhắn mới. Bạn có muốn trả lời không?"
        }
    return data, round((time.perf_counter() - t0) * 1000, 1)

def engine_tts(text: str) -> Tuple[Optional[str], float]:
    """Phát giọng đọc tiếng Việt bằng Piper TTS offline."""
    if not text:
        return None, 0.0
    t0 = time.perf_counter()
    cleanup_temp_files(max_age_seconds=600)
    out_path = TEMP_AUDIO_DIR / f"voice_{int(time.time()*1000)}.wav"
    clean = text.replace('"', '').replace('\n', ' ').strip()

    piper = Path(DEFAULT_PIPER_EXE)
    onnx = Path(DEFAULT_TTS_ONNX)
    if piper.exists() and onnx.exists():
        try:
            proc = subprocess.Popen(
                [str(piper), "--model", str(onnx), "--output_file", str(out_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
            )
            proc.communicate(input=clean, timeout=15)
            if out_path.exists() and out_path.stat().st_size > 0:
                return str(out_path), round((time.perf_counter() - t0) * 1000, 1)
        except Exception as e:
            print(f"[TTS Error] {e}")

    # Fallback âm thanh nhẹ nếu thiếu Piper
    try:
        import numpy as np
        import soundfile as sf
        sr = 22050
        t_arr = np.linspace(0, 0.6, int(sr * 0.6), False)
        audio = np.sin(2 * np.pi * 587 * t_arr) * 0.25
        sf.write(str(out_path), audio, sr)
        return str(out_path), round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        return None, 0.0

# ==============================================================================
# 5. BỘ PHÂN TÍCH Ý ĐỊNH GIỌNG NÓI (INTENT PARSER)
# ==============================================================================
def parse_yes_or_no(text: str) -> Optional[bool]:
    """Phân tích câu nói có phải là Đồng ý (Yes) hay Từ chối (No)."""
    t = text.lower().strip()
    yes_words = [
        "có", "co", "ừ", "uh", "ok", "được", "đồng ý", "dong y", "gửi", "gui",
        "gửi đi", "yes", "yeah", "yep", "đúng", "rồi", "chắc chắn", "xác nhận",
        "gửi luôn", "tót", "cót", "muốn", "trả lời", "chuẩn", "chính xác"
    ]
    no_words = [
        "không", "khong", "thôi", "no", "nope", "bỏ qua", "hủy", "cancel",
        "đừng", "không cần", "khỏi", "để sau", "bỏ", "chưa", "sửa"
    ]
    for w in yes_words:
        if w in t:
            return True
    for w in no_words:
        if w in t:
            return False
    return None

def parse_suggestion_choice(text: str, suggestions: List[str]) -> Optional[str]:
    """
    Phân tích câu nói khi ở chế độ gợi ý:
    - Nếu tài xế nói "Số 1", "Một", "Phương án 1"... → Chọn gợi ý 1
    - Tương tự cho số 2, số 3
    - Nếu câu nói dài (>2 từ) và không chứa từ chọn số → coi như tự nói câu trả lời mới!
    """
    t = text.lower().strip()
    c1_words = ["1", "một", "mot", "nhất", "số 1", "gợi ý 1", "phương án 1", "câu 1", "đầu tiên"]
    c2_words = ["2", "hai", "nhì", "số 2", "gợi ý 2", "phương án 2", "câu 2", "thứ hai"]
    c3_words = ["3", "ba", "tam", "số 3", "gợi ý 3", "phương án 3", "câu 3", "thứ ba"]

    for w in c1_words:
        if w in t and len(t.split()) <= 4:
            return suggestions[0] if len(suggestions) > 0 else None
    for w in c2_words:
        if w in t and len(t.split()) <= 4:
            return suggestions[1] if len(suggestions) > 1 else None
    for w in c3_words:
        if w in t and len(t.split()) <= 4:
            return suggestions[2] if len(suggestions) > 2 else None

    # Nếu câu nói dài, coi như tài xế tự nói câu trả lời mới
    if len(t) > 3:
        return text
    return None

# ==============================================================================
# 6. CONVERSATION STATE & TEMPLATES
# ==============================================================================
# Phases:
#   "idle"           : Chưa có tin nhắn đến
#   "msg_read"       : Đã đọc tin nhắn, đang chờ xác nhận: "Có muốn trả lời không?"
#   "waiting_reply"  : Tài xế muốn trả lời, đang chờ tài xế nói câu trả lời (hoặc phân vân)
#   "suggest_mode"   : Tài xế nghĩ 1 lúc chưa nói → Vào chế độ đọc 3 gợi ý để tài xế chọn
#   "confirm_send"   : Đã có câu trả lời, trợ lý đọc lại và chờ xác nhận: "Có chắc gửi không?"
#   "done"           : Hoàn tất phiên, sẵn sàng cho tin nhắn tiếp theo

def make_empty_state():
    return {
        "phase": "idle",
        "analysis": None,
        "driver_reply": "",
        "chat_log": [],
        "latency": {"stt": 0, "llm": 0, "tts": 0},
    }

CATEGORY_LABELS = {
    "khan_cap": ("🚨 KHẨN CẤP", "#ef4444"),
    "cong_viec": ("💼 CÔNG VIỆC", "#3b82f6"),
    "ban_be": ("☕ BẠN BÈ", "#10b981"),
    "quang_cao": ("📢 QUẢNG CÁO", "#f59e0b"),
}

PRESET_MESSAGES = {
    "🚨 Khẩn cấp": "Anh ơi xe nhà mình bị va quẹt nhẹ ở ngã tư Hàng Xanh, người không sao nhưng công an đang tới, anh gọi lại gấp cho em với!",
    "💼 Công việc": "Tuấn ơi, đối tác vừa gửi email phản hồi hợp đồng cần ký duyệt trước 5h30 chiều nay. Em vào hệ thống kiểm tra và xác nhận sớm giúp anh nhé.",
    "☕ Bạn bè": "Hôm nay tan ca có rảnh không bạn ơi? Mấy anh em đang ngồi ở quán nhậu đường D2, ghé làm vài ly bia chém gió nha!",
    "📢 Quảng cáo": "TRI ÂN KHÁCH HÀNG: Giảm ngay 70% dịch vụ phủ gốm Ceramic và dán cách nhiệt ô tô chính hãng chỉ trong hôm nay.",
}

# ==============================================================================
# 7. GIAO DIỆN HTML RENDERING
# ==============================================================================
def render_chat_log(chat_log: List, phase: str) -> str:
    html = '<div class="chat-timeline">'
    for role, text in chat_log:
        if role == "assistant":
            html += f'''
            <div class="chat-bubble assistant-bubble">
                <div class="bubble-label">🚘 TRỢ LÝ XE HƠI</div>
                <div class="bubble-text">{text}</div>
            </div>'''
        elif role == "driver":
            html += f'''
            <div class="chat-bubble driver-bubble">
                <div class="bubble-label">🧑 TÀI XẾ</div>
                <div class="bubble-text">{text}</div>
            </div>'''
        elif role == "system":
            html += f'''
            <div class="chat-bubble system-bubble">
                <div class="bubble-text">{text}</div>
            </div>'''

    phase_banners = {
        "idle": ("⏸️ ĐANG LÁI XE BÌNH THƯỜNG — Chờ tin nhắn gửi đến xe...", "#64748b"),
        "msg_read": ("🎤 TRỢ LÝ HỎI: Bạn có muốn trả lời tin nhắn này không? (Nói: Có hoặc Không)", "#38bdf8"),
        "waiting_reply": ("🎤 TRỢ LÝ LẮNG NGHE: Hãy đọc câu trả lời, HOẶC nếu phân vân sẽ vào chế độ gợi ý...", "#10b981"),
        "suggest_mode": ("💡 CHẾ ĐỘ GỢI Ý ĐANG BẬT: Trợ lý đã đọc 3 gợi ý. Nói 'Số 1', 'Số 2', 'Số 3' hoặc bấm nút bên dưới!", "#f59e0b"),
        "confirm_send": ("🎤 TRỢ LÝ HỎI LẦN CUỐI: Bạn có xác nhận gửi tin nhắn này không? (Nói: Gửi / Đồng ý hoặc Hủy)", "#ec4899"),
        "done": ("✅ PHIÊN HỘI THOẠI ĐÃ KẾT THÚC — Chúc bạn lái xe an toàn!", "#22c55e"),
    }
    banner_txt, banner_color = phase_banners.get(phase, ("", "#64748b"))
    if banner_txt:
        html += f'''
        <div class="phase-banner" style="border-color:{banner_color}; color:{banner_color}">
            {banner_txt}
        </div>'''

    html += '</div>'
    return html

def render_analysis_card(analysis: Optional[Dict]) -> str:
    if not analysis:
        return '<div class="analysis-card empty">Chưa có tin nhắn phân tích</div>'
    cat = analysis.get("category", "cong_viec")
    label, color = CATEGORY_LABELS.get(cat, ("ℹ️ THÔNG BÁO", "#64748b"))
    summary = analysis.get("summary", "")
    return f'''
    <div class="analysis-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="color:#94a3b8;font-size:0.8rem;font-weight:700">PHÂN LOẠI TIN NHẮN</span>
            <span style="background:{color};color:#fff;padding:3px 10px;border-radius:6px;font-size:0.8rem;font-weight:700">{label}</span>
        </div>
        <div style="font-size:1.05rem;color:#38bdf8;font-weight:600;line-height:1.4">"{summary}"</div>
    </div>'''

def render_latency(lat: Dict) -> str:
    total = lat.get("stt", 0) + lat.get("llm", 0) + lat.get("tts", 0)
    return f'''
    <div class="latency-grid">
        <div class="lat-box"><div class="lat-lbl">STT</div><div class="lat-val">{lat.get("stt",0):.0f}<span class="lat-unit">ms</span></div></div>
        <div class="lat-box"><div class="lat-lbl">LLM</div><div class="lat-val">{lat.get("llm",0):.0f}<span class="lat-unit">ms</span></div></div>
        <div class="lat-box"><div class="lat-lbl">TTS</div><div class="lat-val">{lat.get("tts",0):.0f}<span class="lat-unit">ms</span></div></div>
        <div class="lat-box total"><div class="lat-lbl">TỔNG CỘNG</div><div class="lat-val">{total:.0f}<span class="lat-unit">ms</span></div></div>
    </div>'''

# ==============================================================================
# 8. STEP HANDLERS (LOGIC BƯỚC ĐI CỦA CONVERSATION FLOW)
# ==============================================================================

def step_message_arrives(message: str, state: Dict):
    """
    BƯỚC 1: Tin nhắn đến
    - LLM phân tích phân loại, tóm tắt và 3 gợi ý phản hồi nhanh.
    - TTS đọc to tóm tắt và hỏi: "Bạn có muốn trả lời không?"
    - Chuyển sang phase 'msg_read'.
    """
    if not message or not message.strip():
        return (state, render_chat_log(state["chat_log"], state["phase"]),
                "", None, render_latency(state["latency"]),
                gr.update(), gr.update(), gr.update(), gr.update(visible=False))

    state = make_empty_state()
    state["chat_log"].append(("system", f"📩 <b>Tin nhắn mới đến:</b> {message}"))

    # LLM phân tích
    analysis, llm_ms = engine_llm_analyze(message)
    state["analysis"] = analysis
    state["latency"]["llm"] = llm_ms

    # Trợ lý đọc to câu hỏi
    speech = analysis.get("assistant_speech", f"Bạn có tin nhắn mới: {analysis['summary']}. Bạn có muốn trả lời không?")
    state["chat_log"].append(("assistant", speech))
    audio_path, tts_ms = engine_tts(speech)
    state["latency"]["tts"] = tts_ms

    state["phase"] = "msg_read"

    sug = analysis.get("suggested_replies", ["", "", ""])
    sug1 = f"1️⃣ {sug[0]}" if len(sug) > 0 else "1️⃣ Đang lái xe, gọi sau"
    sug2 = f"2️⃣ {sug[1]}" if len(sug) > 1 else "2️⃣ OK, đã nhận"
    sug3 = f"3️⃣ {sug[2]}" if len(sug) > 2 else "3️⃣ Để tôi xem lại sau"

    return (
        state,
        render_chat_log(state["chat_log"], state["phase"]),
        render_analysis_card(state["analysis"]),
        audio_path,
        render_latency(state["latency"]),
        gr.update(value=sug1),
        gr.update(value=sug2),
        gr.update(value=sug3),
        gr.update(visible=False), # Chế độ gợi ý ban đầu ẩn
    )

def handle_driver_speech_or_text(input_text: str, state: Dict) -> Tuple:
    """
    Xử lý trung tâm cho mọi hành động của tài xế (dù nói qua Mic hoặc bấm nút mô phỏng).
    """
    phase = state.get("phase", "idle")
    analysis = state.get("analysis", {})
    suggestions = analysis.get("suggested_replies", []) if analysis else []

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: msg_read (Trợ lý đang hỏi Có / Không)
    # ─────────────────────────────────────────────────────────────
    if phase == "msg_read":
        yn = parse_yes_or_no(input_text)
        if yn is True:
            speech = "Vâng, xin mời bạn đọc câu trả lời. Nếu bạn đang bận hoặc phân vân, tôi sẽ gợi ý các câu trả lời nhanh."
            state["phase"] = "waiting_reply"
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=True)) # Hiện vùng gợi ý / nút phân vân
        elif yn is False:
            speech = "Dạ vâng, đã bỏ qua tin nhắn. Chúc bạn lái xe an toàn!"
            state["phase"] = "done"
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=False))
        else:
            speech = "Tôi chưa hiểu ý bạn. Bạn có muốn trả lời tin nhắn này không? Hãy nói Có hoặc Không."
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=False))

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: waiting_reply (Đang chờ tài xế đọc câu trả lời)
    # ─────────────────────────────────────────────────────────────
    elif phase == "waiting_reply":
        # Tài xế trực tiếp đọc câu trả lời
        state["driver_reply"] = input_text
        speech = f'Bạn muốn trả lời là: "{input_text}". Bạn có xác nhận gửi không?'
        state["phase"] = "confirm_send"
        state["chat_log"].append(("assistant", speech))
        audio, tts_ms = engine_tts(speech)
        state["latency"]["tts"] += tts_ms
        return (state, render_chat_log(state["chat_log"], state["phase"]),
                render_analysis_card(analysis), audio, render_latency(state["latency"]),
                gr.update(visible=False))

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: suggest_mode (Đang ở chế độ gợi ý phản hồi)
    # ─────────────────────────────────────────────────────────────
    elif phase == "suggest_mode":
        choice = parse_suggestion_choice(input_text, suggestions)
        if choice:
            state["driver_reply"] = choice
            speech = f'Bạn đã chọn trả lời là: "{choice}". Bạn có xác nhận gửi không?'
            state["phase"] = "confirm_send"
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=False))
        else:
            speech = "Bạn hãy chọn: Số 1, Số 2, Số 3, hoặc tự nói câu trả lời của bạn nhé."
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=True))

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: confirm_send (Đang chờ xác nhận gửi lần cuối)
    # ─────────────────────────────────────────────────────────────
    elif phase == "confirm_send":
        yn = parse_yes_or_no(input_text)
        if yn is True:
            reply = state.get("driver_reply", "")
            speech = f'Đã gửi tin nhắn: "{reply}". Chúc bạn lái xe an toàn!'
            state["phase"] = "done"
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=False))
        elif yn is False:
            speech = "Đã hủy gửi. Bạn hãy đọc lại câu trả lời mới hoặc chọn chế độ gợi ý nhé."
            state["driver_reply"] = ""
            state["phase"] = "waiting_reply"
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=True))
        else:
            speech = "Bạn có muốn gửi tin nhắn này không? Hãy nói Gửi hoặc Hủy."
            state["chat_log"].append(("assistant", speech))
            audio, tts_ms = engine_tts(speech)
            state["latency"]["tts"] += tts_ms
            return (state, render_chat_log(state["chat_log"], state["phase"]),
                    render_analysis_card(analysis), audio, render_latency(state["latency"]),
                    gr.update(visible=False))

    # Phase idle or done
    return (state, render_chat_log(state["chat_log"], state["phase"]),
            render_analysis_card(analysis), None, render_latency(state["latency"]),
            gr.update(visible=False))

def step_driver_speaks_mic(audio_path: Optional[str], state: Dict):
    """Tài xế nói qua Microphone."""
    if not state or state.get("phase") == "idle":
        return (state, render_chat_log(state.get("chat_log", []), "idle"),
                "", None, render_latency(state.get("latency", {})), gr.update())

    text, stt_ms = engine_stt(audio_path)
    state["latency"]["stt"] += stt_ms

    # Nếu không thu được tiếng hoặc tài xế im lặng:
    if not text or not text.strip():
        # ĐẶC BIỆT: Nếu đang ở waiting_reply mà tài xế im lặng / không nói gì
        # -> Kích hoạt chế độ gợi ý!
        if state["phase"] == "waiting_reply":
            state["chat_log"].append(("system", "⏳ <i>Tài xế đang suy nghĩ / chưa nói gì...</i>"))
            return trigger_suggestion_mode(state)

        state["chat_log"].append(("system", "⚠️ Không nhận diện được âm thanh, vui lòng nói lại."))
        speech = "Tôi chưa nghe rõ, bạn có thể nói lại không?"
        state["chat_log"].append(("assistant", speech))
        audio, tts_ms = engine_tts(speech)
        state["latency"]["tts"] += tts_ms
        return (state, render_chat_log(state["chat_log"], state["phase"]),
                render_analysis_card(state.get("analysis")), audio,
                render_latency(state["latency"]), gr.update())

    state["chat_log"].append(("driver", text))
    return handle_driver_speech_or_text(text, state)

def step_simulate_driver_speech(simulated_text: str, state: Dict):
    """Mô phỏng tài xế nói một câu thông qua nút bấm nhanh."""
    if not state or state.get("phase") == "idle":
        return (state, render_chat_log(state.get("chat_log", []), "idle"),
                "", None, render_latency(state.get("latency", {})), gr.update())
    state["chat_log"].append(("driver", simulated_text))
    return handle_driver_speech_or_text(simulated_text, state)

def trigger_suggestion_mode(state: Dict):
    """
    KÍCH HOẠT CHẾ ĐỘ GỢI Ý (SUGGESTION MODE):
    Xảy ra khi tài xế nghĩ một lúc mà chưa nói hoặc bấm nút 'Nghĩ 1 lúc / Chưa nói gì'.
    Trợ lý sẽ đọc to 3 lựa chọn và chuyển trạng thái sang 'suggest_mode'.
    """
    analysis = state.get("analysis") or {}
    sug = analysis.get("suggested_replies", [
        "Tôi đang lái xe, sẽ gọi lại sau",
        "OK, tôi đã nhận được thông tin",
        "Để tôi kiểm tra lại rồi báo sau"
    ])
    speech = (
        f"Tôi thấy bạn đang phân vân. Tôi có 3 gợi ý trả lời nhanh: "
        f"Số 1: {sug[0]}. "
        f"Số 2: {sug[1]}. "
        f"Số 3: {sug[2]}. "
        f"Bạn muốn chọn số 1, số 2 hay số 3, hoặc tự nói câu khác?"
    )
    state["phase"] = "suggest_mode"
    state["chat_log"].append(("assistant", speech))
    audio, tts_ms = engine_tts(speech)
    state["latency"]["tts"] += tts_ms
    return (
        state,
        render_chat_log(state["chat_log"], state["phase"]),
        render_analysis_card(analysis),
        audio,
        render_latency(state["latency"]),
        gr.update(visible=True), # Mở bảng gợi ý
    )

def step_driver_hesitates(state: Dict):
    """Mô phỏng trường hợp tài xế nghĩ một lúc mà chưa nói gì."""
    if not state or state.get("phase") not in ("waiting_reply", "msg_read"):
        return (state, render_chat_log(state.get("chat_log", []), state.get("phase", "idle")),
                render_analysis_card(state.get("analysis")), None, render_latency(state.get("latency", {})), gr.update())
    state["chat_log"].append(("system", "⏳ <i>Tài xế đang suy nghĩ / chưa nói gì...</i>"))
    return trigger_suggestion_mode(state)

def step_pick_suggestion_button(btn_value: str, state: Dict):
    """Tài xế bấm trực tiếp vào nút gợi ý trên màn hình."""
    # Bỏ tiền tố '1️⃣ ', '2️⃣ ', '3️⃣ '
    clean_text = re.sub(r'^[123]️⃣\s*', '', btn_value).strip()
    state["chat_log"].append(("driver", f"Đã chọn: \"{clean_text}\""))
    return handle_driver_speech_or_text(clean_text, state)

def step_reset(state: Dict):
    """Làm mới toàn bộ phiên hội thoại."""
    state = make_empty_state()
    return (
        state,
        render_chat_log([], "idle"),
        "",
        None,
        render_latency(state["latency"]),
        gr.update(value="1️⃣ Gợi ý 1"),
        gr.update(value="2️⃣ Gợi ý 2"),
        gr.update(value="3️⃣ Gợi ý 3"),
        gr.update(visible=False),
    )

# ==============================================================================
# 9. GIAO DIỆN GRADIO (COCKPIT INFOTAINMENT TEST BENCH)
# ==============================================================================
CSS = """
body { background: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* Cockpit Header */
.cockpit-hdr {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}
.cockpit-title {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hw-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 10px;
}
.hw-on { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
.hw-off { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

/* Chat Timeline */
.chat-timeline {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-height: 480px;
    overflow-y: auto;
    padding: 12px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
}
.chat-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 90%;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
.assistant-bubble {
    background: #1e3a5f;
    border: 1px solid #2563eb;
    align-self: flex-start;
}
.driver-bubble {
    background: #134e4a;
    border: 1px solid #0d9488;
    align-self: flex-end;
}
.system-bubble {
    background: #1f2937;
    border: 1px dashed #4b5563;
    align-self: center;
    text-align: center;
    font-size: 0.88rem;
    color: #cbd5e1;
    max-width: 98%;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 800;
    color: #94a3b8;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
}
.bubble-text {
    font-size: 0.98rem;
    color: #f8fafc;
}
.phase-banner {
    padding: 10px 16px;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid;
    text-align: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-top: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Analysis Card */
.analysis-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
}
.analysis-card.empty {
    color: #64748b;
    font-style: italic;
    text-align: center;
    padding: 20px;
}

/* Suggestion Buttons */
.sug-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 14px;
    margin-top: 10px;
}
.sug-btn {
    border: 1px solid #38bdf8 !important;
    background: #0f172a !important;
    color: #e0f2fe !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px !important;
    text-align: left !important;
}
.sug-btn:hover {
    background: #1e3a5f !important;
    border-color: #60a5fa !important;
}

/* Latency Box */
.latency-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}
.lat-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
}
.lat-box.total {
    border-color: #38bdf8;
    background: rgba(56, 189, 248, 0.08);
}
.lat-lbl {
    font-size: 0.72rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.5px;
}
.lat-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f1f5f9;
}
.lat-box.total .lat-val {
    color: #38bdf8;
}
.lat-unit {
    font-size: 0.72rem;
    font-weight: 500;
    color: #94a3b8;
    margin-left: 2px;
}

/* Simulation Action Bar */
.action-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
}
"""

def create_ui():
    with gr.Blocks(title="Trợ Lý Giọng Nói Xe Hơi", css=CSS, theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:

        conv_state = gr.State(make_empty_state())

        # ── Header ──
        hw_cls = "hw-on" if HW["cuda_available"] else "hw-off"
        hw_icon = "🟢" if HW["cuda_available"] else "⚠️"
        hw_txt = (f"{hw_icon} {HW['device_name']} | CUDA {HW['cuda_version']} | "
                  f"VRAM {HW['vram_free_gb']}GB free / {HW['vram_total_gb']}GB"
                  if HW["cuda_available"]
                  else f"{hw_icon} CPU Mode — Không tìm thấy GPU CUDA!")
        gr.HTML(f'''
        <div class="cockpit-hdr">
            <div class="cockpit-title">🚘 TRỢ LÝ GIỌNG NÓI XE HƠI — TEST BENCH</div>
            <div style="color:#94a3b8;font-size:0.92rem;margin-top:4px">
                Luồng hội thoại tự nhiên: Tin nhắn đến → Trợ lý đọc & hỏi → Tài xế trả lời / Nghĩ 1 lúc kích hoạt gợi ý → Xác nhận trước khi gửi.
            </div>
            <div class="hw-pill {hw_cls}">{hw_txt}</div>
        </div>''')

        with gr.Row(equal_height=False):

            # ══════════════ CỘT TRÁI: Mô phỏng đầu vào ══════════════
            with gr.Column(scale=5):

                gr.Markdown("### 📩 1. Giả lập tin nhắn đến xe")
                preset_dd = gr.Dropdown(choices=list(PRESET_MESSAGES.keys()),
                                        value="🚨 Khẩn cấp", label="Chọn tình huống tin nhắn mẫu:")
                msg_box = gr.Textbox(value=PRESET_MESSAGES["🚨 Khẩn cấp"],
                                     label="Nội dung tin nhắn đến xe:", lines=3)
                btn_incoming = gr.Button("📨 KÍCH HOẠT TIN NHẮN ĐẾN", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### 🎤 2. Giọng nói tài xế (Microphone)")
                mic_input = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                     label="Nhấn nút đỏ để ghi âm giọng nói:")
                btn_speak_mic = gr.Button("🗣️ GỬI ÂM THANH CHO TRỢ LÝ", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### 🕹️ 3. Phím tắt mô phỏng (Test nhanh)")
                with gr.Accordion("Bấm để xem các nút mô phỏng không cần bật Mic", open=True):
                    gr.Markdown("*(Dành cho việc kiểm thử logic nhanh trong trường hợp môi trường ồn hoặc không có mic)*")

                    with gr.Row():
                        btn_sim_yes = gr.Button("🗣️ Nói 'Có / Trả lời'")
                        btn_sim_no = gr.Button("🗣️ Nói 'Không / Bỏ qua'")

                    btn_hesitate = gr.Button("⏳ [Mô phỏng] Nghĩ một lúc / Chưa nói gì → Kích hoạt gợi ý!",
                                             variant="secondary")

                    with gr.Row():
                        btn_sim_confirm = gr.Button("🗣️ Nói 'Xác nhận gửi đi'")
                        btn_sim_cancel = gr.Button("🗣️ Nói 'Hủy / Sửa lại'")

                # Khối gợi ý nhanh (Xuất hiện khi tài xế nghĩ một lúc)
                with gr.Group(visible=False) as suggestion_group:
                    gr.Markdown("### 💡 4. Chế độ gợi ý phản hồi nhanh")
                    gr.Markdown("*Trợ lý đã đọc 3 gợi ý. Bạn có thể nói 'Số 1/2/3' vào mic hoặc bấm chọn:*")
                    with gr.Column():
                        btn_sug1 = gr.Button("1️⃣ Gợi ý 1", elem_classes=["sug-btn"])
                        btn_sug2 = gr.Button("2️⃣ Gợi ý 2", elem_classes=["sug-btn"])
                        btn_sug3 = gr.Button("3️⃣ Gợi ý 3", elem_classes=["sug-btn"])

                gr.Markdown("---")
                btn_reset = gr.Button("🔄 LÀM MỚI PHIÊN HỘI THOẠI", variant="secondary")

            # ══════════════ CỘT PHẢI: Diễn biến & Output ══════════════
            with gr.Column(scale=6):

                gr.Markdown("### 🗨️ Diễn biến hội thoại theo thời gian thực")
                chat_html = gr.HTML(render_chat_log([], "idle"))

                gr.Markdown("### 📊 Phân tích tin nhắn (Qwen2.5-3B)")
                analysis_html = gr.HTML(render_analysis_card(None))

                gr.Markdown("### 🔊 Giọng đọc trợ lý xe (Piper TTS)")
                audio_player = gr.Audio(label="Trợ lý đang phát âm thanh...", autoplay=True)

                gr.Markdown("### ⏱️ Thống kê độ trễ (Latency Breakdown)")
                latency_html = gr.HTML(render_latency({"stt": 0, "llm": 0, "tts": 0}))

        # ══════════════ EVENT WIRING ══════════════

        # Chọn tin nhắn mẫu
        preset_dd.change(fn=lambda k: PRESET_MESSAGES.get(k, ""), inputs=[preset_dd], outputs=[msg_box])

        # BƯỚC 1: Tin nhắn đến
        btn_incoming.click(
            fn=step_message_arrives,
            inputs=[msg_box, conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html,
                     btn_sug1, btn_sug2, btn_sug3, suggestion_group],
        )

        # Tài xế nói qua Mic
        btn_speak_mic.click(
            fn=step_driver_speaks_mic,
            inputs=[mic_input, conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )

        # Các nút mô phỏng nhanh
        btn_sim_yes.click(
            fn=lambda st: step_simulate_driver_speech("Có, tôi muốn trả lời", st),
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )
        btn_sim_no.click(
            fn=lambda st: step_simulate_driver_speech("Không, bỏ qua tin nhắn", st),
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )
        btn_hesitate.click(
            fn=step_driver_hesitates,
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )
        btn_sim_confirm.click(
            fn=lambda st: step_simulate_driver_speech("Đồng ý gửi đi", st),
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )
        btn_sim_cancel.click(
            fn=lambda st: step_simulate_driver_speech("Hủy gửi", st),
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
        )

        # Bấm chọn trực tiếp gợi ý
        for btn in [btn_sug1, btn_sug2, btn_sug3]:
            btn.click(
                fn=step_pick_suggestion_button,
                inputs=[btn, conv_state],
                outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html, suggestion_group],
            )

        # Reset
        btn_reset.click(
            fn=step_reset,
            inputs=[conv_state],
            outputs=[conv_state, chat_html, analysis_html, audio_player, latency_html,
                     btn_sug1, btn_sug2, btn_sug3, suggestion_group],
        )

    return demo

# ==============================================================================
# 10. MAIN LAUNCHER
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚘 TRỢ LÝ GIỌNG NÓI XE HƠI — TEST BENCH (100% OFFLINE)")
    print("=" * 60)
    print(f"CUDA: {HW['cuda_available']} | GPU: {HW['device_name']}")
    if HW["cuda_available"]:
        print(f"VRAM: {HW['vram_free_gb']} GB free / {HW['vram_total_gb']} GB total")
    print("\nKhởi chạy giao diện tại http://127.0.0.1:7860 ...\n")
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, share=False)
