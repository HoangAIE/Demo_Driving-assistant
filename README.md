# 🚘 In-Car Voice Assistant Test Bench (Offline & CUDA)

Ứng dụng kiểm thử độc lập (Test Bench GUI) mô phỏng hệ thống trợ lý giọng nói trên xe hơi chạy **100% OFFLINE** trên hệ điều hành Windows với GPU NVIDIA (CUDA).

---

## 🌟 Tính năng chính

- **STT Engine (PhoWhisper):** Nhận diện giọng nói tiếng Việt chuẩn xác chạy trên CUDA Float16 (`vinai/phowhisper-base`).
- **Core LLM 3-in-1 Pipeline (Qwen2.5-3B-Instruct GGUF):**
  - Chạy qua `llama-cpp-python` với `n_gpu_layers=-1` offload 100% lên VRAM GPU.
  - Phân loại tin nhắn: `khan_cap`, `cong_viec`, `ban_be`, `quang_cao`.
  - Tóm tắt cực ngắn dưới 15 từ, phục vụ an toàn khi lái xe.
  - Gợi ý 2 câu trả lời nhanh (quick replies) có thể gửi ngay.
  - Tạo lời thoại giọng nói của xe để thông báo cho tài xế.
- **TTS Engine (Piper TTS):** Sinh âm thanh giọng đọc tiếng Việt offline siêu nhẹ với model `vi_VN-vais1000-medium.onnx`.
- **Giao diện Gradio Dashboard 2 cột:**
  - **Cột trái:** Hộp tin nhắn đến (với các preset khẩn cấp, công việc, bạn bè, quảng cáo) & Microphone thu âm giọng nói tài xế.
  - **Cột phải:** Kết quả STT, Thẻ JSON phân loại tin nhắn với nhãn màu trực quan, Trình phát âm thanh xe hơi, và Bảng đo độ trễ **Latency Breakdown** chi tiết (STT ms, LLM ms, TTS ms, Total ms).
- **Tự động dọn dẹp cache:** Quản lý và tự động xóa file âm thanh tạm thời trong `temp_audio/`.

---

## 🚀 Khởi chạy nhanh

1. Đọc hướng dẫn cài đặt thư viện & tải models tại: [SETUP_GUIDE.md](file:///d:/Test_voice/SETUP_GUIDE.md)
2. Tải nhanh các công cụ và models:
   ```powershell
   python download_assets.py
   ```
3. Khởi chạy ứng dụng:
   ```powershell
   python app.py
   ```
4. Truy cập giao diện tại: `http://127.0.0.1:7860`
