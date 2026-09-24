# HƯỚNG DẪN CÀI ĐẶT & CHẠY TRỢ LÝ GIỌNG NÓI XE HƠI (OFFLINE - WINDOWS CUDA)

Tài liệu này hướng dẫn chi tiết từng bước cài đặt môi trường, tải các model và khởi chạy ứng dụng **Test Bench GUI** mô phỏng trợ lý giọng nói trên ô tô chạy **100% OFFLINE** trên Windows với GPU NVIDIA (CUDA).

---

## 📁 1. Cấu trúc thư mục dự án

Để ứng dụng tự động nhận diện các file mà không cần nhập lại đường dẫn thủ công, cấu trúc thư mục nên như sau:

```text
d:\Test_voice\
│
├── app.py                      # File mã nguồn chính chứa toàn bộ logic & Gradio UI
├── requirements.txt            # Danh sách thư viện Python
├── download_assets.py          # Script hỗ trợ tải tự động các models & piper.exe
├── SETUP_GUIDE.md              # Hướng dẫn chi tiết này
│
├── models\                     # Thư mục chứa các weights model
│   ├── Qwen2.5-3B-Instruct-Q4_K_M.gguf     # Core LLM 4-bit (~1.9 GB)
│   ├── vi_VN-vais1000-medium.onnx          # Model TTS tiếng Việt Piper
│   └── vi_VN-vais1000-medium.onnx.json     # File cấu hình giọng đọc TTS
│
├── piper\                      # Thư mục công cụ Piper TTS Windows
│   ├── piper.exe               # File thực thi Piper x64
│   ├── espeak-ng-data\         # Thư mục dữ liệu đi kèm của piper (nếu có)
│   └── ...                     # Các file dll đi kèm khi giải nén piper
│
└── temp_audio\                 # Thư mục tự động tạo chứa file .wav tạm thời
```

---

## ⚙️ 2. Cài đặt môi trường Python & CUDA

### Bước 2.1: Cài đặt PyTorch hỗ trợ CUDA
Mở PowerShell hoặc Command Prompt và kiểm tra CUDA:
```powershell
# Cài đặt PyTorch với CUDA (ví dụ CUDA 12.1 hoặc 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
> **Kiểm tra nhanh trong Python:**
> ```python
> import torch
> print(torch.cuda.is_available())  # Phải in ra True
> print(torch.cuda.get_device_name(0))
> ```

---

### Bước 2.2: Cài đặt `llama-cpp-python` có hỗ trợ CUDA trên Windows
`llama-cpp-python` cần build kèm CUDA/cuBLAS để có thể offload toàn bộ layers (`n_gpu_layers=-1`) lên VRAM GPU. Để tránh phải cài đặt Visual Studio C++ Compiler, bạn có thể cài đặt trực tiếp qua pre-compiled wheel:

```powershell
# Lựa chọn 1: Cài qua index URL wheel chính thức
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Lựa chọn 2: Nếu bạn sử dụng CUDA 12.2 hoặc mới hơn:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

---

### Bước 2.3: Cài đặt các thư viện còn lại
```powershell
pip install gradio transformers accelerate soundfile librosa numpy tqdm requests
```
Hoặc:
```powershell
pip install -r requirements.txt
```

---

## 📥 3. Tải Models & Piper TTS về chạy Offline

Bạn có thể chạy script tải tự động:
```powershell
python download_assets.py
```
Hoặc làm thủ công theo các bước dưới đây:

### 3.1. Tải Piper TTS cho Windows x64
1. Truy cập [Piper GitHub Releases](https://github.com/rhasspy/piper/releases/tag/2023.11.14-2).
2. Tải file `piper_windows_amd64.zip`.
3. Giải nén toàn bộ nội dung file zip vào thư mục `d:\Test_voice\piper\` sao cho file thực thi nằm tại:  
   `d:\Test_voice\piper\piper.exe`.

### 3.2. Tải Voice Model Tiếng Việt cho Piper TTS
Tải 2 file model tiếng Việt (giọng đọc VAIS tự nhiên) từ HuggingFace và lưu vào thư mục `models\`:
- [vi_VN-vais1000-medium.onnx](https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx)
- [vi_VN-vais1000-medium.onnx.json](https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json)

### 3.3. Tải Model LLM GGUF (Qwen2.5-3B-Instruct)
- Tải file model lượng tử 4-bit [Qwen2.5-3B-Instruct-Q4_K_M.gguf (~1.93 GB)](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf)
- Lưu file tải về vào thư mục: `d:\Test_voice\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf`.
> *Gợi ý:* Nếu GPU của bạn có VRAM nhỏ hơn (hoặc muốn tốc độ cực nhanh), có thể dùng bản `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~980 MB).

### 3.4. STT PhoWhisper Model
- Model `vinai/phowhisper-base` sẽ được thư viện `transformers` tự động tải về bộ nhớ đệm cache (`~/.cache/huggingface/hub`) trong lần chạy đầu tiên.
- Khi đã tải 1 lần, các lần sau ứng dụng sẽ chạy **100% offline hoàn toàn không cần kết nối Internet**.

---

## 🚀 4. Khởi chạy ứng dụng Test Bench

Mở terminal tại thư mục dự án và gõ:
```powershell
python app.py
```

Khi khởi chạy thành công:
1. Terminal sẽ hiển thị thông tin GPU NVIDIA, phiên bản CUDA, dung lượng VRAM hiện có.
2. Mở trình duyệt web theo địa chỉ:  
   👉 **`http://127.0.0.1:7860`**

---

## 🎯 5. Hướng dẫn sử dụng & Kiểm thử (Test Cases)

### Kịch bản 1: Tin nhắn Khẩn cấp từ Gia đình
1. Tại cột trái, chọn Preset: **🚨 Khẩn cấp (Gia đình)**.
2. Giữ nguyên ô ghi âm hoặc nói lệnh: *"Đọc và gợi ý giúp tôi"*.
3. Nhấn **🚀 KÍCH HOẠT HỆ THỐNG TRỢ LÝ**.
4. **Quan sát kết quả ở cột phải:**
   - Thẻ phân tích đổi nhãn đỏ **🚨 KHẨN CẤP**.
   - Tóm tắt cực ngắn (< 15 từ): *"Gia đình bị va quẹt xe nhẹ, cần gọi lại gấp"*.
   - Gợi ý phản hồi nhanh an toàn: *"1. Tôi đang lái xe về ngay | 2. Đã gọi hỗ trợ"*.
   - Loa xe tự động phát giọng đọc thông báo.
   - Bảng **Latency Breakdown** đo lường mili-giây từng chặng (STT ms, LLM ms, TTS ms, Total ms).

### Kịch bản 2: Ra lệnh bằng giọng nói (Microphone)
1. Bấm nút Record trên component âm thanh ở cột trái.
2. Nói một câu ngắn bằng tiếng Việt: *"Tôi đang bận lái xe trên cao tốc, nhắn lại là tôi sẽ gọi lại sau 30 phút"*.
3. Bấm **🚀 KÍCH HOẠT HỆ THỐNG TRỢ LÝ**.
4. **PhoWhisper** sẽ chuyển lời nói thành văn bản, LLM kết hợp với tin nhắn đến để tạo câu phản hồi chính xác.

---

## 🧹 6. Cơ chế dọn dẹp bộ nhớ đệm tự động
- Ứng dụng tự động dọn dẹp các file `.wav` tạm thời sinh ra trong thư mục `temp_audio/` sau mỗi lần chạy (xóa file cũ hơn 10 phút) và xóa sạch toàn bộ khi tắt ứng dụng (`atexit`).
