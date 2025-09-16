# 🚀 Hướng dẫn Cài đặt Open Vocabulary Detection

## 📋 Yêu cầu hệ thống

- **Python**: 3.8+ (khuyến nghị 3.9+)
- **CUDA**: 11.0+ (khuyến nghị 11.8)
- **RAM**: Tối thiểu 8GB
- **GPU**: Tối thiểu 6GB VRAM (khuyến nghị 8GB+)

## 🔧 Cài đặt tự động

### **Cách 1: Sử dụng script tự động**
```bash
python scripts/install_dependencies.py
```

### **Cách 2: Cài đặt thủ công**

#### **Bước 1: Cài đặt PyTorch**
```bash
# Cho CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cho CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cho CPU only
pip install torch torchvision torchaudio
```

#### **Bước 2: Cài đặt PyTorch Lightning**
```bash
pip install pytorch-lightning
```

#### **Bước 3: Cài đặt các thư viện khác**
```bash
pip install timm transformers torchmetrics[detection] scipy
pip install matplotlib tensorboardX pillow opencv-python tqdm
```

#### **Bước 4: Cài đặt từ requirements.txt**
```bash
pip install -r requirements.txt
```

## 🧪 Kiểm tra cài đặt

### **Test cơ bản**
```bash
python scripts/test_imports.py
```

### **Test model creation**
```bash
python -c "
import sys
sys.path.append('src')
from models.open_vocab import build_open_vocab_detector
config = {'image_encoder': {'model_name': 'swin_tiny_patch4_window7_224', 'pretrained': False, 'out_dim': 256}, 'text_encoder': {'model_name': 'openai/clip-vit-base-patch32'}, 'fusion': {'dim': 256, 'num_layers': 1, 'num_heads': 4, 'text_dim': 512}, 'box_head': {'input_dim': 256, 'hidden_dim': 128, 'num_queries': 20}}
model = build_open_vocab_detector(config)
print('✅ Model created successfully!')
"
```

## 🐛 Xử lý lỗi thường gặp

### **1. Lỗi SDPA không tìm thấy**
```
ImportError: cannot import name 'SDPA' from 'torch.nn.attention'
```
**Giải pháp**: Đã được sửa tự động - sử dụng `MultiheadAttention` thay thế

### **2. Lỗi torchmetrics không tìm thấy**
```
ImportError: No module named 'torchmetrics'
```
**Giải pháp**: 
```bash
pip install torchmetrics[detection]
```

### **3. Lỗi torchvision.ops.MLP không tìm thấy**
```
ImportError: cannot import name 'MLP' from 'torchvision.ops'
```
**Giải pháp**: Đã được sửa tự động - sử dụng fallback implementation

### **4. Lỗi CUDA không tìm thấy**
```
RuntimeError: CUDA out of memory
```
**Giải pháp**: 
- Giảm batch size trong config
- Sử dụng CPU: `accelerator='cpu'`
- Cài đặt PyTorch CPU version

### **5. Lỗi transformers không tìm thấy**
```
ImportError: No module named 'transformers'
```
**Giải pháp**:
```bash
pip install transformers
```

## 📊 Kiểm tra GPU

### **Kiểm tra CUDA**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### **Kiểm tra memory**
```python
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## 🚀 Chạy project

### **1. Training**
```bash
# Training tối ưu
python scripts/train_optimized.py

# Training thông thường
python scripts/train.py
```

### **2. Testing**
```bash
# Test model
python run.py test

# Benchmark
python scripts/benchmark.py
```

### **3. Evaluation**
```bash
python run.py eval
```

## 📁 Cấu trúc project sau cài đặt

```
open_vocab_detection/
├── src/
│   ├── models/          # Model definitions
│   ├── training/        # Training modules
│   └── data/           # Data modules
├── scripts/
│   ├── train_optimized.py    # Optimized training
│   ├── benchmark.py          # Performance benchmark
│   └── install_dependencies.py  # Auto installer
├── requirements.txt     # Dependencies
├── INSTALLATION_GUIDE.md    # This file
└── OPTIMIZATION_GUIDE.md    # Optimization guide
```

## 🔧 Troubleshooting

### **Nếu gặp lỗi import**
1. Kiểm tra Python version: `python --version`
2. Kiểm tra pip version: `pip --version`
3. Cập nhật pip: `pip install --upgrade pip`
4. Cài đặt lại dependencies: `pip install -r requirements.txt`

### **Nếu gặp lỗi CUDA**
1. Kiểm tra CUDA version: `nvidia-smi`
2. Cài đặt PyTorch phù hợp với CUDA version
3. Sử dụng CPU nếu không có GPU

### **Nếu gặp lỗi memory**
1. Giảm batch size trong config
2. Sử dụng gradient checkpointing
3. Sử dụng mixed precision training

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra log lỗi chi tiết
2. Chạy `python scripts/test_imports.py`
3. Kiểm tra requirements.txt
4. Cài đặt lại dependencies

---

**🎉 Chúc mừng! Project đã sẵn sàng để sử dụng!**
