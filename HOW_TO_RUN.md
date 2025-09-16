# 🚀 Hướng dẫn chạy Open Vocabulary Detection

## 📋 Yêu cầu hệ thống

- **Python**: 3.8+
- **CUDA**: 11.0+ (khuyến nghị)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **GPU**: Tối thiểu 6GB VRAM

## ⚡ Cài đặt nhanh

### 1. Clone và cài đặt dependencies
```bash
git clone <your-repo>
cd open_vocab_detection
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
```bash
# Tạo thư mục dữ liệu
mkdir -p data/flickr30k_images
mkdir -p data/Annotations
mkdir -p data/Sentences

# Copy dữ liệu Flickr30k vào các thư mục tương ứng
# - flickr30k_images/: Chứa các file ảnh .jpg
# - Annotations/: Chứa các file annotation .xml  
# - Sentences/: Chứa các file câu mô tả .txt
```

## 🎯 Các cách chạy

### 1. **Test nhanh (Khuyến nghị đầu tiên)**
```bash
python scripts/quick_test.py
```
- Kiểm tra model có tạo được không
- Test forward pass
- Kiểm tra memory usage
- **Thời gian**: ~2 phút

### 2. **Training với model tối ưu (Khuyến nghị)**
```bash
python scripts/train_optimized.py
```
- Model nhẹ, training nhanh
- Batch size 24, image size 256x256
- Mixed precision, gradient clipping
- **Thời gian**: ~2-3 giờ (20 epochs)

### 3. **Training với model gốc (So sánh)**
```bash
python scripts/train.py
```
- Model nặng hơn, training chậm hơn
- Batch size 16, image size 384x384
- **Thời gian**: ~6-8 giờ (30 epochs)

### 4. **So sánh hiệu suất**
```bash
python scripts/compare_models.py
```
- So sánh model cũ vs mới
- Đo memory usage, speed, parameters
- **Thời gian**: ~5 phút

### 5. **Evaluation**
```bash
python scripts/evaluate.py
```
- Đánh giá model đã train
- Tính mAP, precision, recall
- **Thời gian**: ~30 phút

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir lightning_logs
```
Truy cập: http://localhost:6006

### Key metrics:
- `train/total_loss`: Loss training (nên giảm)
- `val/mAP`: Mean Average Precision (metric chính)
- `val/total_loss`: Loss validation (nên gần train loss)
- GPU memory usage

## 🔧 Tuning cho hệ thống yếu

### Nếu GPU < 6GB VRAM:
```bash
# Chỉnh sửa trong train_optimized.py
batch_size = 8          # Giảm từ 24
img_size = (224, 224)   # Giảm từ 256x256
accumulate_grad_batches = 2  # Tăng từ 1
```

### Nếu RAM < 8GB:
```bash
# Chỉnh sửa trong train_optimized.py
num_workers = 2         # Giảm từ 8
batch_size = 4          # Giảm từ 24
```

### Nếu muốn training nhanh hơn:
```bash
# Chỉnh sửa trong train_optimized.py
max_epochs = 10         # Giảm từ 20
limit_val_batches = 0.2 # Giảm từ 0.5
```

## 🐛 Troubleshooting

### Lỗi "CUDA out of memory":
```bash
# Giảm batch size
batch_size = 4
# Hoặc tăng gradient accumulation
accumulate_grad_batches = 4
```

### Lỗi "Module not found":
```bash
# Cài đặt lại dependencies
pip install -r requirements.txt
# Hoặc cài đặt thủ công
pip install torch torchvision pytorch-lightning transformers
```

### Training quá chậm:
```bash
# Kiểm tra GPU usage
nvidia-smi
# Giảm image size
img_size = (224, 224)
# Tăng batch size nếu có đủ memory
batch_size = 32
```

### Kết quả kém:
```bash
# Tăng learning rate
lr = 5e-4
# Tăng epochs
max_epochs = 30
# Tăng similarity loss weight
lambda_sim = 0.5
```

## 📁 Cấu trúc output

```
lightning_logs/
├── open_vocab_optimized/     # Logs từ train_optimized.py
│   ├── version_0/
│   │   ├── events.out.tfevents.*
│   │   └── hparams.yaml
│   └── checkpoints/          # Model checkpoints
│       ├── last.ckpt
│       └── epoch=XX-map=X.XXX.ckpt
└── open_vocab_detection/     # Logs từ train.py
```

## 🎉 Kết quả mong đợi

### Model tối ưu (train_optimized.py):
- **Parameters**: ~28M
- **Memory**: ~3GB VRAM
- **Speed**: ~100 samples/s
- **mAP**: 0.15-0.25 (tùy dataset)

### Model gốc (train.py):
- **Parameters**: ~88M  
- **Memory**: ~6GB VRAM
- **Speed**: ~30 samples/s
- **mAP**: 0.20-0.30 (tùy dataset)

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Chạy `quick_test.py` để kiểm tra
2. Kiểm tra logs trong `lightning_logs/`
3. Xem TensorBoard để monitor training
4. Thử giảm batch size nếu out of memory

**Chúc bạn training thành công! 🚀**
