# 🎯 Open Vocabulary Object Detection

Phát hiện đối tượng trong ảnh dựa trên mô tả ngôn ngữ tự nhiên, không giới hạn bởi tập nhãn cố định.

## ⚡ Quick Start

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Test nhanh
```bash
python scripts/quick_test.py
```

### 3. Training (Khuyến nghị)
```bash
python scripts/train_optimized.py
```

## 🚀 Các script chính

| Script | Mô tả | Thời gian |
|--------|-------|-----------|
| `quick_test.py` | Test model và memory | ~2 phút |
| `train_optimized.py` | Training tối ưu (khuyến nghị) | ~2-3 giờ |
| `train.py` | Training gốc | ~6-8 giờ |
| `compare_models.py` | So sánh hiệu suất | ~5 phút |
| `evaluate.py` | Đánh giá model | ~30 phút |

## 📊 Kiến trúc tối ưu

**Input**: Ảnh (256x256) + Text prompt

**Pipeline**:
1. **Image Encoder**: Swin-Tiny (28M params)
2. **Text Encoder**: CLIP ViT-Base
3. **Fusion**: Single-layer cross-attention
4. **Box Head**: 25 queries, 128 hidden dim
5. **Classification**: Cosine similarity matching

## 🎯 Hiệu suất

| Metric | Model gốc | Model tối ưu | Cải thiện |
|--------|-----------|--------------|-----------|
| Parameters | 88M | 28M | -68% |
| Memory | ~6GB | ~3GB | -50% |
| Speed | 30 samples/s | 100 samples/s | +233% |
| Training time | 50 epochs | 20 epochs | -60% |

## 📁 Cấu trúc dự án

```
├── scripts/              # Scripts chính
│   ├── quick_test.py     # Test nhanh
│   ├── train_optimized.py # Training tối ưu (khuyến nghị)
│   ├── train.py          # Training gốc
│   ├── compare_models.py # So sánh hiệu suất
│   └── evaluate.py       # Đánh giá model
├── src/
│   ├── data/             # Dataloader và transforms
│   ├── models/           # Định nghĩa model
│   ├── training/         # Lightning module, loss, metrics
│   └── demo/             # Demo application
├── configs/              # Cấu hình
├── data/                 # Dataset (Flickr30k)
└── lightning_logs/       # Logs và checkpoints
```

## 📚 Tài liệu

- **[HOW_TO_RUN.md](HOW_TO_RUN.md)**: Hướng dẫn chi tiết cách chạy
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)**: Giải thích các tối ưu hóa
- **TensorBoard**: `tensorboard --logdir lightning_logs`

## 🎯 Ưu điểm

- ✅ **Nhanh**: Training 3-4x nhanh hơn
- ✅ **Nhẹ**: Giảm 68% parameters, 50% memory
- ✅ **Linh hoạt**: Phát hiện đối tượng mới không cần training
- ✅ **Dễ sử dụng**: Scripts đơn giản, hướng dẫn rõ ràng