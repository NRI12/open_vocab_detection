# 🎯 Open Vocabulary Object Detection

Phát hiện đối tượng trong ảnh dựa trên mô tả ngôn ngữ tự nhiên, không giới hạn bởi tập nhãn cố định.

## ⚡ Quick Start

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Chạy
```bash
python run.py train    # Training
python run.py test     # Test model
python run.py eval     # Đánh giá
```

## 🚀 Scripts

| Lệnh | Mô tả | Thời gian |
|------|-------|-----------|
| `python run.py test` | Test model | ~1 phút |
| `python run.py train` | Training | ~2-3 giờ |
| `python run.py eval` | Đánh giá | ~30 phút |

## 📊 Kiến trúc tối ưu

**Input**: Ảnh (224x224) + Text prompt

**Pipeline**:
1. **Image Encoder**: Swin-Tiny (28M params)
2. **Text Encoder**: CLIP ViT-Base
3. **Fusion**: Single-layer cross-attention
4. **Box Head**: 25 queries, 128 hidden dim
5. **Classification**: Cosine similarity matching

## 🎯 Hiệu suất

| Metric | Model tối ưu |
|--------|--------------|
| Parameters | 28M |
| Memory | ~3GB VRAM |
| Speed | 100 samples/s |
| Training time | 20 epochs |

## 📁 Cấu trúc dự án

```
├── scripts/              # Scripts chính
│   ├── train.py          # Training
│   └── evaluate.py       # Đánh giá
├── src/
│   ├── data/             # Dataloader và transforms
│   ├── models/           # Định nghĩa model
│   ├── training/         # Lightning module, loss, metrics
│   └── demo/             # Demo application
├── configs/              # Cấu hình
├── data/                 # Dataset (Flickr30k)
└── lightning_logs/       # Logs và checkpoints
```

## 🎯 Ưu điểm

- ✅ **Nhanh**: Training 3-4x nhanh hơn
- ✅ **Nhẹ**: 28M parameters, 3GB VRAM
- ✅ **Linh hoạt**: Phát hiện đối tượng mới không cần training
- ✅ **Dễ sử dụng**: Chỉ cần `python run.py train`

## 📊 Monitoring

```bash
tensorboard --logdir lightning_logs
```

## 🔧 Yêu cầu hệ thống

- **Python**: 3.8+
- **CUDA**: 11.0+ (khuyến nghị)
- **RAM**: Tối thiểu 8GB
- **GPU**: Tối thiểu 6GB VRAM