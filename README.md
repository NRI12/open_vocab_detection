# Open Vocabulary Object Detection

## Mô tả bài toán

Phát hiện và phân đoạn các đối tượng trong ảnh dựa trên mô tả bằng ngôn ngữ tự nhiên, không giới hạn bởi tập nhãn cố định.

## Kiến trúc hệ thống

**Input**: Ảnh + Text prompt

**Pipeline**:
1. **Image Encoder**: Trích xuất đặc trưng ảnh (ViT/Swin Transformer)
2. **Text Encoder**: Mã hóa văn bản (CLIP)
3. **Cross-Attention Fusion**: Kết hợp thông tin ảnh-văn bản
4. **Box Head**: Dự đoán bounding boxes
5. **Mask Head**: Tạo mask segmentation
6. **Open-Vocab Classification**: So sánh embedding để gán nhãn

## Dataset

**Flickr30k Entities**: 30,000 ảnh với caption và bounding box annotation

## Cài đặt

```bash
pip install -r requirements.txt
python scripts/download_data.py --data_dir ./data
```

## Sử dụng

**Training**:
```bash
python scripts/train.py
```

**Evaluation**:
```bash
python scripts/evaluate.py
```

**Demo**:
```bash
python src/demo/app.py
```

## Cấu trúc thư mục

```
├── configs/          # Cấu hình training và model
├── scripts/          # Script training, evaluation, download data  
├── src/
│   ├── data/         # Dataloader và transforms
│   ├── models/       # Định nghĩa model
│   ├── training/     # Lightning module, loss, metrics
│   ├── utils/        # Utilities
│   └── demo/         # Demo application
```

## Ưu điểm

- Phát hiện đối tượng mới không cần training thêm
- Tích hợp hiểu biết ngôn ngữ và thị giác
- Flexible với mô tả tự nhiên