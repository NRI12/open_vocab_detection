import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import Flickr30kDataModule
from training.lightning_module import OpenVocabLightningModule

def main():
    # Config tối ưu cho hiệu suất cao nhất
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',  # Model nhẹ nhất
                'pretrained': True,
                'out_dim': 256  # Dimension nhỏ nhất
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,  # Chỉ 1 layer
                'num_heads': 4,   # Ít heads hơn
                'text_dim': 512
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 20  # Ít queries hơn
            }
        },
        'lr': 5e-4,  # Learning rate cao hơn
        'weight_decay': 1e-4,
        'use_amp': True,  # Bật mixed precision
        'loss': {
            'lambda_cls': 0.3,     # Giảm classification loss
            'lambda_bbox': 5.0,
            'lambda_giou': 2.0,
            'lambda_sim': 0.1,     # Giảm similarity loss
            'temperature': 0.1,
            'region_dim': 256,
            'text_dim': 512
        }
    }
    
    # Data module với settings tối ưu
    dm = Flickr30kDataModule(
        data_dir='./data',
        batch_size=32,  # Batch size lớn hơn
        num_workers=8,  # Nhiều workers
        img_size=(224, 224),
        pin_memory=True,  # Pin memory cho GPU
        persistent_workers=True  # Giữ workers
    )
    
    # Model
    model = OpenVocabLightningModule(config)
    
    # Callbacks tối ưu
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mAP',
        mode='max',
        save_top_k=1,  # Chỉ lưu 1 checkpoint tốt nhất
        filename='best-{epoch:02d}-{val/mAP:.3f}',
        save_last=True,
        every_n_epochs=5  # Lưu mỗi 5 epochs
    )
    
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=5,  # Patience thấp hơn
        mode='min',
        min_delta=0.001,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='open_vocab_ultra_optimized')
    
    # Trainer với tất cả tối ưu hóa
    trainer = pl.Trainer(
        max_epochs=15,  # Ít epochs hơn
        accelerator='gpu',
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else 'auto',
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=5,  # Log thường xuyên hơn
        precision=16,  # Mixed precision
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,
        deterministic=False,  # Cho phép non-deterministic để tăng tốc
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=0.3,  # Chỉ dùng 30% validation
        sync_batchnorm=True,  # Sync batch norm
        replace_sampler_ddp=False,  # Tối ưu DDP
        detect_anomaly=False,  # Tắt anomaly detection để tăng tốc
    )
    
    # Train
    print("🚀 Starting ULTRA OPTIMIZED training...")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"📦 Batch size: {dm.batch_size}")
    print(f"🖼️  Image size: {dm.img_size}")
    print(f"⏱️  Max epochs: {trainer.max_epochs}")
    print(f"🎯 Mixed precision: {config['use_amp']}")
    print(f"💾 Memory pinning: True")
    print(f"🔄 Persistent workers: True")
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    import torch
    main()