import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import Flickr30kDataModule
from training.lightning_module import OpenVocabLightningModule

def main():
    # Optimized config for better performance and faster training
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',  # Much lighter model
                'pretrained': True,
                'out_dim': 256  # Further reduced dimension
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,  # Single layer
                'num_heads': 4    # Fewer heads
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 25  # Much fewer queries
            }
        },
        'lr': 3e-4,  # Higher learning rate for faster convergence
        'weight_decay': 1e-4,
        'loss': {
            'lambda_cls': 0.5,     # Reduced classification loss
            'lambda_bbox': 5.0,
            'lambda_giou': 2.0,
            'lambda_sim': 0.2,     # Much reduced similarity loss
            'temperature': 0.1
        }
    }
    
    # Data with optimized settings
    dm = Flickr30kDataModule(
        data_dir='./data',
        batch_size=24,  # Larger batch size for better GPU utilization
        num_workers=8,  # More workers
        img_size=(256, 256)  # Smaller images for faster training
    )
    
    # Model
    model = OpenVocabLightningModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mAP',
        mode='max',
        save_top_k=2,  # Save fewer checkpoints
        filename='epoch{epoch:02d}-map{val/mAP:.3f}',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=8,  # Reduced patience
        mode='min',
        min_delta=0.001
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='open_vocab_optimized')
    
    # Trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=20,  # Fewer epochs
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=10,  # More frequent logging
        precision=16,  # Mixed precision
        gradient_clip_val=0.5,  # Gradient clipping
        accumulate_grad_batches=1,  # No accumulation needed with larger batch
        deterministic=False,  # Allow non-deterministic for speed
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=False,  # Set to True for quick testing
        limit_train_batches=1.0,  # Use full dataset
        limit_val_batches=0.5,  # Use half of validation for speed
    )
    
    # Train
    print("Starting optimized training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Batch size: {dm.batch_size}")
    print(f"Image size: {dm.img_size}")
    print(f"Max epochs: {trainer.max_epochs}")
    
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()
