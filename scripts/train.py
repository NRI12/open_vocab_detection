import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import Flickr30kDataModule
from training.lightning_module import OpenVocabLightningModule

def main():
    # Config - Optimized for better performance
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',  # Lighter model
                'pretrained': True,
                'out_dim': 384  # Reduced dimension
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'  # Smaller CLIP
            },
            'fusion': {
                'dim': 384,
                'num_layers': 1,  # Reduced layers
                'num_heads': 6
            },
            'box_head': {
                'input_dim': 384,
                'hidden_dim': 192,
                'num_queries': 50  # Reduced queries
            }
        },
        'lr': 2e-4,  # Higher learning rate
        'weight_decay': 1e-4,
        'loss': {
            'lambda_cls': 1.0,
            'lambda_bbox': 5.0,
            'lambda_giou': 2.0,
            'lambda_sim': 0.5,  # Reduced similarity loss weight
            'temperature': 0.1
        }
    }
    
    # Data - Optimized settings
    dm = Flickr30kDataModule(
        data_dir='./data',
        batch_size=16,  # Increased batch size
        num_workers=6,  # More workers
        img_size=(384, 384)  # Smaller image size for faster training
    )
    
    # Model
    model = OpenVocabLightningModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/mAP',
        mode='max',
        save_top_k=3,
        filename='epoch{epoch:02d}-map{val/mAP:.3f}'
    )
    
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=10,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='open_vocab_detection')
    
    # Trainer - Optimized for performance
    trainer = pl.Trainer(
        max_epochs=30,  # Reduced epochs
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=25,  # More frequent logging
        precision=16,  # Mixed precision for memory efficiency
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=2,  # Gradient accumulation
        deterministic=False  # Allow non-deterministic for speed
    )
    
    # Train
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()