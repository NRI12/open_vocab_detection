import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import Flickr30kDataModule
from training.lightning_module import OpenVocabLightningModule

def main():
    # Config
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_base_patch4_window7_224',
                'pretrained': True,
                'out_dim': 512
            },
            'text_encoder': {},
            'fusion': {
                'dim': 512,
                'num_layers': 2,
                'num_heads': 8
            },
            'box_head': {
                'input_dim': 512,
                'hidden_dim': 256,
                'num_queries': 100
            }
        },
        'lr': 1e-4,
        'weight_decay': 1e-4
    }
    
    # Data
    dm = Flickr30kDataModule(
        data_dir='./data',
        batch_size=8,
        num_workers=4,
        img_size=(224, 224)
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
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        val_check_interval=0.5,
        log_every_n_steps=50
    )
    
    # Train
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()