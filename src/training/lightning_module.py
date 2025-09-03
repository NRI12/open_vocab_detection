import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.open_vocab import build_open_vocab_detector
from training.losses import DetectionLoss
from training.metrics import DetectionMetrics

class OpenVocabLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = build_open_vocab_detector(config['model'])
        self.criterion = DetectionLoss(config.get('loss', {}))
        self.metrics = DetectionMetrics()
        
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
    def forward(self, images, texts):
        return self.model(images, texts)
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        targets = {
            'boxes': batch['boxes'],
            'labels': batch['labels']
        }
        
        outputs = self(images, texts)
        loss_dict = self.criterion(outputs, targets, texts)
        
        total_loss = sum(loss_dict.values())
        
        self.log_dict({f'train/{k}': v for k, v in loss_dict.items()})
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        targets = {
            'boxes': batch['boxes'],
            'labels': batch['labels']
        }
        
        outputs = self(images, texts)
        loss_dict = self.criterion(outputs, targets, texts)
        
        total_loss = sum(loss_dict.values())
        
        self.log_dict({f'val/{k}': v for k, v in loss_dict.items()})
        self.log('val/total_loss', total_loss, prog_bar=True)
        
        # Metrics
        metrics = self.metrics(outputs, targets)
        self.log_dict({f'val/{k}': v for k, v in metrics.items()})
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }