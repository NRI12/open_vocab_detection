
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .flickr30k import Flickr30kDataset
from .transforms import get_transforms

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    texts = [item['text'] for item in batch]
    img_ids = [item['img_id'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]

    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'texts': texts,
        'img_ids': img_ids,
        'orig_sizes': orig_sizes
    }

class Flickr30kDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir='./data',
                 batch_size=16,
                 num_workers=4,
                 img_size=(384, 384)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = min(num_workers, 8)  # Cap workers
        self.img_size = img_size
        self.persistent_workers = True  # Keep workers alive
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Flickr30kDataset(
                self.data_dir, 'train', 
                transform=get_transforms('train', self.img_size)
            )
            self.val_dataset = Flickr30kDataset(
                self.data_dir, 'val',
                transform=get_transforms('val', self.img_size)
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = Flickr30kDataset(
                self.data_dir, 'test',
                transform=get_transforms('test', self.img_size)
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=self.persistent_workers,
            pin_memory=True  # Faster GPU transfer
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )