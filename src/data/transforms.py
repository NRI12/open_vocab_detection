import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from PIL import Image
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes, orig_size):
        for t in self.transforms:
            image, boxes = t(image, boxes, orig_size)
        return image, boxes

class Resize:
    def __init__(self, size=(800, 800)):
        self.size = size
    
    def __call__(self, image, boxes, orig_size):
        w, h = orig_size
        target_w, target_h = self.size
        
        # Preserve aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize với aspect ratio preserved
        image = F.resize(image, (new_h, new_w))
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        image = F.pad(image, (pad_w, pad_h, target_w - new_w - pad_w, target_h - new_h - pad_h))
        
        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w  # x coordinates
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h  # y coordinates
        
        return image, boxes

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, orig_size):
        if random.random() < self.p:
            image = F.hflip(image)
            
            # Flip boxes
            if len(boxes) > 0:
                w = image.size[0]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # flip x coordinates
        
        return image, boxes

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, boxes, orig_size):
        image = self.jitter(image)
        return image, boxes

class ToTensor:
    def __call__(self, image, boxes, orig_size):
        image = F.to_tensor(image)
        return image, boxes

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, boxes, orig_size):
        image = F.normalize(image, self.mean, self.std)
        return image, boxes

def get_transforms(split='train', img_size=(800, 800)):
    if split == 'train':
        return Compose([
            Resize(img_size),
            RandomHorizontalFlip(0.5),
            ColorJitter(0.2, 0.2, 0.2, 0.1),
            ToTensor(),
            Normalize()
        ])
    else:
        return Compose([
            Resize(img_size),
            ToTensor(),
            Normalize()
        ])