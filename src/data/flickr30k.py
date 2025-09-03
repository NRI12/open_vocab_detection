import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class Flickr30kDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'flickr30k_images')
        self.ann_dir = os.path.join(data_dir, 'Annotations')
        self.sent_dir = os.path.join(data_dir, 'Sentences')
        self.transform = transform
        
        with open(os.path.join(data_dir, f'{split}.txt')) as f:
            self.img_ids = [line.strip().replace('.jpg', '') for line in f]
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (w, h)
        
        sent_path = os.path.join(self.sent_dir, f'{img_id}.txt')
        with open(sent_path) as f:
            sentences = f.readlines()
        
        entity_pattern = r'\[/EN#(\d+)/(\w+)\s+([^\]]+)\]'
        entity_map = {}
        all_phrases = []
        
        for sentence in sentences:
            entities = re.findall(entity_pattern, sentence.strip())
            for eid, cat, text in entities:
                entity_map[eid] = text.strip()
                all_phrases.append(text.strip())
        
        combined_text = " | ".join(set(all_phrases))
        
        ann_path = os.path.join(self.ann_dir, f'{img_id}.xml')
        tree = ET.parse(ann_path)
        
        boxes = []
        labels = []
        
        for obj in tree.findall('.//object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            if bbox is not None and name in entity_map:
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(entity_map[name])
        
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        if self.transform:
            image, boxes = self.transform(image, boxes, orig_size)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'text': combined_text,
            'img_id': img_id,
            'orig_size': orig_size
        }