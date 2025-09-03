import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_encoder import build_image_encoder
from models.text_encoder import build_text_encoder
from models.fusion import build_fusion_module
from models.box_head import build_box_head

class OpenVocabDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.image_encoder = build_image_encoder(config.get('image_encoder', {}))
        self.text_encoder = build_text_encoder(config.get('text_encoder', {}))
        self.fusion = build_fusion_module(config.get('fusion', {}))
        self.box_head = build_box_head(config.get('box_head', {}))
        
        img_dim = config.get('fusion', {}).get('dim', 768)
        text_dim = self.text_encoder.embed_dim
        self.img_proj = nn.Linear(img_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        
    def forward(self, images, texts):
        # Encode
        img_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Fusion
        fused_features = self.fusion(img_features, text_features)
        
        # Detection
        detection_out = self.box_head(fused_features)
        
        # Open-vocab classification
        region_features = detection_out['decoder_features']
        img_proj = self.img_proj(region_features)
        text_proj = self.text_proj(text_features.unsqueeze(1))
        
        # Cosine similarity
        img_proj = F.normalize(img_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        similarity = torch.matmul(img_proj, text_proj.transpose(-2, -1)).squeeze(-1)
        
        return {
            'bbox_pred': detection_out['bbox_pred'],
            'cls_pred': detection_out['cls_pred'],
            'similarity': similarity,
            'region_features': region_features,
            'text_features': text_features
        }

def build_open_vocab_detector(config):
    return OpenVocabDetector(config)