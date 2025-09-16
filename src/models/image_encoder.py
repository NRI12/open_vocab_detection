import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP

class ImageEncoder(nn.Module):    
    def __init__(self, 
                 model_name: str = 'swin_base_patch4_window7_224',
                 pretrained: bool = True,
                 out_dim: int = 768):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            if len(dummy_output.shape) == 4:  # Conv output (B, C, H, W)
                self.feature_dim = dummy_output.shape[1]
                self.is_conv = True
            else:  # Transformer output (B, N, D)
                self.feature_dim = dummy_output.shape[-1]
                self.is_conv = False
        
        # Sử dụng MLP từ torchvision.ops - tối ưu hơn
        if self.feature_dim != out_dim:
            self.proj = MLP(self.feature_dim, [out_dim], dropout=0.1)
        else:
            self.proj = nn.Identity()
        
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, H*W, D) or (B, N, D)
        """
        features = self.backbone(x)
        
        if self.is_conv:
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        features = self.proj(features)
        return features

def swin_base_encoder(pretrained=True):
    return ImageEncoder('swin_base_patch4_window7_224', pretrained, 768)

def vit_base_encoder(pretrained=True):
    return ImageEncoder('vit_base_patch16_224', pretrained, 768)

def build_image_encoder(config):
    model_name = config.get('model_name', 'swin_base_patch4_window7_224')
    pretrained = config.get('pretrained', True)
    out_dim = config.get('out_dim', 768)
    
    return ImageEncoder(model_name, pretrained, out_dim)