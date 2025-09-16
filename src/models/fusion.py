import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, dim=768, num_layers=2, num_heads=8, dropout=0.1, text_dim=512):
        super().__init__()
        
        # Projection layers to match dimensions
        self.img_proj = nn.Identity()  # Image features already have correct dim
        self.text_proj = nn.Linear(text_dim, dim) if text_dim != dim else nn.Identity()
        
        # Sử dụng MultiheadAttention - tương thích với mọi phiên bản PyTorch
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
    
    def forward(self, img_features, text_features):
        # Project features to same dimension
        x = self.img_proj(img_features)
        text_seq = self.text_proj(text_features).unsqueeze(1)
        
        for cross_attn, norm, ffn, ffn_norm in zip(
            self.cross_attn_layers, self.norms, self.ffns, self.ffn_norms
        ):
            # Sử dụng MultiheadAttention trực tiếp
            attn_out, _ = cross_attn(query=x, key=text_seq, value=text_seq)
            x = norm(x + attn_out)
            x = ffn_norm(x + ffn(x))
        
        return x

def build_fusion_module(config):
    return FusionModule(
        dim=config.get('dim', 768),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        text_dim=config.get('text_dim', 512)
    )