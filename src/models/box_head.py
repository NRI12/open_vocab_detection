import torch
import torch.nn as nn

# Fallback cho MLP nếu torchvision.ops.MLP không có sẵn
try:
    from torchvision.ops import MLP
except ImportError:
    # Fallback implementation
    class MLP(nn.Module):
        def __init__(self, in_channels, hidden_channels, dropout=0.0, activation_layer=nn.ReLU):
            super().__init__()
            layers = []
            prev_channels = in_channels
            for hidden_channels in hidden_channels:
                layers.extend([
                    nn.Linear(prev_channels, hidden_channels),
                    activation_layer(),
                    nn.Dropout(dropout)
                ])
                prev_channels = hidden_channels
            self.layers = nn.Sequential(*layers[:-1])  # Remove last dropout
        
        def forward(self, x):
            return self.layers(x)

class BoxHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_queries=100):
        super().__init__()

        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries,input_dim))

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # Sử dụng MLP từ torchvision.ops - tối ưu hơn
        self.bbox_head = MLP(
            input_dim, 
            [hidden_dim, hidden_dim, 4], 
            dropout=0.1,
            activation_layer=nn.ReLU
        )

        self.cls_head = MLP(
            input_dim, 
            [hidden_dim, 1], 
            dropout=0.1,
            activation_layer=nn.ReLU
        )
    def forward(self, features):
        B, N, D = features.shape
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        decoder_out = self.decoder(queries, features)
        
        bbox_pred = self.bbox_head(decoder_out).sigmoid()
        cls_pred = self.cls_head(decoder_out)
        
        return {
            'bbox_pred': bbox_pred,
            'cls_pred': cls_pred, 
            'decoder_features': decoder_out
        }

def build_box_head(config):
    return BoxHead(
        input_dim=config.get('input_dim', 768),
        hidden_dim=config.get('hidden_dim', 256),
        num_queries=config.get('num_queries', 100)
    )