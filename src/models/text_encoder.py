import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder(nn.Module):    
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.text_model.config.hidden_size  # 512
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings
        Returns:
            text_features: (B, D)
        """
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(next(self.parameters()).device)
        
        # Encode
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            text_features = outputs.pooler_output  # (B, D)
        
        return text_features


def build_text_encoder(config=None):
    return TextEncoder('openai/clip-vit-base-patch32')

# (3,512) # shape
# encoder = TextEncoder()
# features = encoder(["A woman", "pink dress", "dog running"])
# print(features.shape)
