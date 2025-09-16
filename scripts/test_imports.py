import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test tất cả imports để đảm bảo không có lỗi"""
    print("🔍 TESTING IMPORTS")
    print("=" * 40)
    
    try:
        print("1. Testing basic imports...")
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        print("   ✅ PyTorch imports OK")
        
        print("2. Testing torchvision imports...")
        from torchvision.ops import generalized_box_iou, box_iou
        print("   ✅ TorchVision ops OK")
        
        print("3. Testing scipy imports...")
        from scipy.optimize import linear_sum_assignment
        print("   ✅ SciPy imports OK")
        
        print("4. Testing timm imports...")
        import timm
        print("   ✅ TIMM imports OK")
        
        print("5. Testing transformers imports...")
        from transformers import CLIPTextModel, CLIPTokenizer
        print("   ✅ Transformers imports OK")
        
        print("6. Testing pytorch_lightning imports...")
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        print("   ✅ PyTorch Lightning imports OK")
        
        print("7. Testing torchmetrics imports...")
        try:
            from torchmetrics.detection import MeanAveragePrecision
            from torchmetrics.functional import pairwise_cosine_similarity
            print("   ✅ TorchMetrics imports OK")
        except ImportError:
            print("   ⚠️  TorchMetrics not available, using fallback")
        
        print("8. Testing custom modules...")
        from models.image_encoder import build_image_encoder
        from models.text_encoder import build_text_encoder
        from models.fusion import build_fusion_module
        from models.box_head import build_box_head
        from models.open_vocab import build_open_vocab_detector
        print("   ✅ Custom model imports OK")
        
        print("9. Testing training modules...")
        from training.losses import DetectionLoss
        from training.lightning_module import OpenVocabLightningModule
        print("   ✅ Training module imports OK")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Project is ready to run!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test tạo model để đảm bảo không có lỗi"""
    print("\n🔍 TESTING MODEL CREATION")
    print("=" * 40)
    
    try:
        from models.open_vocab import build_open_vocab_detector
        
        config = {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': False,  # Không load pretrained để test nhanh
                'out_dim': 256
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,
                'num_heads': 4,
                'text_dim': 512
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 20
            }
        }
        
        print("Creating model...")
        model = build_open_vocab_detector(config)
        print("   ✅ Model created successfully!")
        
        print("Testing forward pass...")
        import torch
        images = torch.randn(2, 3, 224, 224)
        texts = ["a person walking", "a car on the road"]
        
        with torch.no_grad():
            outputs = model(images, texts)
        
        print("   ✅ Forward pass successful!")
        print(f"   Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ MODEL CREATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 TESTING PROJECT SETUP")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test model creation
        model_ok = test_model_creation()
        
        if model_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Project is ready for training!")
        else:
            print("\n❌ Model creation failed!")
    else:
        print("\n❌ Import tests failed!")
