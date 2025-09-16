import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.open_vocab import build_open_vocab_detector

def test_model_creation():
    """Test if the optimized model can be created successfully"""
    print("Testing optimized model creation...")
    
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': False,  # Don't download for test
                'out_dim': 256
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,
                'num_heads': 4
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 25
            }
        }
    }
    
    try:
        model = build_open_vocab_detector(config)
        print("✅ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': False,
                'out_dim': 256
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,
                'num_heads': 4
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 25
            }
        }
    }
    
    try:
        model = build_open_vocab_detector(config)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 256, 256)
        dummy_texts = ["a person", "a dog"]
        
        with torch.no_grad():
            outputs = model(dummy_images, dummy_texts)
        
        print("✅ Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
        print(f"Bbox pred shape: {outputs['bbox_pred'].shape}")
        print(f"Similarity shape: {outputs['similarity'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return False

def test_memory_usage():
    """Test memory usage"""
    print("\nTesting memory usage...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True
    
    config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': False,
                'out_dim': 256
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 256,
                'num_layers': 1,
                'num_heads': 4
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 25
            }
        }
    }
    
    try:
        model = build_open_vocab_detector(config).cuda()
        model.eval()
        
        # Test with different batch sizes
        for batch_size in [4, 8, 16]:
            torch.cuda.empty_cache()
            dummy_images = torch.randn(batch_size, 3, 256, 256).cuda()
            dummy_texts = ["a person"] * batch_size
            
            with torch.no_grad():
                _ = model(dummy_images, dummy_texts)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Batch size {batch_size}: {memory_used:.2f} GB")
            
            if memory_used > 6:  # More than 6GB is too much
                print(f"⚠️  High memory usage with batch size {batch_size}")
                break
        
        print("✅ Memory test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in memory test: {e}")
        return False

def main():
    print("🧪 Quick Test for Optimized Open Vocabulary Detection")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == '__main__':
    main()
