import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
from models.open_vocab import build_open_vocab_detector

def benchmark_model(config, device='cuda', num_iterations=10):
    """Benchmark model performance"""
    model = build_open_vocab_detector(config).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy inputs
    batch_size = 8
    img_size = config['model']['image_encoder'].get('out_dim', 768)
    text_encoder_dim = 512  # CLIP base
    
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_texts = ["a person", "a dog", "a car", "a building"] * (batch_size // 4)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_images, dummy_texts)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_images, dummy_texts)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    throughput = batch_size / avg_time
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'memory_usage': torch.cuda.max_memory_allocated() / 1024**3  # GB
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}")
    print("=" * 60)
    
    # Original config (heavy)
    original_config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_base_patch4_window7_224',
                'pretrained': True,
                'out_dim': 512
            },
            'text_encoder': {
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'fusion': {
                'dim': 512,
                'num_layers': 2,
                'num_heads': 8
            },
            'box_head': {
                'input_dim': 512,
                'hidden_dim': 256,
                'num_queries': 100
            }
        }
    }
    
    # Optimized config (light)
    optimized_config = {
        'model': {
            'image_encoder': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'pretrained': True,
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
    
    print("Original Model (Heavy):")
    print("-" * 30)
    try:
        original_stats = benchmark_model(original_config, device)
        print(f"Total parameters: {original_stats['total_params']:,}")
        print(f"Trainable parameters: {original_stats['trainable_params']:,}")
        print(f"Average inference time: {original_stats['avg_inference_time']:.3f}s")
        print(f"Throughput: {original_stats['throughput']:.1f} samples/s")
        print(f"Memory usage: {original_stats['memory_usage']:.2f} GB")
    except Exception as e:
        print(f"Error benchmarking original model: {e}")
    
    print("\nOptimized Model (Light):")
    print("-" * 30)
    try:
        torch.cuda.empty_cache()
        optimized_stats = benchmark_model(optimized_config, device)
        print(f"Total parameters: {optimized_stats['total_params']:,}")
        print(f"Trainable parameters: {optimized_stats['trainable_params']:,}")
        print(f"Average inference time: {optimized_stats['avg_inference_time']:.3f}s")
        print(f"Throughput: {optimized_stats['throughput']:.1f} samples/s")
        print(f"Memory usage: {optimized_stats['memory_usage']:.2f} GB")
        
        # Calculate improvements
        if 'original_stats' in locals():
            param_reduction = (1 - optimized_stats['total_params'] / original_stats['total_params']) * 100
            speed_improvement = optimized_stats['throughput'] / original_stats['throughput']
            memory_reduction = (1 - optimized_stats['memory_usage'] / original_stats['memory_usage']) * 100
            
            print(f"\nImprovements:")
            print(f"Parameter reduction: {param_reduction:.1f}%")
            print(f"Speed improvement: {speed_improvement:.1f}x")
            print(f"Memory reduction: {memory_reduction:.1f}%")
            
    except Exception as e:
        print(f"Error benchmarking optimized model: {e}")

if __name__ == '__main__':
    main()
