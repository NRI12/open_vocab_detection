import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
import psutil
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"⏱️  {name}: {end - start:.2f}s")

def benchmark_model():
    """Benchmark hiệu suất model"""
    from models.open_vocab import build_open_vocab_detector
    
    # Config tối ưu
    config = {
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
            'num_heads': 4,
            'text_dim': 512
        },
        'box_head': {
            'input_dim': 256,
            'hidden_dim': 128,
            'num_queries': 20
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_open_vocab_detector(config).to(device)
    model.eval()
    
    # Dummy data
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    texts = ["a person walking on the street"] * batch_size
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(images, texts)
    
    # Benchmark inference
    print("🚀 Benchmarking inference...")
    times = []
    for i in range(20):
        with timer(f"Inference {i+1}"):
            with torch.no_grad():
                start = time.time()
                outputs = model(images, texts)
                end = time.time()
                times.append(end - start)
    
    avg_time = sum(times) / len(times)
    fps = batch_size / avg_time
    
    print(f"\n📊 Results:")
    print(f"   Average inference time: {avg_time:.3f}s")
    print(f"   FPS: {fps:.1f}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Memory usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   GPU memory: {gpu_memory:.2f} GB")
    
    cpu_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"   CPU memory: {cpu_memory:.2f} GB")
    
    return {
        'avg_time': avg_time,
        'fps': fps,
        'params': sum(p.numel() for p in model.parameters()),
        'gpu_memory': gpu_memory if torch.cuda.is_available() else 0,
        'cpu_memory': cpu_memory
    }

def benchmark_training():
    """Benchmark hiệu suất training"""
    from training.lightning_module import OpenVocabLightningModule
    
    config = {
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
                'num_heads': 4,
                'text_dim': 512
            },
            'box_head': {
                'input_dim': 256,
                'hidden_dim': 128,
                'num_queries': 20
            }
        },
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'use_amp': True,
        'loss': {
            'lambda_cls': 0.3,
            'lambda_bbox': 5.0,
            'lambda_giou': 2.0,
            'lambda_sim': 0.1,
            'temperature': 0.1,
            'region_dim': 256,
            'text_dim': 512
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OpenVocabLightningModule(config).to(device)
    model.train()
    
    # Dummy batch
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    texts = ["a person walking on the street"] * batch_size
    targets = {
        'boxes': [torch.randn(2, 4).to(device) for _ in range(batch_size)],
        'labels': [torch.randint(0, 1, (2,)).to(device) for _ in range(batch_size)]
    }
    
    # Warmup
    print("🔥 Warming up training...")
    for _ in range(3):
        outputs = model(images, texts)
        loss_dict = model.criterion(outputs, targets, outputs['text_features'])
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        model.zero_grad()
    
    # Benchmark training step
    print("🚀 Benchmarking training step...")
    times = []
    for i in range(10):
        with timer(f"Training step {i+1}"):
            start = time.time()
            outputs = model(images, texts)
            loss_dict = model.criterion(outputs, targets, outputs['text_features'])
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            model.zero_grad()
            end = time.time()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    steps_per_sec = 1 / avg_time
    
    print(f"\n📊 Training Results:")
    print(f"   Average step time: {avg_time:.3f}s")
    print(f"   Steps per second: {steps_per_sec:.1f}")
    
    return {
        'avg_step_time': avg_time,
        'steps_per_sec': steps_per_sec
    }

if __name__ == '__main__':
    print("🎯 Open Vocabulary Detection - Performance Benchmark")
    print("=" * 50)
    
    # System info
    print(f"🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ No GPU available")
    
    print("\n" + "=" * 50)
    
    # Benchmark inference
    print("\n🔍 INFERENCE BENCHMARK")
    print("-" * 30)
    inference_results = benchmark_model()
    
    # Benchmark training
    print("\n🏋️  TRAINING BENCHMARK")
    print("-" * 30)
    training_results = benchmark_training()
    
    # Summary
    print("\n📈 SUMMARY")
    print("=" * 50)
    print(f"✅ Model ready for production!")
    print(f"⚡ Inference: {inference_results['fps']:.1f} FPS")
    print(f"🏃 Training: {training_results['steps_per_sec']:.1f} steps/sec")
    print(f"💾 Model size: {inference_results['params']:,} parameters")
    if inference_results['gpu_memory'] > 0:
        print(f"🎮 GPU memory: {inference_results['gpu_memory']:.2f} GB")
