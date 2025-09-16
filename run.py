#!/usr/bin/env python3
"""
🚀 Open Vocabulary Detection - Main Runner Script

Script tổng hợp để chạy các tác vụ chính của dự án.
Sử dụng: python run.py [command]
"""

import sys
import os
import subprocess
import argparse

def run_command(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n🔄 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Thành công!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi: {e}")
        if e.stderr:
            print(f"Chi tiết: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Open Vocabulary Detection Runner')
    parser.add_argument('command', choices=[
        'test', 'train', 'train-opt', 'compare', 'eval', 'all'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    print("🎯 Open Vocabulary Detection")
    print("=" * 50)
    
    if args.command == 'test':
        success = run_command(
            "python scripts/quick_test.py",
            "Test nhanh model và memory"
        )
        
    elif args.command == 'train':
        success = run_command(
            "python scripts/train.py",
            "Training với model gốc"
        )
        
    elif args.command == 'train-opt':
        success = run_command(
            "python scripts/train_optimized.py", 
            "Training với model tối ưu (khuyến nghị)"
        )
        
    elif args.command == 'compare':
        success = run_command(
            "python scripts/compare_models.py",
            "So sánh hiệu suất model cũ vs mới"
        )
        
    elif args.command == 'eval':
        success = run_command(
            "python scripts/evaluate.py",
            "Đánh giá model đã train"
        )
        
    elif args.command == 'all':
        print("🚀 Chạy tất cả các bước...")
        
        steps = [
            ("python scripts/quick_test.py", "1. Test nhanh"),
            ("python scripts/compare_models.py", "2. So sánh hiệu suất"),
            ("python scripts/train_optimized.py", "3. Training tối ưu")
        ]
        
        for cmd, desc in steps:
            if not run_command(cmd, desc):
                print(f"❌ Dừng tại bước: {desc}")
                break
        else:
            print("\n🎉 Hoàn thành tất cả các bước!")
            print("📊 Xem kết quả: tensorboard --logdir lightning_logs")
    
    if success:
        print(f"\n✅ Hoàn thành: {args.command}")
    else:
        print(f"\n❌ Lỗi khi chạy: {args.command}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("🎯 Open Vocabulary Detection - Main Runner")
        print("=" * 50)
        print("Sử dụng:")
        print("  python run.py test        # Test nhanh")
        print("  python run.py train-opt   # Training tối ưu (khuyến nghị)")
        print("  python run.py train       # Training gốc")
        print("  python run.py compare     # So sánh hiệu suất")
        print("  python run.py eval        # Đánh giá model")
        print("  python run.py all         # Chạy tất cả")
        print("\n📚 Xem thêm: HOW_TO_RUN.md")
    else:
        main()
