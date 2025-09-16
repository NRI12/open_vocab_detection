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
        'test', 'train', 'eval', 'all'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    print("🎯 Open Vocabulary Detection")
    print("=" * 50)
    
    if args.command == 'test':
        success = run_command(
            "python -c \"print('Python OK')\"",
            "Test nhanh Python"
        )
        
    elif args.command == 'train':
        success = run_command(
            "python scripts/train.py",
            "Training model tối ưu"
        )
        
    elif args.command == 'eval':
        success = run_command(
            "python scripts/evaluate.py",
            "Đánh giá model đã train"
        )
        
    elif args.command == 'all':
        print("🚀 Chạy tất cả các bước...")
        
        steps = [
            ("python -c \"print('Python OK')\"", "1. Test Python"),
            ("python scripts/train.py", "2. Training")
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
        print("  python run.py test        # Test model")
        print("  python run.py train       # Training")
        print("  python run.py eval        # Đánh giá")
        print("  python run.py all         # Chạy tất cả")
        print("\n📚 Xem thêm: README.md")
    else:
        main()
