#!/usr/bin/env python3
"""
Script cài đặt dependencies cho Open Vocabulary Detection
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Chạy command và hiển thị kết quả"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed!")
        print(f"   Error: {e.stderr}")
        return False

def install_dependencies():
    """Cài đặt tất cả dependencies"""
    print("🚀 INSTALLING DEPENDENCIES FOR OPEN VOCABULARY DETECTION")
    print("=" * 60)
    
    # Danh sách dependencies theo thứ tự ưu tiên
    dependencies = [
        {
            "name": "PyTorch",
            "command": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "description": "Installing PyTorch with CUDA 11.8 support"
        },
        {
            "name": "PyTorch Lightning",
            "command": "pip install pytorch-lightning",
            "description": "Installing PyTorch Lightning"
        },
        {
            "name": "TIMM",
            "command": "pip install timm",
            "description": "Installing TIMM for vision models"
        },
        {
            "name": "Transformers",
            "command": "pip install transformers",
            "description": "Installing Transformers for text models"
        },
        {
            "name": "TorchMetrics",
            "command": "pip install torchmetrics[detection]",
            "description": "Installing TorchMetrics for evaluation"
        },
        {
            "name": "SciPy",
            "command": "pip install scipy",
            "description": "Installing SciPy for optimization"
        },
        {
            "name": "Other utilities",
            "command": "pip install matplotlib tensorboardX pillow opencv-python tqdm",
            "description": "Installing utility packages"
        }
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for dep in dependencies:
        print(f"\n📦 Installing {dep['name']}...")
        if run_command(dep['command'], dep['description']):
            success_count += 1
        else:
            print(f"   ⚠️  Failed to install {dep['name']}, continuing...")
    
    print(f"\n📊 INSTALLATION SUMMARY:")
    print(f"   ✅ Successfully installed: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("✅ Project is ready to run!")
        return True
    else:
        print("\n⚠️  Some dependencies failed to install.")
        print("   You may need to install them manually.")
        return False

def test_installation():
    """Test xem installation có thành công không"""
    print("\n🔍 TESTING INSTALLATION...")
    print("=" * 40)
    
    test_imports = [
        "import torch",
        "import torchvision", 
        "import pytorch_lightning",
        "import timm",
        "import transformers",
        "import scipy"
    ]
    
    success_count = 0
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"   ✅ {test_import}")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {test_import} - {e}")
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   ✅ Successful: {success_count}/{len(test_imports)}")
    
    if success_count == len(test_imports):
        print("   🎉 All tests passed!")
        return True
    else:
        print("   ⚠️  Some tests failed!")
        return False

def main():
    """Main function"""
    print("🎯 OPEN VOCABULARY DETECTION - DEPENDENCY INSTALLER")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required!")
        return False
    
    # Install dependencies
    install_success = install_dependencies()
    
    if install_success:
        # Test installation
        test_success = test_installation()
        
        if test_success:
            print("\n🎉 SETUP COMPLETE!")
            print("✅ You can now run the project!")
            print("\nNext steps:")
            print("1. python scripts/train_optimized.py  # Start training")
            print("2. python scripts/benchmark.py        # Benchmark performance")
            print("3. python run.py test                 # Test model")
        else:
            print("\n⚠️  Setup completed but some tests failed.")
            print("   You may need to install missing packages manually.")
    else:
        print("\n❌ Setup failed!")
        print("   Please check the error messages above.")

if __name__ == '__main__':
    main()
