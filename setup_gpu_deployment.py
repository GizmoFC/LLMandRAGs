#!/usr/bin/env python3
"""
GPU Deployment Setup Script
Automates the setup process for GPU-accelerated RAG system
"""

import subprocess
import sys
import os
import json
import requests
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        return False

def install_gpu_models():
    """Install GPU-optimized models"""
    models = [
        "llama3.1:8b-instruct-q4_K_M",
        "gemma2:9b-instruct-q4_K_M", 
        "codellama:13b-instruct-q4_K_M"
    ]
    
    print("\n📥 Installing GPU-optimized models...")
    
    for model in models:
        print(f"   Installing {model}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            if result.returncode == 0:
                print(f"   ✅ {model} installed successfully")
            else:
                print(f"   ❌ Failed to install {model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Timeout installing {model}")
        except FileNotFoundError:
            print("   ❌ Ollama not found in PATH")
            break

def create_ollama_config():
    """Create Ollama configuration for GPU optimization"""
    config_dir = Path.home() / ".ollama"
    config_file = config_dir / "config.json"
    
    config = {
        "gpu_layers": 50,
        "numa": False,
        "gpu_memory_utilization": 0.9
    }
    
    try:
        config_dir.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Ollama GPU configuration created")
    except Exception as e:
        print(f"❌ Failed to create Ollama config: {e}")

def install_python_dependencies():
    """Install Python dependencies for GPU support"""
    print("\n📦 Installing Python dependencies...")
    
    dependencies = [
        "torch",
        "torchvision", 
        "torchaudio",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"   ✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {dep}")

def test_gpu_system():
    """Test the GPU-optimized system"""
    print("\n🧪 Testing GPU system...")
    
    try:
        from gpu_rag_system import GPURAGSystem
        
        # Initialize system
        system = GPURAGSystem()
        
        # Test query
        test_query = "Find intellectual property clauses"
        result = system.process_query(test_query)
        
        print(f"✅ GPU system test successful")
        print(f"   Query type: {result.get('query_type')}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Device: {result.get('device')}")
        print(f"   Context docs: {len(result.get('context_docs', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU system test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 GPU Deployment Setup for RAG System")
    print("=" * 50)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Check Ollama
    ollama_running = check_ollama_installation()
    
    if not ollama_running:
        print("\n❌ Please start Ollama first:")
        print("   ollama serve")
        return
    
    # Install Python dependencies if GPU is available
    if gpu_available:
        install_python_dependencies()
    
    # Create Ollama config
    create_ollama_config()
    
    # Install GPU models
    install_gpu_models()
    
    # Test the system
    if test_gpu_system():
        print("\n🎉 GPU deployment setup complete!")
        print("\n📋 Next steps:")
        print("   1. Run: python gpu_rag_dashboard.py")
        print("   2. Open: http://localhost:8013")
        print("   3. Select your preferred model")
        print("   4. Start querying with GPU acceleration!")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("   You can still use the system with CPU")

if __name__ == "__main__":
    main() 