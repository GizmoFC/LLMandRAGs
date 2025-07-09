"""
Install quantized Gemma3 model for better memory efficiency
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🧠 Installing Quantized Gemma3 for Better Memory Efficiency")
    print("=" * 60)
    
    # Check if Ollama is available
    success, stdout, stderr = run_command("ollama --version")
    if not success:
        print("❌ Ollama not found. Please install Ollama first.")
        print("   Download from: https://ollama.ai/")
        return
    
    print("✅ Ollama found")
    
    # Available quantized models
    quantized_models = [
        "gemma3:2b",           # ~1.5GB - Very fast, good for simple tasks
        "gemma3:2b-instruct",  # ~1.5GB - Instruction-tuned version
        "gemma3:8b",           # ~4.5GB - Better quality, similar to current
        "gemma3:8b-instruct",  # ~4.5GB - Instruction-tuned version
        "llama3.1:8b",         # ~4.5GB - Alternative high-quality model
        "llama3.1:8b-instruct" # ~4.5GB - Instruction-tuned version
    ]
    
    print("\n📋 Available quantized models:")
    for i, model in enumerate(quantized_models, 1):
        print(f"  {i}. {model}")
    
    print("\n💡 Recommendations:")
    print("  - For complex clause analysis: gemma3:8b-instruct")
    print("  - For fast responses: gemma3:2b-instruct")
    print("  - For best quality: llama3.1:8b-instruct")
    
    # Try to install a recommended model
    recommended_model = "gemma3:8b-instruct"
    print(f"\n🚀 Installing recommended model: {recommended_model}")
    
    success, stdout, stderr = run_command(f"ollama pull {recommended_model}")
    
    if success:
        print(f"✅ Successfully installed {recommended_model}")
        print("\n📝 Next steps:")
        print("1. Update your RAG server to use the new model")
        print("2. Test with complex queries")
        print("3. Adjust context size based on performance")
    else:
        print(f"❌ Failed to install {recommended_model}")
        print(f"Error: {stderr}")
        
        # Try a smaller model
        print(f"\n🔄 Trying smaller model: gemma3:2b-instruct")
        success, stdout, stderr = run_command("ollama pull gemma3:2b-instruct")
        
        if success:
            print("✅ Successfully installed gemma3:2b-instruct")
            print("This model uses much less memory and should work well for most tasks.")
        else:
            print("❌ Failed to install smaller model too.")
            print("Please check your internet connection and try manually:")
            print("  ollama pull gemma3:2b-instruct")

if __name__ == "__main__":
    main() 