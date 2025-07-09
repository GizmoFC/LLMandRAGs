# GPU Deployment Guide for RAG System

## 🎯 **Option 1: Ollama with GPU Acceleration (Easiest)**

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker installed
- Ollama with GPU support

### Setup Steps

1. **Install Ollama with GPU Support**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull GPU-optimized models
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull gemma2:9b-instruct-q4_K_M
ollama pull codellama:13b-instruct-q4_K_M
```

2. **Configure Ollama for GPU**
```bash
# Create Ollama config
mkdir -p ~/.ollama
cat > ~/.ollama/config.json << EOF
{
  "gpu_layers": 50,
  "numa": false,
  "gpu_memory_utilization": 0.9
}
EOF
```

3. **Update Your RAG Configuration**
```python
# In rag_search_enhanced.py - update model configurations
self.config = {
    "max_context_docs": 6,      # Can handle more with GPU
    "max_prompt_length": 3000,  # Larger prompts
    "timeout": 120,             # Faster with GPU
    "temperature": 0.3,
    "num_predict": 1024,        # Longer responses
    "top_k": 15,                # More candidates
}

# Use larger models
"model": "llama3.1:8b-instruct-q4_K_M"  # GPU-optimized
```

## 🎯 **Option 2: Direct CUDA Deployment (Maximum Performance)**

### Prerequisites
- NVIDIA GPU with CUDA 11.8+
- Python with CUDA support
- Transformers library

### Setup Steps

1. **Install CUDA Dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install sentence-transformers
```

2. **Create GPU-Optimized RAG System**
```python
# gpu_rag_system.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class GPURAGSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models on GPU
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True  # 4-bit quantization for memory efficiency
        )
        
        # GPU-optimized embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device
        )
        
        # Load FAISS index (can be on CPU)
        self.faiss_index = faiss.read_index("embeddings/output/cuad_faiss.index")
        
    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]  # Remove input prompt
```

## 🎯 **Option 3: Docker Container Deployment**

### Create Dockerfile
```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-pip \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8012

# Run application
CMD ["python3", "unified_rag_dashboard.py"]
```

### Docker Compose for GPU
```yaml
# docker-compose.gpu.yml
version: '3.8'
services:
  rag-system:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8012:8012"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/app/data
      - ./embeddings:/app/embeddings
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

## 🎯 **Option 4: Cloud GPU Deployment**

### AWS SageMaker Setup
```python
# sagemaker_deployment.py
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

def deploy_to_sagemaker():
    role = get_execution_role()
    
    # Create model
    model = PyTorchModel(
        model_data='s3://your-bucket/model.tar.gz',
        role=role,
        entry_point='inference.py',
        source_dir='./code',
        framework_version='2.0.0',
        py_version='py39',
        instance_type='ml.g4dn.xlarge'  # GPU instance
    )
    
    # Deploy
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.g4dn.xlarge'
    )
    
    return predictor
```

## 📊 **Performance Comparison**

| Setup | Model Size | Speed | Memory | Cost |
|-------|------------|-------|--------|------|
| **CPU Only** | 2B-8B | 1x | 8-16GB | $0 |
| **Ollama + GPU** | 8B-13B | 5-10x | 8-16GB | $0 |
| **Direct CUDA** | 8B-70B | 10-20x | 16-32GB | $0 |
| **Cloud GPU** | 8B-70B | 10-20x | 16-32GB | $1-5/hr |

## 🔧 **Recommended GPU Configuration**

### For Your Use Case:
- **GPU**: NVIDIA RTX 4090 or A100 (16-24GB VRAM)
- **Model**: Llama 3.1 8B or Gemma 2 9B
- **Quantization**: 4-bit (Q4_K_M) for memory efficiency
- **Context**: 6-8 documents (can handle more with GPU)

### Expected Performance:
- **Search**: 0.1-0.3 seconds (same)
- **Generation**: 2-8 seconds (vs 15-60 seconds on CPU)
- **Total**: 3-10 seconds (vs 20-80 seconds on CPU)

## 🚀 **Quick Start: Ollama GPU Setup**

1. **Install Ollama with GPU support**
2. **Pull GPU-optimized models**
3. **Update your config to use larger models**
4. **Test with your existing dashboard**

Would you like me to create the specific configuration files for your setup? 