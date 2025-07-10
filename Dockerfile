# GPU-Optimized RAG System Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional GPU dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    transformers \
    accelerate \
    bitsandbytes \
    sentence-transformers \
    faiss-gpu

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create Ollama configuration for GPU
RUN mkdir -p /root/.ollama
RUN echo '{"gpu_layers": 50, "numa": false, "gpu_memory_utilization": 0.9}' > /root/.ollama/config.json

# Copy application files
COPY . .

# Create directories for data and embeddings
RUN mkdir -p /app/data /app/embeddings/output

# Expose ports
EXPOSE 8013 11434

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting GPU-Optimized RAG System..."\n\
echo "📦 Installing GPU models..."\n\
ollama pull llama3.1:8b-instruct-q4_K_M &\n\
ollama pull gemma2:9b-instruct-q4_K_M &\n\
ollama pull codellama:13b-instruct-q4_K_M &\n\
wait\n\
echo "✅ Models installed"\n\
echo "🔧 Starting Ollama..."\n\
ollama serve &\n\
sleep 5\n\
echo "🌐 Starting GPU Dashboard..."\n\
python3 gpu_rag_dashboard.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8013/ || exit 1

# Default command
CMD ["/app/start.sh"] 