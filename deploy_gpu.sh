#!/bin/bash

# GPU-Optimized RAG System Deployment Script
echo "🚀 Deploying GPU-Optimized RAG System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not work."
    echo "   Make sure you have nvidia-docker2 installed and configured."
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t rag-gpu-system .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start the services
echo "🚀 Starting GPU-optimized RAG system..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
if curl -f http://localhost:8013/ > /dev/null 2>&1; then
    echo "✅ GPU Dashboard is running at http://localhost:8013"
else
    echo "⚠️  Dashboard not ready yet. Check logs with: docker-compose logs"
fi

if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama API is running at http://localhost:11434"
else
    echo "⚠️  Ollama not ready yet. Check logs with: docker-compose logs"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Access dashboard: http://localhost:8013"
echo "   Check Ollama: curl http://localhost:11434/api/tags"
echo ""
echo "🔧 To monitor GPU usage:"
echo "   docker exec -it rag-gpu-system_rag-gpu-system_1 nvidia-smi" 