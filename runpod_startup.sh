#!/bin/bash

# RunPod Startup Script for Mistral RAG System
echo "🚀 Starting RunPod GPU RAG System with Mistral..."

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  No GPU detected, using CPU"
fi

# Install dependencies if not already installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Create Ollama config for GPU
mkdir -p ~/.ollama
cat > ~/.ollama/config.json << EOF
{
  "gpu_layers": 50,
  "numa": false,
  "gpu_memory_utilization": 0.9
}
EOF

# Check if Mistral model is installed
if ! ollama list | grep -q "mistral:7b-instruct"; then
    echo "📥 Installing Mistral model..."
    ollama pull mistral:7b-instruct
else
    echo "✅ Mistral model already installed"
fi

# Start Ollama in background
echo "🔧 Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
sleep 10

# Test Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama failed to start"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install fastapi uvicorn requests sentence-transformers faiss-cpu numpy

# Start the GPU dashboard
echo "🌐 Starting GPU Dashboard..."
python3 gpu_rag_dashboard.py &

# Wait for dashboard to start
sleep 5

# Check if dashboard is running
if curl -s http://localhost:8013/ > /dev/null; then
    echo "✅ GPU Dashboard is running at http://localhost:8013"
else
    echo "⚠️  Dashboard not ready yet"
fi

echo ""
echo "🎉 RunPod RAG System is ready!"
echo ""
echo "📋 Access Points:"
echo "   Dashboard: http://localhost:8013"
echo "   Ollama API: http://localhost:11434"
echo ""
echo "🔧 Useful Commands:"
echo "   Check GPU: nvidia-smi"
echo "   Check Ollama: curl http://localhost:11434/api/tags"
echo "   Check Dashboard: curl http://localhost:8013/"
echo ""
echo "💡 Example Queries:"
echo "   - 'Analyze this confidentiality clause: [paste clause]'"
echo "   - 'Improve this termination clause: [paste clause]'"
echo "   - 'Compare intellectual property clauses across contracts'"
echo "   - 'Find data protection and privacy clauses'"

# Keep the script running
wait $OLLAMA_PID 