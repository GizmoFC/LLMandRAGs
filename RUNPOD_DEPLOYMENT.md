# 🚀 RunPod Deployment Guide (Mistral)

## Quick Setup

### 1. Upload Files to RunPod
Upload these files to your RunPod container:
```
gpu_rag_dashboard.py
gpu_rag_system.py
rag_search_enhanced.py
complex_query_handler.py
memory_monitor.py
requirements.txt
cuad_clause_library.jsonl
embeddings/output/cuad_faiss.index
runpod_startup.sh
```

### 2. Make Script Executable
```bash
chmod +x runpod_startup.sh
```

### 3. Run the Startup Script
```bash
./runpod_startup.sh
```

## Manual Setup (Alternative)

### 1. Install Dependencies
```bash
# Install Python packages
pip install fastapi uvicorn requests sentence-transformers faiss-cpu numpy

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Configure Ollama for GPU
```bash
mkdir -p ~/.ollama
cat > ~/.ollama/config.json << EOF
{
  "gpu_layers": 50,
  "numa": false,
  "gpu_memory_utilization": 0.9
}
EOF
```

### 3. Install Mistral Model
```bash
ollama pull mistral:7b-instruct
```

### 4. Start Ollama
```bash
ollama serve &
```

### 5. Start the Dashboard
```bash
python3 gpu_rag_dashboard.py
```

## Access Points

- **Dashboard**: http://localhost:8013
- **Ollama API**: http://localhost:11434

## Test the System

### Check GPU
```bash
nvidia-smi
```

### Check Ollama
```bash
curl http://localhost:11434/api/tags
```

### Test Dashboard
```bash
curl http://localhost:8013/
```

## Example Queries

1. **"Analyze this confidentiality clause: [paste clause here]"**
2. **"Improve this termination clause: [paste clause here]"**
3. **"Compare intellectual property clauses across different contracts"**
4. **"Find data protection and privacy clauses with examples"**

## Performance

- **Response Time**: 3-8 seconds
- **Context Documents**: 6-10
- **Model**: Mistral 7B (7 billion parameters)
- **GPU Acceleration**: Full CUDA support

## Troubleshooting

### If Ollama fails to start:
```bash
pkill ollama
ollama serve &
```

### If dashboard fails to start:
```bash
# Check if port 8013 is free
netstat -tulpn | grep 8013
# Kill process if needed
kill -9 <PID>
```

### If GPU not detected:
```bash
nvidia-smi
# Check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Model Options

The dashboard supports multiple models:
- **Mistral 7B** (Recommended) - Balanced performance
- **Llama 3.1 8B** - Fast responses
- **Gemma 2 9B** - Good analysis
- **CodeLlama 13B** - Advanced capabilities

## Monitoring

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Memory Usage
```bash
htop
```

### Logs
```bash
# Check Ollama logs
tail -f ~/.ollama/logs/ollama.log

# Check dashboard logs (if running in terminal)
# They will appear in the terminal where you started the dashboard
``` 