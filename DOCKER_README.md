# 🐳 Docker Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (nvidia-docker2)

### Deploy with One Command
```bash
chmod +x deploy_gpu.sh
./deploy_gpu.sh
```

## 📋 Manual Deployment

### 1. Build the Image
```bash
docker build -t rag-gpu-system .
```

### 2. Start Services
```bash
docker-compose up -d
```

### 3. Check Status
```bash
docker-compose ps
docker-compose logs -f
```

## 🌐 Access Points

- **GPU Dashboard**: http://localhost:8013
- **Ollama API**: http://localhost:11434
- **Health Check**: http://localhost:8013/

## 🔧 Configuration

### Environment Variables
```yaml
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
```

### GPU Configuration
- **GPU Layers**: 50
- **Memory Utilization**: 90%
- **NUMA**: Disabled

### Ports
- **8013**: Web Dashboard
- **11434**: Ollama API

## 📦 Included Models

The container automatically installs:
- `llama3.1:8b-instruct-q4_K_M` (Fast)
- `gemma2:9b-instruct-q4_K_M` (Balanced)
- `codellama:13b-instruct-q4_K_M` (Advanced)

## 🛠️ Management Commands

### View Logs
```bash
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### Update and Redeploy
```bash
docker-compose down
docker build -t rag-gpu-system .
docker-compose up -d
```

### Monitor GPU Usage
```bash
docker exec -it rag-gpu-system_rag-gpu-system_1 nvidia-smi
```

## 🔍 Troubleshooting

### Check GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Check Ollama Status
```bash
curl http://localhost:11434/api/tags
```

### Check Dashboard Status
```bash
curl http://localhost:8013/
```

### View Container Logs
```bash
docker-compose logs rag-gpu-system
```

## 📊 Performance

### Expected Performance (GPU)
- **Search Time**: 0.1-0.3 seconds
- **Generation Time**: 2-8 seconds
- **Total Response**: 3-10 seconds
- **Context Documents**: 6-10
- **Model Size**: 8B-13B parameters

### Memory Requirements
- **GPU VRAM**: 16-24GB recommended
- **System RAM**: 32GB recommended
- **Storage**: 50GB for models + data

## 🔒 Security Notes

- Container runs as root (required for GPU access)
- Ollama API exposed on localhost only
- No external network access by default
- Data volumes mounted for persistence

## 📈 Scaling

### Multiple GPUs
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
          capabilities: [gpu]
```

### Multiple Instances
```bash
docker-compose up -d --scale rag-gpu-system=2
```

## 🎯 Use Cases

- **Contract Analysis**: Complex legal document analysis
- **Clause Improvement**: AI-powered contract drafting
- **Legal Research**: Multi-document comparison
- **Compliance Checking**: Automated contract review

## 📞 Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify GPU: `nvidia-smi`
3. Test Ollama: `curl http://localhost:11434/api/tags`
4. Check dashboard: http://localhost:8013 