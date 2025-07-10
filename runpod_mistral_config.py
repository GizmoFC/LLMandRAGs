"""
RunPod Configuration for Mistral Model
Optimized settings for GPU container deployment
"""

# Model Configuration
MISTRAL_MODEL = "mistral:7b-instruct"
OLLAMA_URL = "http://localhost:11434"

# GPU-Optimized Settings
GPU_CONFIG = {
    "max_context_docs": 8,      # More context with GPU
    "max_prompt_length": 4000,  # Larger prompts
    "timeout": 60,              # Faster with GPU
    "temperature": 0.3,
    "num_predict": 1024,        # Longer responses
    "top_k": 20,                # More candidates
    "gpu_layers": 50,           # GPU acceleration
    "gpu_memory_utilization": 0.9
}

# Query Type Configurations
QUERY_CONFIGS = {
    "clause_improvement": {
        "max_context_docs": 8,
        "max_prompt_length": 4000,
        "temperature": 0.4,
        "num_predict": 1024
    },
    "clause_analysis": {
        "max_context_docs": 6,
        "max_prompt_length": 3500,
        "temperature": 0.2,
        "num_predict": 768
    },
    "clause_comparison": {
        "max_context_docs": 10,
        "max_prompt_length": 4500,
        "temperature": 0.3,
        "num_predict": 1280
    },
    "clause_search": {
        "max_context_docs": 4,
        "max_prompt_length": 2000,
        "temperature": 0.1,
        "num_predict": 512
    }
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    "enable_memory_monitoring": True,
    "log_performance_metrics": True,
    "gpu_memory_threshold": 0.8  # 80% GPU memory usage threshold
} 