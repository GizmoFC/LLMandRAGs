"""
GPU-Optimized RAG Dashboard
Enhanced dashboard for GPU deployment with larger models and better performance
"""

import json
import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from typing import Optional

# Import the GPU-optimized RAG system
from gpu_rag_system import GPURAGSystem
from memory_monitor import MemoryMonitor

# Initialize GPU-optimized systems
gpu_rag_system = GPURAGSystem()
memory_monitor = MemoryMonitor()

app = FastAPI(title="GPU-Optimized RAG Dashboard", version="2.0")

# HTML template for the GPU dashboard
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>GPU-Optimized RAG Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
        .container {{ background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px #0001; }}
        h1 {{ color: #007bff; }}
        .gpu-badge {{ background: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }}
        textarea, select {{ width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }}
        button {{ background: #007bff; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
        .response, .context, .memory {{ margin: 18px 0; padding: 16px; border-radius: 8px; }}
        .response {{ background: #e9f7ef; border-left: 5px solid #28a745; }}
        .context {{ background: #f1f3f4; font-size: 0.95em; }}
        .memory {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
        .stats {{ background: #e2e3e5; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .section {{ margin-bottom: 32px; }}
        .performance {{ background: #d4edda; border-left: 5px solid #28a745; padding: 10px; margin: 10px 0; }}
        .model-selector {{ display: flex; gap: 10px; margin: 10px 0; flex-wrap: wrap; }}
        .model-option {{ padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }}
        .model-option.selected {{ background: #007bff; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU-Optimized RAG Dashboard <span class="gpu-badge">GPU ACCELERATED</span></h1>
        
        <div class="performance">
            <strong>⚡ Performance Mode:</strong> GPU-accelerated with Mistral and enhanced context
        </div>
        
        <div class="section">
            <form method="post">
                <label><strong>Query:</strong></label><br>
                <textarea name="query" placeholder="Ask about clauses, request improvements, or analyze legal text...">{query}</textarea><br>
                
                <label><strong>Model Selection:</strong></label><br>
                <div class="model-selector">
                    <div class="model-option {model_mistral}" onclick="selectModel('mistral:7b-instruct')">🌪️ Mistral 7B (Recommended)</div>
                    <div class="model-option {model_llama}" onclick="selectModel('llama3.1:8b-instruct-q4_K_M')">🦙 Llama 3.1 8B (Fast)</div>
                    <div class="model-option {model_gemma}" onclick="selectModel('gemma2:9b-instruct-q4_K_M')">💎 Gemma 2 9B (Balanced)</div>
                    <div class="model-option {model_code}" onclick="selectModel('codellama:13b-instruct-q4_K_M')">🐍 CodeLlama 13B (Advanced)</div>
                </div>
                <input type="hidden" name="model" id="selected_model" value="{selected_model}">
                
                <button type="submit">🚀 Process Query (GPU Accelerated)</button>
            </form>
        </div>
        
        <div class="section">
            <h3>💡 GPU-Optimized Example Queries:</h3>
            <ul>
                <li><strong>Complex Analysis:</strong> "Analyze the risks and benefits of this non-compete clause: [paste clause here]"</li>
                <li><strong>Multi-Document Comparison:</strong> "Compare intellectual property clauses across software, consulting, and employment contracts"</li>
                <li><strong>Advanced Improvement:</strong> "Improve this termination clause with best practices from similar industries: [paste clause here]"</li>
                <li><strong>Comprehensive Search:</strong> "Find all clauses related to data protection, privacy, and cybersecurity with specific examples"</li>
            </ul>
        </div>
        
        {results}
        
        <div class="section">
            <h3>🖥️ GPU Performance Status</h3>
            <div class="memory">
                <pre>{memory_status}</pre>
            </div>
        </div>
    </div>
    
    <script>
        function selectModel(model) {{
            // Update visual selection
            document.querySelectorAll('.model-option').forEach(opt => opt.classList.remove('selected'));
            event.target.classList.add('selected');
            
            // Update hidden input
            document.getElementById('selected_model').value = model;
        }}
        
        // Initialize with default model
        document.addEventListener('DOMContentLoaded', function() {{
            selectModel('{selected_model}');
        }});
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    memory_status = json.dumps(memory_monitor.get_current_memory_status(), indent=2)
    return html_template.format(
        query="", 
        results="", 
        memory_status=memory_status,
        model_mistral="selected",
        model_llama="",
        model_gemma="",
        model_code="",
        selected_model="mistral:7b-instruct"
    )

@app.post("/", response_class=HTMLResponse)
async def process_query(query: str = Form(...), model: str = Form("mistral:7b-instruct")):
    memory_status = json.dumps(memory_monitor.get_current_memory_status(), indent=2)
    
    if not query.strip():
        return html_template.format(
            query="", 
            results="<p>Please provide a query.</p>", 
            memory_status=memory_status,
            model_mistral="selected" if "mistral" in model else "",
            model_llama="selected" if "llama" in model else "",
            model_gemma="selected" if "gemma" in model else "",
            model_code="selected" if "code" in model else "",
            selected_model=model
        )
    
    # Process with GPU-optimized system
    result = gpu_rag_system.process_query(query, model)
    
    # Format response and context
    response_html = f"""
    <div class='stats'>
        <strong>Query Type:</strong> {result.get('query_type', 'Unknown')}<br>
        <strong>Model Used:</strong> {result.get('model_used', 'Unknown')}<br>
        <strong>Device:</strong> {result.get('device', 'Unknown')}<br>
        <strong>Processing Time:</strong> {result.get('processing_time', 0):.2f}s<br>
        <strong>Context Size:</strong> {result.get('context_size', 0)} words<br>
        <strong>Context Documents:</strong> {len(result.get('context_docs', []))}
    </div>
    <div class='response'><h3>🤖 GPU-Accelerated Answer:</h3><p>{result['answer'].replace(chr(10), '<br>')}</p></div>
    <div class='response'><h3>📚 Referenced Contracts & Clauses:</h3>{chr(10).join([f'<div class="context"><strong>Contract:</strong> {doc.get("contract_title", doc.get("title", "Unknown Contract"))}<br><strong>Clause Type:</strong> {doc.get("clause_type", "General")}<br><strong>Relevance Score:</strong> {doc.get("relevance_score", "N/A") if isinstance(doc.get("relevance_score"), (int, float)) else "N/A"}<br><strong>Excerpt:</strong> {doc.get("text", "") if len(doc.get("text", "")) > 50 else doc.get("full_context", "")[:400]}...</div>' for doc in result.get('context_docs', [])])}</div>
    """
    
    return html_template.format(
        query=query, 
        results=response_html, 
        memory_status=memory_status,
        model_mistral="selected" if "mistral" in model else "",
        model_llama="selected" if "llama" in model else "",
        model_gemma="selected" if "gemma" in model else "",
        model_code="selected" if "code" in model else "",
        selected_model=model
    )

@app.get("/api/gpu_status")
async def api_gpu_status():
    """Get GPU status and performance metrics"""
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    
    if gpu_available:
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    else:
        gpu_info = {
            "gpu_available": False,
            "message": "No GPU detected, using CPU"
        }
    
    return gpu_info

@app.get("/api/memory_status")
async def api_memory_status():
    return memory_monitor.get_current_memory_status()

@app.get("/api/available_models")
async def api_available_models():
    """Get list of available GPU-optimized models"""
    return {
        "models": [
            {
                "name": "llama3.1:8b-instruct-q4_K_M",
                "description": "Fast and efficient 8B parameter model",
                "best_for": "General queries, quick responses"
            },
            {
                "name": "gemma2:9b-instruct-q4_K_M", 
                "description": "Balanced 9B parameter model",
                "best_for": "Analysis and comparison tasks"
            },
            {
                "name": "codellama:13b-instruct-q4_K_M",
                "description": "Advanced 13B parameter model",
                "best_for": "Complex legal analysis and drafting"
            }
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting GPU-Optimized RAG Dashboard at http://localhost:8013")
    print("⚡ GPU acceleration enabled for enhanced performance")
    uvicorn.run(app, host="0.0.0.0", port=8013) 