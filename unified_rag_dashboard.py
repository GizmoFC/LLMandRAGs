"""
Unified RAG Dashboard
A single web interface for advanced contract clause search, improvement, and memory monitoring.
"""

import json
import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from typing import Optional

# Import the enhanced RAG, complex query handler, and memory monitor
from rag_search_enhanced import EnhancedRAGSystem
from complex_query_handler import ComplexQueryHandler
from memory_monitor import MemoryMonitor

# Initialize systems with lightweight model
rag_system = EnhancedRAGSystem()
complex_handler = ComplexQueryHandler()
memory_monitor = MemoryMonitor()

# Override default models to use lightweight gemma3:latest
rag_system.config["model"] = "gemma3:latest"
for config in complex_handler.configs.values():
    config.model = "gemma3:latest"

app = FastAPI(title="Unified RAG Dashboard", version="1.0")

# HTML template for the dashboard
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Unified RAG Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
        .container {{ background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px #0001; }}
        h1 {{ color: #007bff; }}
        textarea, select {{ width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }}
        button {{ background: #007bff; color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
        .response, .context, .memory {{ margin: 18px 0; padding: 16px; border-radius: 8px; }}
        .response {{ background: #e9f7ef; border-left: 5px solid #28a745; }}
        .context {{ background: #f1f3f4; font-size: 0.95em; }}
        .memory {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
        .stats {{ background: #e2e3e5; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .section {{ margin-bottom: 32px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Unified RAG Dashboard</h1>
        <div class="section">
            <form method="post">
                <label><strong>Query:</strong></label><br>
                <textarea name="query" placeholder="Ask about clauses, request improvements, or analyze legal text...">{query}</textarea><br>
                <label><strong>Handler:</strong></label><br>
                <select name="handler">
                    <option value="rag">Enhanced RAG (context-aware)</option>
                    <option value="complex">Complex Query Handler (memory-optimized)</option>
                </select><br>
                <button type="submit">🚀 Process Query</button>
            </form>
        </div>
        <div class="section">
            <h3>💡 Example Queries:</h3>
            <ul>
                <li><strong>Clause Improvement:</strong> "Improve this confidentiality clause: [paste clause here]"</li>
                <li><strong>Analysis:</strong> "Analyze the risks in this intellectual property clause"</li>
                <li><strong>Search:</strong> "Find clauses related to data protection and privacy"</li>
                <li><strong>Comparison:</strong> "Compare termination clauses across different contract types"</li>
            </ul>
        </div>
        {results}
        <div class="section">
            <h3>🖥️ Real-Time Memory Status</h3>
            <div class="memory">
                <pre>{memory_status}</pre>
            </div>
        </div>
    </div>
</body>
</html>
"""

DEFAULT_MODEL = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434"

@app.get("/", response_class=HTMLResponse)
async def home():
    memory_status = json.dumps(memory_monitor.get_current_memory_status(), indent=2)
    return html_template.format(
        query="", 
        results="", 
        memory_status=memory_status
    )

@app.post("/", response_class=HTMLResponse)
async def process_query(query: str = Form(...), handler: str = Form("rag")):
    memory_status = json.dumps(memory_monitor.get_current_memory_status(), indent=2)
    if not query.strip():
        return html_template.format(query="", results="<p>Please provide a query.</p>", memory_status=memory_status)
    
    if handler == "rag":
        result = rag_system.process_query(query, model=DEFAULT_MODEL)
    else:
        result = complex_handler.process_complex_query(query)
    
    # Format response and context
    response_html = f"""
    <div class='stats'>
        <strong>Query Type:</strong> {result.get('query_type', 'Unknown')}<br>
        <strong>Model Used:</strong> {result.get('model_used', 'Unknown')}<br>
        <strong>Processing Time:</strong> {result.get('processing_time', 0):.2f}s<br>
        <strong>Context Size:</strong> {result.get('context_size', result.get('prompt_length', 0))} words<br>
        <strong>Context Documents:</strong> {len(result.get('context_docs', []))}
    </div>
    <div class='response'><h3>🤖 Answer:</h3><p>{result['answer'].replace(chr(10), '<br>')}</p></div>
    <div class='response'><h3>📚 Referenced Contracts & Clauses:</h3>{chr(10).join([f'<div class="context"><strong>Contract:</strong> {doc.get("contract_title", doc.get("title", "Unknown Contract"))}<br><strong>Clause Type:</strong> {doc.get("clause_type", "General")}<br><strong>Relevance Score:</strong> {doc.get("relevance_score", "N/A") if isinstance(doc.get("relevance_score"), (int, float)) else "N/A"}<br><strong>Excerpt:</strong> {doc.get("text", "") if len(doc.get("text", "")) > 50 else doc.get("full_context", "")[:300]}...</div>' for doc in result.get('context_docs', [])])}</div>
    """
    return html_template.format(query=query, results=response_html, memory_status=memory_status)

@app.get("/api/memory_status")
async def api_memory_status():
    return memory_monitor.get_current_memory_status()

@app.get("/api/memory_recommendations")
async def api_memory_recommendations():
    return {"recommendations": memory_monitor.get_memory_recommendations()}

@app.get("/api/install_model/{model_name}")
async def api_install_model(model_name: str):
    # Use the install_quantized_model.py logic or recommend manual install
    return {"message": f"To install: ollama pull {model_name}"}

if __name__ == "__main__":
    print("Starting Unified RAG Dashboard at http://localhost:8012")
    uvicorn.run(app, host="0.0.0.0", port=8012) 