"""
RAG Contract Clause Search
=========================

Combines FAISS search with Ollama for intelligent answer generation.
"""

import json
import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx
import asyncio
from prompts import get_prompt_template

# === Configuration ===
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
METADATA_PATH = "embeddings/output/cuad_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3"  # Using Gemma3 model
MAX_CONTEXT_DOCS = 4  # Reduced from 8 to improve performance
MAX_PROMPT_LENGTH = 2000  # Limit prompt length in words
REQUEST_TIMEOUT = 120  # Reduced timeout to 2 minutes

# === Load FAISS Index and Model ===
print("Loading FAISS index and embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print(f"System ready! Loaded {len(metadata)} clauses.")

# === Ollama Client ===
async def query_ollama(prompt: str) -> str:
    """Query Ollama for text generation with improved timeout handling"""
    import time
    start_time = time.time()
    
    try:
        # Truncate prompt if too long
        words = prompt.split()
        if len(words) > MAX_PROMPT_LENGTH:
            prompt = " ".join(words[:MAX_PROMPT_LENGTH]) + "\n\n[Context truncated for performance]"
            print(f"Prompt truncated from {len(words)} to {MAX_PROMPT_LENGTH} words")
        
        print(f"Sending request to Ollama (model: {OLLAMA_MODEL}, ~{len(words)} words)...")
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower temperature for faster, more focused responses
                        "top_p": 0.8,
                        "num_predict": 500,  # Reduced from 1000 for faster responses
                        "top_k": 20
                    }
                }
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response generated")
                print(f"✅ Ollama response received in {elapsed:.1f}s ({len(response_text)} chars)")
                return response_text
            else:
                error_msg = f"Error: Ollama returned status {response.status_code}"
                print(f"❌ {error_msg} after {elapsed:.1f}s")
                return error_msg
                
    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        error_msg = f"Error: Request timed out after {elapsed:.1f}s. Try a simpler question or reduce context."
        print(f"⏰ {error_msg}")
        return error_msg
    except httpx.ConnectError:
        error_msg = "Error: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
        print(f"🔌 {error_msg}")
        return error_msg
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error connecting to Ollama after {elapsed:.1f}s: {str(e)}"
        print(f"💥 {error_msg}")
        return error_msg

def build_rag_prompt(question: str, context_docs: list, prompt_type: str = "legal") -> str:
    """Build a sophisticated RAG prompt using LangChain templates"""
    # Format context documents
    context_text = "\n\n".join([
        f"Clause Type: {doc.get('clause_type', 'Unknown')}\n"
        f"Contract: {doc.get('title', 'Unknown')}\n"
        f"Text: {doc.get('text', 'Unknown')}"
        for doc in context_docs
    ])
    
    # Get the appropriate prompt template
    prompt_template = get_prompt_template(prompt_type)
    
    # Format the prompt with context and question
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    
    return formatted_prompt

# === FastAPI App ===
app = FastAPI(title="RAG Contract Clause Search")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Contract Clause Search</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }}
            .rag-button {{ padding: 10px 20px; font-size: 16px; background: #28a745; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .contract-title {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; }}
            .answer {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0; }}
            .pagination {{ margin: 20px 0; text-align: center; }}
            .pagination-btn {{ padding: 8px 16px; margin: 0 5px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            .pagination-btn:hover {{ background: #0056b3; }}
            .page-info {{ margin: 0 15px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>🔍 RAG Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses from the CUAD dataset using semantic search and AI-powered answers.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box">
            <div style="margin-bottom: 15px;">
                <label for="prompt_type" style="font-weight: bold; margin-right: 10px;">Analysis Style:</label>
                <select name="prompt_type" id="prompt_type" style="padding: 5px; border-radius: 4px;">
                    <option value="legal">Detailed Legal Analysis</option>
                    <option value="concise">Concise Summary</option>
                    <option value="analysis">Clause Analysis</option>
                </select>
            </div>
            <button type="submit" name="mode" value="search" class="search-button">Search Only</button>
            <button type="submit" name="mode" value="rag" class="rag-button">Ask AI (RAG)</button>
        </form>
    </body>
    </html>
    """
    return html

@app.post("/", response_class=HTMLResponse)
async def search_or_rag(query: str = Form(...), mode: str = Form("search"), page: int = Form(1), prompt_type: str = Form("legal")):
    if not query.strip():
        return await home()
    
    print(f"Processing: {query} (mode: {mode}, page: {page})")
    
    try:
        # Get query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search FAISS index - get more results for pagination
        k = 20  # Get more results initially
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        # Get context documents
        context_docs = []
        for idx in indices[0]:
            if idx < len(metadata):
                context_docs.append(metadata[idx])
        
        # Pagination logic
        results_per_page = 5
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page
        paginated_docs = context_docs[start_idx:end_idx]
        total_results = len(context_docs)
        total_pages = (total_results + results_per_page - 1) // results_per_page
        
        # Format search results
        results_html = f"<h2>Search Results for: '{query}'</h2>"
        results_html += f"<p>Found {total_results} relevant clauses (showing {len(paginated_docs)} of {total_results}):</p>"
        
        for i, doc in enumerate(paginated_docs, start_idx + 1):
            clause_type = doc.get('clause_type', 'Unknown')[:100] + "..." if len(doc.get('clause_type', '')) > 100 else doc.get('clause_type', 'Unknown')
            title = doc.get('title', 'Unknown')
            source = doc.get('source', 'Unknown')
            
            results_html += f"""
            <div class="result">
                <div class="clause-type">Clause Type: {clause_type}</div>
                <div class="contract-title">Contract: {title}</div>
                <div class="text"><strong>Source:</strong> {source}</div>
            </div>
            """
        
        # Add pagination controls
        if total_pages > 1:
            results_html += "<div class='pagination'>"
            if page > 1:
                results_html += f"""
                <form method="post" style="display: inline;">
                    <input type="hidden" name="query" value="{query}">
                    <input type="hidden" name="mode" value="{mode}">
                    <input type="hidden" name="prompt_type" value="{prompt_type}">
                    <input type="hidden" name="page" value="{page - 1}">
                    <button type="submit" class="pagination-btn">← Previous</button>
                </form>
                """
            
            results_html += f"<span class='page-info'>Page {page} of {total_pages}</span>"
            
            if page < total_pages:
                results_html += f"""
                <form method="post" style="display: inline;">
                    <input type="hidden" name="query" value="{query}">
                    <input type="hidden" name="mode" value="{mode}">
                    <input type="hidden" name="prompt_type" value="{prompt_type}">
                    <input type="hidden" name="page" value="{page + 1}">
                    <button type="submit" class="pagination-btn">Next →</button>
                </form>
                """
            results_html += "</div>"
        
        # Generate AI answer if RAG mode
        ai_answer = ""
        if mode == "rag":
            ai_answer = f"""
            <div class="answer">
                <h3>🤖 AI Answer:</h3>
                <p>Generating answer...</p>
            </div>
            """
            
            # Build RAG prompt and query Ollama (limit context size)
            rag_context_docs = context_docs[:MAX_CONTEXT_DOCS]  # Limit to first 8 most relevant docs
            rag_prompt = build_rag_prompt(query, rag_context_docs, prompt_type)
            
            # Log context size for monitoring
            prompt_tokens = len(rag_prompt.split())  # Rough token estimation
            print(f"RAG Context: {len(rag_context_docs)} docs, ~{prompt_tokens} words")
            
            ai_response = await query_ollama(rag_prompt)
            
            ai_answer = f"""
            <div class="answer">
                <h3>🤖 AI Answer:</h3>
                <p>{ai_response}</p>
            </div>
            """
        
    except Exception as e:
        results_html = f"<p>Error during search: {str(e)}</p>"
        ai_answer = ""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Contract Clause Search</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }}
            .rag-button {{ padding: 10px 20px; font-size: 16px; background: #28a745; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .contract-title {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; }}
            .answer {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>🔍 RAG Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses from the CUAD dataset using semantic search and AI-powered answers.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box" value="{query}">
            <button type="submit" name="mode" value="search" class="search-button">Search Only</button>
            <button type="submit" name="mode" value="rag" class="rag-button">Ask AI (RAG)</button>
        </form>
        
        {ai_answer}
        {results_html}
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    print("🚀 Starting Optimized RAG server at http://localhost:8006")
    print("📊 Performance optimizations:")
    print(f"   - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"   - Max prompt length: {MAX_PROMPT_LENGTH} words")
    print(f"   - Request timeout: {REQUEST_TIMEOUT}s")
    print("   - Reduced model parameters for faster inference")
    uvicorn.run(app, host="0.0.0.0", port=8006) 