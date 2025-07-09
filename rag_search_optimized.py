"""
Optimized RAG Contract Clause Search
===================================

Enhanced version with better performance, caching, and timeout handling.
"""

import json
import os
import asyncio
from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx
from prompts import get_prompt_template
import time
from typing import List, Dict, Any

# === Configuration ===
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
METADATA_PATH = "embeddings/output/cuad_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3"
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

# === Optimized Ollama Client ===
async def query_ollama_optimized(prompt: str) -> str:
    """Optimized Ollama query with better timeout and error handling"""
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

def build_optimized_rag_prompt(question: str, context_docs: List[Dict[str, Any]], prompt_type: str = "legal") -> str:
    """Build an optimized RAG prompt with context truncation"""
    # Truncate context documents to improve performance
    truncated_docs = []
    total_length = 0
    max_context_length = 1500  # Limit context length in words
    
    for doc in context_docs:
        doc_text = f"Clause Type: {doc.get('clause_type', 'Unknown')}\nContract: {doc.get('title', 'Unknown')}\nText: {doc.get('text', 'Unknown')}"
        doc_words = doc_text.split()
        
        if total_length + len(doc_words) > max_context_length:
            # Truncate this document
            remaining_words = max_context_length - total_length
            if remaining_words > 50:  # Only add if we have meaningful content
                truncated_text = " ".join(doc_words[:remaining_words]) + "... [truncated]"
                truncated_docs.append(truncated_text)
            break
        else:
            truncated_docs.append(doc_text)
            total_length += len(doc_words)
    
    context_text = "\n\n".join(truncated_docs)
    
    # Use concise prompt for better performance
    if prompt_type == "legal":
        prompt_template = """You are a legal contract assistant. Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt_template = get_prompt_template(prompt_type)
    
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    
    print(f"Built prompt with {len(truncated_docs)} docs, {len(formatted_prompt.split())} words")
    return formatted_prompt

# === FastAPI App ===
app = FastAPI(title="Optimized RAG Contract Clause Search")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimized RAG Contract Clause Search</title>
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
            .loading {{ color: #666; font-style: italic; }}
            .performance {{ font-size: 12px; color: #888; margin-top: 10px; }}
        </style>
        <script>
            function showLoading() {{
                document.getElementById('loading').style.display = 'block';
                document.getElementById('rag-button').disabled = true;
            }}
        </script>
    </head>
    <body>
        <h1>🚀 Optimized RAG Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses with improved performance and timeout handling.</p>
        
        <form method="post" onsubmit="showLoading()">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box">
            <div style="margin-bottom: 15px;">
                <label for="prompt_type" style="font-weight: bold; margin-right: 10px;">Analysis Style:</label>
                <select name="prompt_type" id="prompt_type" style="padding: 5px; border-radius: 4px;">
                    <option value="legal">Fast Legal Analysis</option>
                    <option value="concise">Concise Summary</option>
                    <option value="analysis">Detailed Analysis</option>
                </select>
            </div>
            <button type="submit" name="mode" value="search" class="search-button">Search Only</button>
            <button type="submit" name="mode" value="rag" class="rag-button" id="rag-button">Ask AI (RAG)</button>
        </form>
        
        <div id="loading" style="display: none;" class="loading">
            <p>🤖 Generating AI response... This may take up to 2 minutes for complex queries.</p>
        </div>
    </body>
    </html>
    """
    return html

@app.post("/", response_class=HTMLResponse)
async def search_or_rag(query: str = Form(...), mode: str = Form("search"), page: int = Form(1), prompt_type: str = Form("legal")):
    start_time = time.time()
    
    if not query.strip():
        return await home()
    
    print(f"Processing: {query} (mode: {mode}, page: {page})")
    
    try:
        # Get query embedding
        embedding_start = time.time()
        query_embedding = embedding_model.encode([query])
        embedding_time = time.time() - embedding_start
        print(f"Embedding generated in {embedding_time:.2f}s")
        
        # Search FAISS index
        search_start = time.time()
        k = 20
        distances, indices = index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - search_start
        print(f"FAISS search completed in {search_time:.2f}s")
        
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
            rag_start = time.time()
            
            # Build optimized RAG prompt
            rag_context_docs = context_docs[:MAX_CONTEXT_DOCS]
            rag_prompt = build_optimized_rag_prompt(query, rag_context_docs, prompt_type)
            
            print(f"Starting RAG query with {len(rag_context_docs)} docs...")
            ai_response = await query_ollama_optimized(rag_prompt)
            
            rag_time = time.time() - rag_start
            total_time = time.time() - start_time
            
            ai_answer = f"""
            <div class="answer">
                <h3>🤖 AI Answer:</h3>
                <p>{ai_response}</p>
                <div class="performance">
                    <strong>Performance:</strong> RAG processing took {rag_time:.1f}s, total request took {total_time:.1f}s
                </div>
            </div>
            """
        
    except Exception as e:
        total_time = time.time() - start_time
        results_html = f"<p>Error during search after {total_time:.1f}s: {str(e)}</p>"
        ai_answer = ""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimized RAG Contract Clause Search</title>
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
            .performance {{ font-size: 12px; color: #888; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>🚀 Optimized RAG Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses with improved performance and timeout handling.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box" value="{query}">
            <div style="margin-bottom: 15px;">
                <label for="prompt_type" style="font-weight: bold; margin-right: 10px;">Analysis Style:</label>
                <select name="prompt_type" id="prompt_type" style="padding: 5px; border-radius: 4px;">
                    <option value="legal" {"selected" if prompt_type == "legal" else ""}>Fast Legal Analysis</option>
                    <option value="concise" {"selected" if prompt_type == "concise" else ""}>Concise Summary</option>
                    <option value="analysis" {"selected" if prompt_type == "analysis" else ""}>Detailed Analysis</option>
                </select>
            </div>
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
    print("🚀 Starting Optimized RAG server at http://localhost:8005")
    print("📊 Performance optimizations:")
    print(f"   - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"   - Max prompt length: {MAX_PROMPT_LENGTH} words")
    print(f"   - Request timeout: {REQUEST_TIMEOUT}s")
    print("   - Reduced model parameters for faster inference")
    uvicorn.run(app, host="0.0.0.0", port=8005) 