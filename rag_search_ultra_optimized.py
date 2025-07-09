"""
Ultra-Optimized CUAD RAG Contract Clause Search
==============================================

Minimal context version to prevent timeouts while maintaining quality.
"""

import json
import os
import asyncio
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import httpx
import time
from typing import List, Dict, Any

# === Configuration ===
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
CUAD_DATA_PATH = "cuad_clause_library.jsonl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3"
MAX_CONTEXT_DOCS = 1  # Ultra-minimal: just 1 document
MAX_PROMPT_LENGTH = 500  # Very short prompts
REQUEST_TIMEOUT = 60  # Reduced timeout

# === Load Data ===
print("Loading CUAD clause library...")
clauses = []
with open(CUAD_DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            clauses.append(json.loads(line))

print(f"✅ Loaded {len(clauses)} clauses with actual text content")

# === Load FAISS Index and Model ===
print("Loading FAISS index and embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

print(f"System ready! Loaded {len(clauses)} clauses with text content.")

# === Ultra-Optimized Ollama Client ===
async def query_ollama_ultra_optimized(prompt: str) -> str:
    """Ultra-optimized Ollama query with minimal context"""
    start_time = time.time()
    
    try:
        # Truncate prompt if too long
        words = prompt.split()
        if len(words) > MAX_PROMPT_LENGTH:
            prompt = " ".join(words[:MAX_PROMPT_LENGTH]) + "\n\n[Context truncated for ultra-optimization]"
            print(f"Prompt truncated from {len(words)} to {MAX_PROMPT_LENGTH} words")
        
        print(f"Sending ultra-optimized request to Ollama (model: {OLLAMA_MODEL}, ~{len(words)} words)...")
        
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.5,
                        "num_predict": 200,  # Very short responses
                        "top_k": 5
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
                error_text = response.text
                error_msg = f"Error: Ollama returned status {response.status_code}: {error_text}"
                print(f"❌ {error_msg} after {elapsed:.1f}s")
                return error_msg
                
    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        error_msg = f"Error: Request timed out after {elapsed:.1f}s. Context too large."
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

def build_ultra_optimized_prompt(question: str, context_docs: List[Dict[str, Any]]) -> str:
    """Build an ultra-optimized RAG prompt with minimal context"""
    if not context_docs:
        return f"Question: {question}\n\nAnswer: I don't have enough context to answer this question."
    
    # Use only the first (most relevant) document
    doc = context_docs[0]
    clause_type = doc.get('clause_type', 'Unknown')
    text = doc.get('text', 'Unknown')
    source = doc.get('source', 'Unknown')
    
    # Truncate text aggressively
    if len(text) > 300:
        text = text[:300] + "... [truncated]"
    
    # Very simple prompt format
    prompt = f"""Based on this contract clause, answer the question.

Clause Type: {clause_type}
Source: {source}
Text: {text}

Question: {question}

Answer:"""
    
    print(f"Built ultra-optimized prompt: {len(prompt.split())} words")
    return prompt

# === FastAPI App ===
app = FastAPI(title="Ultra-Optimized CUAD RAG")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultra-Optimized CUAD RAG</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }}
            .rag-button {{ padding: 10px 20px; font-size: 16px; background: #28a745; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .source {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; white-space: pre-wrap; }}
            .answer {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0; }}
            .performance {{ font-size: 12px; color: #888; margin-top: 10px; }}
            .ultra-info {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>⚡ Ultra-Optimized CUAD RAG</h1>
        <div class="ultra-info">
            <strong>Ultra Optimizations:</strong> Single context doc, 500-word prompts, 60s timeout, minimal parameters
        </div>
        <p>Search through {len(clauses)} contract clauses with ultra-fast processing.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box">
            <button type="submit" name="mode" value="search" class="search-button">Search Only</button>
            <button type="submit" name="mode" value="rag" class="rag-button">Ask AI (RAG)</button>
        </form>
    </body>
    </html>
    """
    return html

@app.post("/", response_class=HTMLResponse)
async def search_or_rag(query: str = Form(...), mode: str = Form("search"), page: int = Form(1)):
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
        k = 5  # Very small search
        distances, indices = index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - search_start
        print(f"FAISS search completed in {search_time:.2f}s")
        
        # Get context documents from CUAD data
        context_docs = []
        for idx in indices[0]:
            if idx < len(clauses):
                context_docs.append(clauses[idx])
        
        # Pagination logic
        results_per_page = 2  # Very small results
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page
        paginated_docs = context_docs[start_idx:end_idx]
        total_results = len(context_docs)
        total_pages = (total_results + results_per_page - 1) // results_per_page
        
        # Format search results
        results_html = f"<h2>Search Results for: '{query}'</h2>"
        results_html += f"<p>Found {total_results} relevant clauses (showing {len(paginated_docs)} of {total_results}):</p>"
        
        for i, doc in enumerate(paginated_docs, start_idx + 1):
            clause_type = doc.get('clause_type', 'Unknown')[:60] + "..." if len(doc.get('clause_type', '')) > 60 else doc.get('clause_type', 'Unknown')
            source = doc.get('source', 'Unknown')
            text = doc.get('text', 'Unknown')[:100] + "..." if len(doc.get('text', '')) > 100 else doc.get('text', 'Unknown')
            
            results_html += f"""
            <div class="result">
                <div class="clause-type">Clause Type: {clause_type}</div>
                <div class="source">Source: {source}</div>
                <div class="text"><strong>Text:</strong> {text}</div>
            </div>
            """
        
        # Generate AI answer if RAG mode
        ai_answer = ""
        if mode == "rag":
            rag_start = time.time()
            
            # Build ultra-optimized RAG prompt
            rag_context_docs = context_docs[:MAX_CONTEXT_DOCS]
            rag_prompt = build_ultra_optimized_prompt(query, rag_context_docs)
            
            print(f"Starting ultra-optimized RAG query with {len(rag_context_docs)} docs...")
            ai_response = await query_ollama_ultra_optimized(rag_prompt)
            
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
        <title>Ultra-Optimized CUAD RAG</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }}
            .rag-button {{ padding: 10px 20px; font-size: 16px; background: #28a745; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .source {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; white-space: pre-wrap; }}
            .answer {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0; }}
            .performance {{ font-size: 12px; color: #888; margin-top: 10px; }}
            .ultra-info {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>⚡ Ultra-Optimized CUAD RAG</h1>
        <div class="ultra-info">
            <strong>Ultra Optimizations:</strong> Single context doc, 500-word prompts, 60s timeout, minimal parameters
        </div>
        <p>Search through {len(clauses)} contract clauses with ultra-fast processing.</p>
        
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
    print("⚡ Starting Ultra-Optimized CUAD RAG server at http://localhost:8010")
    print("📊 Ultra optimizations:")
    print(f"   - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"   - Max prompt length: {MAX_PROMPT_LENGTH} words")
    print(f"   - Request timeout: {REQUEST_TIMEOUT}s")
    print(f"   - Ultra-minimal model parameters")
    uvicorn.run(app, host="0.0.0.0", port=8010) 