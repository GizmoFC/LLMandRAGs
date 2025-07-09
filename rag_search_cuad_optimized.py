"""
Memory-Optimized CUAD RAG Contract Clause Search
===============================================

Optimized for lower memory usage while maintaining quality.
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
MAX_CONTEXT_DOCS = 2  # Reduced from 4 to save memory
MAX_PROMPT_LENGTH = 1000  # Reduced from 2000 to save memory
REQUEST_TIMEOUT = 120

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

# === Memory-Optimized Ollama Client ===
async def query_ollama_optimized(prompt: str) -> str:
    """Memory-optimized Ollama query"""
    start_time = time.time()
    
    try:
        # Truncate prompt if too long
        words = prompt.split()
        if len(words) > MAX_PROMPT_LENGTH:
            prompt = " ".join(words[:MAX_PROMPT_LENGTH]) + "\n\n[Context truncated for memory optimization]"
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
                        "temperature": 0.1,  # Lower temperature for more focused responses
                        "top_p": 0.7,
                        "num_predict": 300,  # Reduced for memory efficiency
                        "top_k": 10
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

def build_memory_optimized_prompt(question: str, context_docs: List[Dict[str, Any]], prompt_type: str = "legal") -> str:
    """Build a memory-optimized RAG prompt"""
    # Truncate context documents to improve memory usage
    truncated_docs = []
    total_length = 0
    max_context_length = 800  # Reduced from 1500 to save memory
    
    for doc in context_docs:
        # Use actual text content from CUAD data
        clause_type = doc.get('clause_type', 'Unknown')
        text = doc.get('text', 'Unknown')
        source = doc.get('source', 'Unknown')
        
        # Truncate text to save memory
        if len(text) > 500:
            text = text[:500] + "... [truncated]"
        
        doc_text = f"Clause Type: {clause_type}\nSource: {source}\nText: {text}"
        doc_words = doc_text.split()
        
        if total_length + len(doc_words) > max_context_length:
            remaining_words = max_context_length - total_length
            if remaining_words > 30:  # Reduced threshold
                truncated_text = " ".join(doc_words[:remaining_words]) + "... [truncated]"
                truncated_docs.append(truncated_text)
            break
        else:
            truncated_docs.append(doc_text)
            total_length += len(doc_words)
    
    context_text = "\n\n".join(truncated_docs)
    
    # Use very concise prompt for memory efficiency
    prompt_template = """You are a legal contract assistant. Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
    
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    
    print(f"Built memory-optimized prompt with {len(truncated_docs)} docs, {len(formatted_prompt.split())} words")
    return formatted_prompt

# === FastAPI App ===
app = FastAPI(title="Memory-Optimized CUAD RAG")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory-Optimized CUAD RAG</title>
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
            .memory-info {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>💾 Memory-Optimized CUAD RAG</h1>
        <div class="memory-info">
            <strong>Memory Optimizations:</strong> Reduced context (2 docs), shorter prompts (1000 words), optimized model parameters
        </div>
        <p>Search through {len(clauses)} contract clauses with memory-efficient processing.</p>
        
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
        k = 10  # Reduced from 20 to save memory
        distances, indices = index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - search_start
        print(f"FAISS search completed in {search_time:.2f}s")
        
        # Get context documents from CUAD data
        context_docs = []
        for idx in indices[0]:
            if idx < len(clauses):
                context_docs.append(clauses[idx])
        
        # Pagination logic
        results_per_page = 3  # Reduced from 5 to save memory
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page
        paginated_docs = context_docs[start_idx:end_idx]
        total_results = len(context_docs)
        total_pages = (total_results + results_per_page - 1) // results_per_page
        
        # Format search results
        results_html = f"<h2>Search Results for: '{query}'</h2>"
        results_html += f"<p>Found {total_results} relevant clauses (showing {len(paginated_docs)} of {total_results}):</p>"
        
        for i, doc in enumerate(paginated_docs, start_idx + 1):
            clause_type = doc.get('clause_type', 'Unknown')[:80] + "..." if len(doc.get('clause_type', '')) > 80 else doc.get('clause_type', 'Unknown')
            source = doc.get('source', 'Unknown')
            text = doc.get('text', 'Unknown')[:150] + "..." if len(doc.get('text', '')) > 150 else doc.get('text', 'Unknown')
            
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
            
            # Build memory-optimized RAG prompt
            rag_context_docs = context_docs[:MAX_CONTEXT_DOCS]
            rag_prompt = build_memory_optimized_prompt(query, rag_context_docs)
            
            print(f"Starting memory-optimized RAG query with {len(rag_context_docs)} docs...")
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
        <title>Memory-Optimized CUAD RAG</title>
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
            .memory-info {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>💾 Memory-Optimized CUAD RAG</h1>
        <div class="memory-info">
            <strong>Memory Optimizations:</strong> Reduced context (2 docs), shorter prompts (1000 words), optimized model parameters
        </div>
        <p>Search through {len(clauses)} contract clauses with memory-efficient processing.</p>
        
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
    print("💾 Starting Memory-Optimized CUAD RAG server at http://localhost:8008")
    print("📊 Memory optimizations:")
    print(f"   - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"   - Max prompt length: {MAX_PROMPT_LENGTH} words")
    print(f"   - Reduced model parameters for memory efficiency")
    uvicorn.run(app, host="0.0.0.0", port=8008) 