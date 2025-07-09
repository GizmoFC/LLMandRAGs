"""
RAG Contract Clause Search using CUAD Clause Library
==================================================

Uses the actual clause text from cuad_clause_library.jsonl for better RAG performance.
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
CUAD_DATA_PATH = "cuad_clause_library.jsonl"  # Use the actual clause library
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3"
MAX_CONTEXT_DOCS = 4
MAX_PROMPT_LENGTH = 2000
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
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "num_predict": 500,
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
    """Build an optimized RAG prompt with actual clause text"""
    # Truncate context documents to improve performance
    truncated_docs = []
    total_length = 0
    max_context_length = 1500
    
    for doc in context_docs:
        # Use actual text content from CUAD data
        clause_type = doc.get('clause_type', 'Unknown')
        text = doc.get('text', 'Unknown')
        source = doc.get('source', 'Unknown')
        
        doc_text = f"Clause Type: {clause_type}\nSource: {source}\nText: {text}"
        doc_words = doc_text.split()
        
        if total_length + len(doc_words) > max_context_length:
            remaining_words = max_context_length - total_length
            if remaining_words > 50:
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
        prompt_template = """You are a precise legal contract assistant. Analyze the following contract clauses and answer the question with specific details.

**Rules:**
- Answer ONLY based on the provided context
- Be specific about parties, obligations, timeframes, and conditions
- If context is insufficient, state what's missing
- Do not speculate or add external information

**Contract Context:**
{context}

**Question:**
{question}

**Answer:**"""
    
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    
    print(f"Built prompt with {len(truncated_docs)} docs, {len(formatted_prompt.split())} words")
    return formatted_prompt

# === FastAPI App ===
app = FastAPI(title="CUAD RAG Contract Clause Search")

@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CUAD RAG Contract Clause Search</title>
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
        </style>
    </head>
    <body>
        <h1>📋 CUAD RAG Contract Clause Search</h1>
        <p>Search through {len(clauses)} contract clauses with actual text content from the CUAD dataset.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box">
            <div style="margin-bottom: 15px;">
                <label for="prompt_type" style="font-weight: bold; margin-right: 10px;">Analysis Style:</label>
                <select name="prompt_type" id="prompt_type" style="padding: 5px; border-radius: 4px;">
                    <option value="legal">Fast Legal Analysis</option>
                    <option value="concise">Concise Summary</option>
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
        
        # Get context documents from CUAD data
        context_docs = []
        for idx in indices[0]:
            if idx < len(clauses):
                context_docs.append(clauses[idx])
        
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
            source = doc.get('source', 'Unknown')
            text = doc.get('text', 'Unknown')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', 'Unknown')
            
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
        <title>CUAD RAG Contract Clause Search</title>
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
        </style>
    </head>
    <body>
        <h1>📋 CUAD RAG Contract Clause Search</h1>
        <p>Search through {len(clauses)} contract clauses with actual text content from the CUAD dataset.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Ask a question about contract clauses..." class="search-box" value="{query}">
            <div style="margin-bottom: 15px;">
                <label for="prompt_type" style="font-weight: bold; margin-right: 10px;">Analysis Style:</label>
                <select name="prompt_type" id="prompt_type" style="padding: 5px; border-radius: 4px;">
                    <option value="legal" {"selected" if prompt_type == "legal" else ""}>Fast Legal Analysis</option>
                    <option value="concise" {"selected" if prompt_type == "concise" else ""}>Concise Summary</option>
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
    print("📋 Starting CUAD RAG server at http://localhost:8007")
    print("📊 Using actual clause text from cuad_clause_library.jsonl")
    print(f"   - Max context docs: {MAX_CONTEXT_DOCS}")
    print(f"   - Max prompt length: {MAX_PROMPT_LENGTH} words")
    print(f"   - Request timeout: {REQUEST_TIMEOUT}s")
    uvicorn.run(app, host="0.0.0.0", port=8007) 