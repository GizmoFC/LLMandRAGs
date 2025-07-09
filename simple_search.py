"""
Simple Contract Clause Search
============================

Direct FAISS search without LangChain wrapper.
"""

import json
import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import uvicorn
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama_server.ollama_qa import get_qa_chain
from pydantic import BaseModel

qa_chain = get_qa_chain()

# === Configuration ===
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
METADATA_PATH = "embeddings/output/cuad_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load FAISS Index and Model ===
print("Loading FAISS index and embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print(f"System ready! Loaded {len(metadata)} clauses.")

# === FastAPI App ===
app = FastAPI(title="Contract Clause Search")

class QueryRequest(BaseModel):
    query: str

@app.get("/rag", response_class=HTMLResponse)
async def rag_form():
    return """
    <html>
    <head>
        <title>RAG Q&A</title>
    </head>
    <body>
        <h1>📄 Ask a Legal Question (RAG)</h1>
        <form method="post">
            <input type="text" name="query" placeholder="Ask a legal question..." class="search-box">
            <button type="submit" class="search-button">Ask</button>
        </form>
    </body>
    </html>
    """

@app.post("/rag", response_class=HTMLResponse)
async def rag_answer(request: Request):
    form = await request.form()
    query = form.get("query", "").strip()

    if not query:
        return await rag_form()

    result = qa_chain(query)
    answer = result["result"]
    sources = result["source_documents"]

    sources_html = "".join(
        f"<li>{doc.metadata.get('source', 'Unknown')}</li>" for doc in sources
    )

    html = f"""
    <html>
    <head><title>RAG Q&A</title></head>
    <body>
        <h1>📄 Legal Q&A (RAG)</h1>
        <form method="post">
            <input type="text" name="query" value="{query}" class="search-box">
            <button type="submit" class="search-button">Ask</button>
        </form>

        <h2>Answer:</h2>
        <p>{answer}</p>

        <h3>Sources:</h3>
        <ul>{sources_html}</ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
async def home():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Contract Clause Search</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .contract-title {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🔍 Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses from the CUAD dataset using semantic search.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Enter your search query..." class="search-box">
            <button type="submit" class="search-button">Search</button>
        </form>
    </body>
    </html>
    """
    return html

@app.post("/", response_class=HTMLResponse)
async def search(query: str = Form(...)):
    if not query.strip():
        return await home()
    
    print(f"Searching for: {query}")
    
    try:
        # Get query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search FAISS index
        k = 5  # Number of results
        distances, indices = index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results_html = f"<h2>Search Results for: '{query}'</h2>"
        results_html += f"<p>Found {len(indices[0])} relevant clauses:</p>"
        
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                meta = metadata[idx]
                clause_type = meta.get('clause_type', 'Unknown')[:100] + "..." if len(meta.get('clause_type', '')) > 100 else meta.get('clause_type', 'Unknown')
                title = meta.get('title', 'Unknown')
                source = meta.get('source', 'Unknown')
                
                results_html += f"""
                <div class="result">
                    <div class="clause-type">Clause Type: {clause_type}</div>
                    <div class="contract-title">Contract: {title}</div>
                    <div class="text"><strong>Source:</strong> {source}</div>
                </div>
                """
    except Exception as e:
        results_html = f"<p>Error during search: {str(e)}</p>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Contract Clause Search</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }}
            .search-button {{ padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .clause-type {{ font-weight: bold; color: #007bff; }}
            .contract-title {{ color: #666; font-size: 14px; }}
            .text {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🔍 Contract Clause Search</h1>
        <p>Search through {len(metadata)} contract clauses from the CUAD dataset using semantic search.</p>
        
        <form method="post">
            <input type="text" name="query" placeholder="Enter your search query..." class="search-box" value="{query}">
            <button type="submit" class="search-button">Search</button>
        </form>
        
        {results_html}
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    print("Starting web server at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001) 