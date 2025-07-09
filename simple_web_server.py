"""
Simple Web Server for Contract Clause Search
===========================================

A web interface for searching through contract clauses using FAISS embeddings.
This version doesn't require Ollama - just semantic search functionality.

Usage:
    python simple_web_server.py
    Then visit http://localhost:8000
"""

import json
import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ollama_server.ollama_qa import get_qa_chain
from pydantic import BaseModel


# === Configuration ===
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load FAISS Index ===
print("Loading FAISS index and embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, index_name="cuad_faiss", allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Load metadata
with open("embeddings/output/cuad_metadata.json", "r") as f:
    metadata = json.load(f)

print("System ready!")

# === FastAPI App ===
app = FastAPI(title="Contract Clause Search")
qa_chain = get_qa_chain()

# HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Contract Clause Search</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .search-box { width: 100%; padding: 10px; font-size: 16px; margin-bottom: 20px; }
        .search-button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        .result { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .clause-type { font-weight: bold; color: #007bff; }
        .contract-title { color: #666; font-size: 14px; }
        .text { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🔍 Contract Clause Search</h1>
    <p>Search through 13,823 contract clauses from the CUAD dataset using semantic search.</p>
    
    <form method="post">
        <input type="text" name="query" placeholder="Enter your search query..." class="search-box" value="{query}">
        <button type="submit" class="search-button">Search</button>
    </form>
    
    {results}
</body>
</html>
"""

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: QueryRequest):
    result = qa_chain(request.query)
    return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template.format(query="", results="")

@app.post("/", response_class=HTMLResponse)
async def search(query: str = Form(...)):
    if not query.strip():
        return html_template.format(query="", results="<p>Please enter a search query.</p>")
    
    print(f"Searching for: {query}")
    
    # Perform search
    docs = retriever.get_relevant_documents(query)
    
    # Format results
    results_html = f"<h2>Search Results for: '{query}'</h2>"
    results_html += f"<p>Found {len(docs)} relevant clauses:</p>"
    
    for i, doc in enumerate(docs, 1):
        results_html += f"""
        <div class="result">
            <div class="clause-type">Clause Type: {doc.metadata.get('clause_type', 'Unknown')[:100]}...</div>
            <div class="contract-title">Contract: {doc.metadata.get('title', 'Unknown')}</div>
            <div class="text"><strong>Text:</strong> {doc.page_content[:300]}...</div>
        </div>
        """
    
    return html_template.format(query=query, results=results_html)

if __name__ == "__main__":
    print("Starting web server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 