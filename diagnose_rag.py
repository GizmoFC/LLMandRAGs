"""
Diagnostic script for RAG system
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import httpx
import time

# Load the same components as the RAG system
print("🔍 Loading RAG components...")

# Load FAISS index
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
METADATA_PATH = "embeddings/output/cuad_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading metadata...")
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"✅ Loaded {len(metadata)} clauses")

# Test query
test_query = "What are non-compete clauses?"
print(f"\n🔍 Testing query: '{test_query}'")

# Get embedding
print("Generating query embedding...")
query_embedding = embedding_model.encode([test_query])
print(f"✅ Embedding shape: {query_embedding.shape}")

# Search FAISS
print("Searching FAISS index...")
k = 4  # Use same as optimized version
distances, indices = index.search(query_embedding.astype('float32'), k)
print(f"✅ Found {len(indices[0])} results")

# Get context documents
context_docs = []
for i, idx in enumerate(indices[0]):
    if idx < len(metadata):
        doc = metadata[idx]
        context_docs.append(doc)
        print(f"  {i+1}. {doc.get('clause_type', 'Unknown')} - {doc.get('title', 'Unknown')}")

print(f"\n📄 Context documents: {len(context_docs)}")

# Test Ollama connection
async def test_ollama():
    print("\n🤖 Testing Ollama connection...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3",
                    "prompt": "Hello, this is a test.",
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 50
                    }
                }
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Ollama working: {result.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {str(e)}")
        return False

# Test RAG prompt building
def test_prompt_building():
    print("\n📝 Testing prompt building...")
    
    # Build context text
    context_text = "\n\n".join([
        f"Clause Type: {doc.get('clause_type', 'Unknown')}\n"
        f"Contract: {doc.get('title', 'Unknown')}\n"
        f"Text: {doc.get('text', 'Unknown')}"
        for doc in context_docs
    ])
    
    # Build prompt
    prompt = f"""You are a legal contract assistant. Answer based ONLY on the provided context.

Context:
{context_text}

Question: {test_query}

Answer:"""
    
    print(f"✅ Prompt built: {len(prompt.split())} words")
    print(f"📄 Prompt preview: {prompt[:200]}...")
    return prompt

# Run tests
if __name__ == "__main__":
    print("🧪 Running RAG diagnostics...")
    
    # Test prompt building
    prompt = test_prompt_building()
    
    # Test Ollama
    ollama_working = asyncio.run(test_ollama())
    
    print(f"\n📊 Diagnostic Summary:")
    print(f"  ✅ FAISS index: {len(metadata)} clauses loaded")
    print(f"  ✅ Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  ✅ Context docs: {len(context_docs)} found")
    print(f"  ✅ Prompt length: {len(prompt.split())} words")
    print(f"  {'✅' if ollama_working else '❌'} Ollama connection: {'Working' if ollama_working else 'Failed'}")
    
    if ollama_working and len(context_docs) > 0:
        print("\n🎉 RAG system appears to be working correctly!")
        print("The 'Unknown' answer might be due to:")
        print("  1. Context not being relevant to the query")
        print("  2. Model not finding specific information in the context")
        print("  3. Prompt format issues")
    else:
        print("\n⚠️  Issues detected that need to be fixed.") 