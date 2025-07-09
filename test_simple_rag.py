"""
Simple RAG test to isolate timeout issues
"""

import json
import asyncio
import httpx
import time
from sentence_transformers import SentenceTransformer
import faiss

# Load data
print("Loading data...")
with open("cuad_clause_library.jsonl", 'r', encoding='utf-8') as f:
    clauses = [json.loads(line) for line in f if line.strip()]

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/output/cuad_faiss.index")

print(f"Loaded {len(clauses)} clauses")

async def test_simple_rag():
    query = "What are confidentiality clauses?"
    
    print(f"\n🔍 Testing query: '{query}'")
    
    # Get embedding and search
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), 1)  # Just 1 result
    
    if indices[0][0] < len(clauses):
        doc = clauses[indices[0][0]]
        clause_type = doc.get('clause_type', 'Unknown')
        text = doc.get('text', 'Unknown')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', 'Unknown')
        
        print(f"Found relevant clause: {clause_type}")
        print(f"Text preview: {text[:100]}...")
        
        # Build minimal prompt
        prompt = f"""Based on this contract clause, answer the question.

Clause Type: {clause_type}
Text: {text}

Question: {query}

Answer:"""
        
        print(f"\n📝 Prompt length: {len(prompt.split())} words")
        print(f"Prompt preview: {prompt[:200]}...")
        
        # Test Ollama
        print(f"\n🤖 Testing Ollama with minimal prompt...")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma3",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 150
                        }
                    }
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "No response")
                    print(f"✅ Success! Response in {elapsed:.1f}s")
                    print(f"Answer: {response_text[:200]}...")
                    return True
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            print(f"⏰ Timeout after {elapsed:.1f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"💥 Error after {elapsed:.1f}s: {str(e)}")
            return False
    else:
        print("❌ No relevant clause found")
        return False

if __name__ == "__main__":
    print("🧪 Simple RAG Test")
    print("=" * 40)
    
    success = asyncio.run(test_simple_rag())
    
    if success:
        print("\n🎉 Simple RAG test PASSED!")
        print("The issue might be in the web server or larger prompts.")
    else:
        print("\n❌ Simple RAG test FAILED!")
        print("The issue is with the basic RAG functionality.") 