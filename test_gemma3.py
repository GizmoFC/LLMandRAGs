import httpx
import asyncio
import time

async def test_gemma3():
    """Test Gemma3 with a RAG-like prompt"""
    
    prompt = """You are a legal contract assistant. Use the following contract clauses to answer the question as accurately as possible.

Context from contract clauses:
Clause Type: Termination
Contract: Sample Agreement
Text: Either party may terminate this agreement with 30 days written notice.

Question: What are the termination conditions?

Please provide a clear, accurate answer based on the contract clauses above. If the context doesn't contain enough information to answer the question, say so.

Answer:"""

    print("Testing Gemma3 with RAG-like prompt...")
    print(f"Prompt length: {len(prompt)} characters")
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # 2 minute timeout
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Response time: {duration:.2f} seconds")
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result.get('response', 'No response')}")
                print(f"Done: {result.get('done', False)}")
                print(f"Total duration: {result.get('total_duration', 0) / 1e9:.2f} seconds")
            else:
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_gemma3()) 