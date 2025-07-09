"""
Test script for optimized RAG functionality
"""

import asyncio
import time
from rag_search_optimized import query_ollama_optimized, build_optimized_rag_prompt

async def test_optimized_rag():
    print("🧪 Testing Optimized RAG System...")
    
    # Test with a simple query
    test_query = "What are non-compete clauses?"
    test_context = [
        {
            "clause_type": "Non-Compete Agreement",
            "title": "Employment Contract",
            "text": "Employee agrees not to compete with the company for 12 months after termination in the same industry within 50 miles of the company location."
        },
        {
            "clause_type": "Confidentiality",
            "title": "Employment Contract", 
            "text": "Employee shall maintain confidentiality of all company trade secrets and proprietary information."
        }
    ]
    
    print(f"📝 Building optimized prompt for: '{test_query}'")
    start_time = time.time()
    
    # Build optimized prompt
    prompt = build_optimized_rag_prompt(test_query, test_context, "legal")
    build_time = time.time() - start_time
    
    print(f"✅ Prompt built in {build_time:.2f}s")
    print(f"📊 Prompt length: {len(prompt.split())} words")
    print(f"📄 Prompt preview: {prompt[:200]}...")
    
    # Test Ollama query
    print(f"\n🤖 Testing Ollama query...")
    ollama_start = time.time()
    
    try:
        response = await query_ollama_optimized(prompt)
        ollama_time = time.time() - ollama_start
        
        print(f"✅ Ollama response received in {ollama_time:.1f}s")
        print(f"📝 Response: {response[:300]}...")
        
        if ollama_time < 60:
            print("🎉 Performance test PASSED - Response under 60 seconds!")
        else:
            print("⚠️  Performance test WARNING - Response took over 60 seconds")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_optimized_rag()) 