"""
Check available Ollama models and test memory usage
"""

import httpx
import asyncio

async def check_models():
    print("🔍 Checking available Ollama models...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get available models
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Found {len(models.get('models', []))} models:")
                for model in models.get('models', []):
                    print(f"  - {model.get('name', 'Unknown')} ({model.get('size', 'Unknown size')})")
            else:
                print(f"❌ Error getting models: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")

async def test_model(model_name):
    print(f"\n🧪 Testing model: {model_name}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
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
                print(f"✅ {model_name} working: {result.get('response', 'No response')[:100]}...")
                return True
            else:
                error_text = response.text
                print(f"❌ {model_name} error ({response.status_code}): {error_text}")
                return False
                
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return False

async def main():
    print("🔍 Ollama Model Checker")
    print("=" * 40)
    
    # Check available models
    await check_models()
    
    # Test common smaller models
    small_models = [
        "llama2:7b",
        "llama2:7b-chat", 
        "mistral:7b",
        "mistral:7b-instruct",
        "phi:2.7b",
        "phi:2.7b-chat",
        "gemma2:2b",
        "gemma2:2b-it"
    ]
    
    print(f"\n🧪 Testing smaller models...")
    working_models = []
    
    for model in small_models:
        if await test_model(model):
            working_models.append(model)
    
    print(f"\n📊 Summary:")
    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
        print(f"💡 Recommended: {working_models[0]} (first working model)")
    else:
        print("❌ No working models found")
        print("💡 Try installing a smaller model: ollama pull llama2:7b")

if __name__ == "__main__":
    asyncio.run(main()) 