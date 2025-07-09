"""
Test context quality and improve search results
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load components
print("🔍 Loading RAG components...")
FAISS_INDEX_PATH = "embeddings/output/cuad_faiss.index"
METADATA_PATH = "embeddings/output/cuad_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Test different queries
test_queries = [
    "What are non-compete clauses?",
    "non-compete agreement",
    "employee non-compete",
    "restrictive covenant",
    "competition restriction"
]

for query in test_queries:
    print(f"\n🔍 Testing query: '{query}'")
    
    # Get embedding
    query_embedding = embedding_model.encode([query])
    
    # Search with more results
    k = 10  # Get more results to see what's available
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    print(f"Top {len(indices[0])} results:")
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            doc = metadata[idx]
            clause_type = doc.get('clause_type', 'Unknown')
            title = doc.get('title', 'Unknown')
            text = doc.get('text', 'Unknown')[:100] + "..." if len(doc.get('text', '')) > 100 else doc.get('text', 'Unknown')
            
            print(f"  {i+1}. Distance: {distances[0][i]:.3f}")
            print(f"     Type: {clause_type}")
            print(f"     Title: {title}")
            print(f"     Text: {text}")
            print()

# Search for specific non-compete related terms
print("\n🔍 Searching for specific non-compete terms...")
specific_terms = ["non-compete", "noncompete", "restrictive covenant", "competition"]

for term in specific_terms:
    print(f"\nSearching for: '{term}'")
    
    # Get embedding
    term_embedding = embedding_model.encode([term])
    
    # Search
    k = 5
    distances, indices = index.search(term_embedding.astype('float32'), k)
    
    print(f"Top results for '{term}':")
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            doc = metadata[idx]
            clause_type = doc.get('clause_type', 'Unknown')
            title = doc.get('title', 'Unknown')
            
            # Check if the term appears in the text
            text = doc.get('text', '').lower()
            term_found = term.lower() in text
            
            print(f"  {i+1}. {clause_type} - {title} {'✅' if term_found else '❌'}")

print("\n📊 Summary:")
print("If you're getting 'Unknown' answers, it might be because:")
print("1. The retrieved context doesn't contain specific non-compete information")
print("2. The model needs more specific context about non-compete clauses")
print("3. The search terms need to be more specific")

print("\n💡 Suggestions:")
print("1. Try more specific queries like 'employee non-compete restrictions'")
print("2. Use the 'Search Only' mode first to see what documents are available")
print("3. Consider expanding the context window if relevant documents are found") 