"""
GPU-Optimized RAG System
Enhanced version for GPU deployment with larger models and better performance
"""

import json
import time
import torch
import requests
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DEFAULT_MODEL = "mistral:7b-instruct"
OLLAMA_URL = "http://localhost:11434"

class GPURAGSystem:
    def __init__(self):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # GPU-optimized configuration
        self.config = {
            "max_context_docs": 8,      # Can handle more with GPU
            "max_prompt_length": 4000,  # Larger prompts
            "timeout": 60,              # Faster with GPU
            "temperature": 0.3,
            "num_predict": 1024,        # Longer responses
            "top_k": 20,                # More candidates
            "model": "llama3.1:8b-instruct-q4_K_M"  # GPU-optimized model
        }
        
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_model = None
        self.faiss_index = None
        self.clauses = []
        self.clause_texts = []
        
        print("🚀 Initializing GPU-Optimized RAG System...")
        self.load_data()
        self.load_embeddings()
        
    def load_data(self):
        """Load clause library with actual text content"""
        print("Loading CUAD clause library...")
        try:
            with open("cuad_clause_library.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    clause = json.loads(line.strip())
                    self.clauses.append(clause)
                    # Store the actual clause text - use text field as primary, fallback to full_context
                    clause_text = clause.get("text", "")
                    if not clause_text:
                        clause_text = clause.get("full_context", "")
                    if clause_text:
                        self.clause_texts.append(clause_text)
                    else:
                        # Fallback to contract title if no text
                        self.clause_texts.append(clause.get("contract_title", "Unknown"))
            
            print(f"✅ Loaded {len(self.clauses)} clauses with actual text content")
        except Exception as e:
            print(f"❌ Error loading clause library: {e}")
            raise
    
    def load_embeddings(self):
        """Load FAISS index and GPU-optimized embedding model"""
        print("Loading FAISS index and GPU embedding model...")
        try:
            # Load embedding model on GPU
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            
            # Load FAISS index
            self.faiss_index = faiss.read_index("embeddings/output/cuad_faiss.index")
            
            print("✅ FAISS index and GPU embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
    
    def classify_query_type(self, query: str) -> Dict[str, Any]:
        """Classify the type of query to optimize context selection"""
        query_lower = query.lower()
        
        # Query type patterns
        patterns = {
            "clause_improvement": [
                "improve", "enhance", "better", "strengthen", "fix", "modify",
                "rewrite", "redraft", "update", "optimize"
            ],
            "clause_analysis": [
                "analyze", "explain", "what does", "how does", "interpret",
                "break down", "examine", "review"
            ],
            "clause_comparison": [
                "compare", "difference", "versus", "vs", "similar", "contrast"
            ],
            "clause_search": [
                "find", "search", "locate", "identify", "what clauses"
            ],
            "general_question": [
                "what is", "how to", "when", "where", "why"
            ]
        }
        
        # Determine query type
        query_type = "general_question"  # default
        confidence = 0.0
        
        for qtype, keywords in patterns.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > 0:
                confidence = matches / len(keywords)
                if confidence > 0.1:  # Threshold for classification
                    query_type = qtype
                    break
        
        # GPU-optimized configuration adjustments
        config_adjustments = {
            "clause_improvement": {
                "max_context_docs": 8,
                "max_prompt_length": 4000,
                "temperature": 0.4,
                "num_predict": 1024
            },
            "clause_analysis": {
                "max_context_docs": 6,
                "max_prompt_length": 3500,
                "temperature": 0.2,
                "num_predict": 768
            },
            "clause_comparison": {
                "max_context_docs": 10,
                "max_prompt_length": 4500,
                "temperature": 0.3,
                "num_predict": 1280
            },
            "clause_search": {
                "max_context_docs": 4,
                "max_prompt_length": 2000,
                "temperature": 0.1,
                "num_predict": 512
            },
            "general_question": {
                "max_context_docs": 4,
                "max_prompt_length": 2500,
                "temperature": 0.3,
                "num_predict": 768
            }
        }
        
        return {
            "type": query_type,
            "confidence": confidence,
            "config": config_adjustments.get(query_type, {})
        }
    
    def search_clauses(self, query: str, top_k: int = 20) -> List[Dict]:
        """GPU-accelerated search with better relevance scoring"""
        start_time = time.time()
        
        # Generate embedding for query on GPU
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Get relevant clauses
        relevant_clauses = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.clauses):
                clause = self.clauses[idx].copy()
                clause["relevance_score"] = float(score)
                clause["rank"] = i + 1
                relevant_clauses.append(clause)
        
        search_time = time.time() - start_time
        print(f"🚀 GPU-accelerated FAISS search completed in {search_time:.2f}s")
        
        return relevant_clauses
    
    def build_enhanced_prompt(self, query: str, context_docs: List[Dict], query_type: str) -> str:
        """Build context-aware prompts optimized for GPU processing"""
        
        # Select relevant context based on query type
        if query_type == "clause_improvement":
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                # Use full_context if text is too short
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1500]  # More context with GPU
                title = doc.get("contract_title", doc.get("title", "Unknown"))
                if clause_text:
                    context_texts.append(f"RELEVANT CLAUSE ({title}):\n{clause_text}\n")
            
            prompt = f"""You are an expert contract lawyer and legal drafting specialist. Your task is to improve the provided clause based on best practices and similar clauses from a comprehensive contract library.

CONTEXT CLAUSES:
{chr(10).join(context_texts)}

USER'S CLAUSE TO IMPROVE:
{query}

INSTRUCTIONS:
1. Analyze the user's clause for potential issues, gaps, or areas for improvement
2. Reference the context clauses for best practices and effective language
3. Provide an improved version of the clause with detailed explanations
4. Highlight key improvements made and their rationale
5. Suggest additional considerations and potential risks
6. Provide alternative approaches if applicable

Please provide a comprehensive improvement with clear explanations and practical recommendations."""

        elif query_type == "clause_analysis":
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1200]
                title = doc.get("contract_title", doc.get("title", "Unknown"))
                if clause_text:
                    context_texts.append(f"REFERENCE CLAUSE ({title}):\n{clause_text}\n")
            
            prompt = f"""You are a legal expert specializing in contract analysis. Analyze the following query using the provided context clauses as reference material.

REFERENCE CLAUSES:
{chr(10).join(context_texts)}

QUERY TO ANALYZE:
{query}

INSTRUCTIONS:
1. Provide a comprehensive analysis of the query
2. Reference relevant aspects from the context clauses
3. Explain legal implications and considerations
4. Offer practical insights and recommendations
5. Identify potential risks and mitigation strategies
6. Use clear, professional language

Please provide a detailed analysis with practical insights and actionable recommendations."""

        else:
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1200]
                title = doc.get("contract_title", doc.get("title", "Unknown"))
                if clause_text:
                    context_texts.append(f"RELEVANT CLAUSE ({title}):\n{clause_text}\n")
            
            prompt = f"""You are a legal expert assistant. Answer the following question using the provided contract clauses as reference material.

RELEVANT CLAUSES:
{chr(10).join(context_texts)}

QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based on the provided context
2. Reference specific clauses when relevant
3. Provide practical and actionable information
4. If the context doesn't fully address the question, acknowledge limitations
5. Use clear, professional language
6. Provide comprehensive insights and recommendations

Please provide a comprehensive answer with practical guidance."""

        # Truncate if too long
        word_count = len(prompt.split())
        if word_count > self.config["max_prompt_length"]:
            words = prompt.split()
            prompt = " ".join(words[:self.config["max_prompt_length"]])
            prompt += "\n\n[Context truncated for length]"
        
        return prompt
    
    def query_ollama(self, prompt: str, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL) -> str:
        """Query Ollama with GPU-optimized parameters"""
        url = ollama_url + "/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config["temperature"],
                "num_predict": self.config["num_predict"],
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "gpu_layers": 50  # Enable GPU acceleration
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, timeout=self.config["timeout"])
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            if not answer or answer.lower() in ["unknown", "i don't know", "i cannot answer"]:
                return "I apologize, but I couldn't generate a meaningful response. This might be due to insufficient context or the complexity of the query. Please try rephrasing your question or providing more specific details."
            
            generation_time = time.time() - start_time
            print(f"🚀 GPU-accelerated generation completed in {generation_time:.2f}s")
            
            return answer
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return f"Request timed out after {elapsed:.1f}s. The query may be too complex or the context too large. Try simplifying your question or reducing the scope."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}. Please check if Ollama is running and the model is available."
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def process_query(self, query: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """Process a query with GPU-optimized context selection and response generation"""
        start_time = time.time()
        
        # Classify query type
        query_info = self.classify_query_type(query)
        query_type = query_info["type"]
        
        # Update configuration based on query type
        for key, value in query_info["config"].items():
            self.config[key] = value
        
        print(f"🔍 Query type: {query_type} (confidence: {query_info['confidence']:.2f})")
        print(f"📊 Using GPU-optimized config: {self.config}")
        
        # Search for relevant clauses
        relevant_clauses = self.search_clauses(query, self.config["top_k"])
        
        if not relevant_clauses:
            return {
                "answer": "I couldn't find any relevant clauses in the database to help answer your question.",
                "context_docs": [],
                "processing_time": time.time() - start_time,
                "model_used": model,
                "context_size": 0,
                "query_type": query_type
            }
        
        # Build enhanced prompt
        prompt = self.build_enhanced_prompt(query, relevant_clauses, query_type)
        
        # Generate response
        answer = self.query_ollama(prompt, model)
        
        total_time = time.time() - start_time
        
        return {
            "answer": answer,
            "context_docs": relevant_clauses[:self.config["max_context_docs"]],
            "processing_time": total_time,
            "model_used": model,
            "context_size": len(prompt.split()),
            "query_type": query_type,
            "device": str(self.device)
        }

def main():
    """Test the GPU-optimized RAG system"""
    system = GPURAGSystem()
    
    test_queries = [
        "Find intellectual property clauses",
        "Analyze this confidentiality clause: [Employee shall not disclose company secrets]",
        "Compare termination clauses across different contract types"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print(f"{'='*60}")
        
        result = system.process_query(query)
        
        print(f"⏱️  Total time: {result['processing_time']:.2f}s")
        print(f"🤖 Answer: {result['answer'][:200]}...")
        print(f"📚 Context docs: {len(result['context_docs'])}")
        print(f"🔧 Device: {result['device']}")

if __name__ == "__main__":
    main() 