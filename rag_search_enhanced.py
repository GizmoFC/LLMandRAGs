"""
Enhanced RAG System for Complex Contract Analysis
Handles clause improvement, analysis, and complex queries with optimized memory usage
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

class QueryRequest(BaseModel):
    query: str
    mode: str = "rag"
    page: int = 1
    model: str = "gemma3:8b-instruct"  # Use quantized model by default

class QueryResponse(BaseModel):
    answer: str
    context_docs: List[Dict]
    processing_time: float
    model_used: str
    context_size: int

class EnhancedRAGSystem:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_model = None
        self.faiss_index = None
        self.clauses = []
        self.clause_texts = []
        
        # Enhanced configuration for complex queries
        self.config = {
            "max_context_docs": 3,      # Increased for complex analysis
            "max_prompt_length": 1500,  # Balanced for quality vs speed
            "timeout": 90,              # Increased timeout for complex queries
            "temperature": 0.3,         # Lower for more focused responses
            "num_predict": 512,         # Reasonable response length
            "top_k": 10,                # More candidates for better selection
        }
        
        print("🚀 Initializing Enhanced RAG System...")
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
        """Load FAISS index and embedding model"""
        print("Loading FAISS index and embedding model...")
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            
            # Load FAISS index
            self.faiss_index = faiss.read_index("embeddings/output/cuad_faiss.index")
            
            print("✅ FAISS index and embedding model loaded successfully")
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
        
        # Adjust configuration based on query type
        config_adjustments = {
            "clause_improvement": {
                "max_context_docs": 4,
                "max_prompt_length": 2000,
                "temperature": 0.4,
                "num_predict": 768
            },
            "clause_analysis": {
                "max_context_docs": 3,
                "max_prompt_length": 1800,
                "temperature": 0.2,
                "num_predict": 512
            },
            "clause_comparison": {
                "max_context_docs": 5,
                "max_prompt_length": 2200,
                "temperature": 0.3,
                "num_predict": 640
            },
            "clause_search": {
                "max_context_docs": 2,
                "max_prompt_length": 1200,
                "temperature": 0.1,
                "num_predict": 256
            },
            "general_question": {
                "max_context_docs": 2,
                "max_prompt_length": 1500,
                "temperature": 0.3,
                "num_predict": 512
            }
        }
        
        return {
            "type": query_type,
            "confidence": confidence,
            "config": config_adjustments.get(query_type, {})
        }
    
    def search_clauses(self, query: str, top_k: int = 10) -> List[Dict]:
        """Enhanced search with better relevance scoring"""
        start_time = time.time()
        
        # Generate embedding for query
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
        print(f"FAISS search completed in {search_time:.2f}s")
        
        return relevant_clauses
    
    def build_enhanced_prompt(self, query: str, context_docs: List[Dict], query_type: str) -> str:
        """Build context-aware prompts for different query types"""
        
        # Select relevant context based on query type
        if query_type == "clause_improvement":
            # For improvement queries, focus on similar clauses and best practices
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                # Use full_context if text is too short
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1000]  # Limit full context length
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
3. Provide an improved version of the clause with explanations
4. Highlight key improvements made
5. Suggest additional considerations if relevant

Please provide a comprehensive improvement with clear explanations."""

        elif query_type == "clause_analysis":
            # For analysis queries, provide detailed explanations
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                # Use full_context if text is too short
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1000]  # Limit full context length
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
5. Use clear, professional language

Please provide a detailed analysis with practical insights."""

        else:
            # General RAG prompt for other query types
            context_texts = []
            for doc in context_docs[:self.config["max_context_docs"]]:
                clause_text = doc.get("text", "")
                # Use full_context if text is too short
                if len(clause_text) < 100:
                    clause_text = doc.get("full_context", "")[:1000]  # Limit full context length
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

Please provide a comprehensive answer."""

        # Truncate if too long
        word_count = len(prompt.split())
        if word_count > self.config["max_prompt_length"]:
            # Truncate context while keeping essential parts
            words = prompt.split()
            prompt = " ".join(words[:self.config["max_prompt_length"]])
            prompt += "\n\n[Context truncated for length]"
        
        return prompt
    
    def query_ollama(self, prompt: str, model: str = "gemma3:8b-instruct") -> str:
        """Query Ollama with enhanced error handling and retry logic"""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config["temperature"],
                "num_predict": self.config["num_predict"],
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
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
            
            return answer
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            return f"Request timed out after {elapsed:.1f}s. The query may be too complex or the context too large. Try simplifying your question or reducing the scope."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}. Please check if Ollama is running and the model is available."
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def process_query(self, query: str, model: str = "gemma3:8b-instruct") -> Dict[str, Any]:
        """Process a query with enhanced context selection and response generation"""
        start_time = time.time()
        
        # Classify query type
        query_info = self.classify_query_type(query)
        query_type = query_info["type"]
        
        # Update configuration based on query type
        for key, value in query_info["config"].items():
            self.config[key] = value
        
        print(f"🔍 Query type: {query_type} (confidence: {query_info['confidence']:.2f})")
        print(f"📊 Using config: {self.config}")
        
        # Search for relevant clauses
        relevant_clauses = self.search_clauses(query, self.config["top_k"])
        
        if not relevant_clauses:
            return {
                "answer": "I couldn't find any relevant clauses in the database to help answer your question.",
                "context_docs": [],
                "processing_time": time.time() - start_time,
                "model_used": model,
                "context_size": 0
            }
        
        # Build enhanced prompt
        prompt = self.build_enhanced_prompt(query, relevant_clauses, query_type)
        
        print(f"📝 Built {query_type} prompt: {len(prompt.split())} words")
        print(f"🔗 Using {len(relevant_clauses[:self.config['max_context_docs']])} context documents")
        
        # Query Ollama
        print(f"🚀 Sending {query_type} request to Ollama (model: {model})...")
        answer = self.query_ollama(prompt, model)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "context_docs": relevant_clauses[:self.config["max_context_docs"]],
            "processing_time": processing_time,
            "model_used": model,
            "context_size": len(prompt.split()),
            "query_type": query_type
        }

# Initialize FastAPI app
app = FastAPI(title="Enhanced CUAD RAG System", version="2.0")

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = EnhancedRAGSystem()
    print("✅ Enhanced RAG System ready!")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced CUAD RAG System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .response { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .context { background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Enhanced CUAD RAG System</h1>
            <p><strong>Capabilities:</strong> Clause improvement, analysis, comparison, and complex legal queries</p>
            
            <form method="POST">
                <label><strong>Query:</strong></label><br>
                <textarea name="query" placeholder="Ask about clauses, request improvements, or analyze legal text..."></textarea><br>
                
                <label><strong>Model:</strong></label><br>
                <select name="model">
                    <option value="gemma3:8b-instruct">Gemma3 8B Instruct (Recommended)</option>
                    <option value="gemma3:2b-instruct">Gemma3 2B Instruct (Fast)</option>
                    <option value="llama3.1:8b-instruct">Llama3.1 8B Instruct (High Quality)</option>
                </select><br><br>
                
                <button type="submit">🚀 Process Query</button>
            </form>
            
            <div id="examples">
                <h3>💡 Example Queries:</h3>
                <ul>
                    <li><strong>Clause Improvement:</strong> "Improve this confidentiality clause: [paste clause here]"</li>
                    <li><strong>Analysis:</strong> "Analyze the risks in this intellectual property clause"</li>
                    <li><strong>Search:</strong> "Find clauses related to data protection and privacy"</li>
                    <li><strong>Comparison:</strong> "Compare termination clauses across different contract types"</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
async def process_query(
    query: str = Form(...),
    model: str = Form("gemma3:8b-instruct")
):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    if not query.strip():
        return "Please provide a query."
    
    print(f"Processing: {query} (model: {model})")
    
    try:
        result = rag_system.process_query(query, model)
        
        # Build HTML response
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced CUAD RAG System - Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .container {{ background: #f5f5f5; padding: 20px; border-radius: 10px; }}
                .response {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
                .context {{ background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }}
                .stats {{ background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .back-btn {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Enhanced CUAD RAG System - Results</h1>
                
                <div class="stats">
                    <strong>Query Type:</strong> {result.get('query_type', 'Unknown')}<br>
                    <strong>Model Used:</strong> {result['model_used']}<br>
                    <strong>Processing Time:</strong> {result['processing_time']:.2f}s<br>
                    <strong>Context Size:</strong> {result['context_size']} words<br>
                    <strong>Context Documents:</strong> {len(result['context_docs'])}
                </div>
                
                <div class="response">
                    <h3>🤖 Answer:</h3>
                    <p>{result['answer'].replace(chr(10), '<br>')}</p>
                </div>
                
                <div class="response">
                    <h3>📚 Context Documents Used:</h3>
                    {chr(10).join([f'<div class="context"><strong>{doc.get("title", "Unknown")}</strong><br>{doc.get("clause_text", "")[:200]}...</div>' for doc in result['context_docs']])}
                </div>
                
                <a href="/" class="back-btn">← Back to Query</a>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

@app.post("/api/query")
async def api_query(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.process_query(request.query, request.model)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Enhanced CUAD RAG System...")
    print("📊 Enhanced features:")
    print("   - Intelligent query classification")
    print("   - Context-aware prompt building")
    print("   - Optimized for complex legal analysis")
    print("   - Support for clause improvement queries")
    print("   - Memory-efficient quantized models")
    
    uvicorn.run(app, host="0.0.0.0", port=8011) 