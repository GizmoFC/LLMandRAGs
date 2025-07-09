"""
Complex Query Handler for Advanced Contract Analysis
Optimized for memory efficiency while handling sophisticated queries
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class QueryConfig:
    """Configuration for different query types"""
    max_context_docs: int
    max_prompt_length: int
    timeout: int
    temperature: float
    num_predict: int
    model: str

class ComplexQueryHandler:
    def __init__(self):
        # Load clause library
        self.clauses = self.load_clauses()
        
        # Query type configurations
        self.configs = {
            "clause_improvement": QueryConfig(
                max_context_docs=4,
                max_prompt_length=1800,
                timeout=120,
                temperature=0.4,
                num_predict=512,
                model="gemma3:2b-instruct"  # Use smaller model for complex tasks
            ),
            "clause_analysis": QueryConfig(
                max_context_docs=3,
                max_prompt_length=1500,
                timeout=90,
                temperature=0.3,
                num_predict=384,
                model="gemma3:2b-instruct"
            ),
            "clause_comparison": QueryConfig(
                max_context_docs=5,
                max_prompt_length=2000,
                timeout=150,
                temperature=0.3,
                num_predict=640,
                model="gemma3:8b-instruct"  # Use larger model for comparisons
            ),
            "general_search": QueryConfig(
                max_context_docs=3,
                max_prompt_length=1200,
                timeout=60,
                temperature=0.2,
                num_predict=256,
                model="gemma3:2b-instruct"
            )
        }
    
    def load_clauses(self) -> List[Dict]:
        """Load clauses with memory optimization"""
        clauses = []
        try:
            with open("cuad_clause_library.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    clause = json.loads(line.strip())
                    # Only keep essential fields to save memory
                    essential_clause = {
                        "contract_title": clause.get("contract_title", ""),
                        "text": clause.get("text", ""),
                        "full_context": clause.get("full_context", ""),
                        "clause_type": clause.get("clause_type", ""),
                        "source": clause.get("source", "")
                    }
                    clauses.append(essential_clause)
            print(f"✅ Loaded {len(clauses)} clauses (memory optimized)")
        except Exception as e:
            print(f"❌ Error loading clauses: {e}")
            return []
        return clauses
    
    def classify_query(self, query: str) -> str:
        """Classify query type for optimal configuration"""
        query_lower = query.lower()
        
        # Improvement keywords
        improvement_keywords = ["improve", "enhance", "better", "strengthen", "fix", "modify", "rewrite", "redraft"]
        if any(keyword in query_lower for keyword in improvement_keywords):
            return "clause_improvement"
        
        # Analysis keywords
        analysis_keywords = ["analyze", "explain", "interpret", "break down", "examine", "review", "what does", "how does"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            return "clause_analysis"
        
        # Comparison keywords
        comparison_keywords = ["compare", "difference", "versus", "vs", "similar", "contrast", "between"]
        if any(keyword in query_lower for keyword in comparison_keywords):
            return "clause_comparison"
        
        return "general_search"
    
    def find_relevant_clauses(self, query: str, max_docs: int) -> List[Dict]:
        """Find relevant clauses using simple keyword matching (memory efficient)"""
        query_words = set(query.lower().split())
        relevant_clauses = []
        
        for clause in self.clauses:
            # Create searchable text
            searchable_text = f"{clause['contract_title']} {clause['text']} {clause['clause_type']}".lower()
            searchable_words = set(searchable_text.split())
            
            # Calculate relevance score
            matches = len(query_words.intersection(searchable_words))
            if matches > 0:
                relevance_score = matches / len(query_words)
                clause_copy = clause.copy()
                clause_copy["relevance_score"] = relevance_score
                relevant_clauses.append(clause_copy)
        
        # Sort by relevance and return top matches
        relevant_clauses.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_clauses[:max_docs]
    
    def build_optimized_prompt(self, query: str, context_clauses: List[Dict], query_type: str) -> str:
        """Build memory-optimized prompts for different query types"""
        config = self.configs[query_type]
        
        if query_type == "clause_improvement":
            return self._build_improvement_prompt(query, context_clauses, config)
        elif query_type == "clause_analysis":
            return self._build_analysis_prompt(query, context_clauses, config)
        elif query_type == "clause_comparison":
            return self._build_comparison_prompt(query, context_clauses, config)
        else:
            return self._build_general_prompt(query, context_clauses, config)
    
    def _build_improvement_prompt(self, query: str, context_clauses: List[Dict], config: QueryConfig) -> str:
        """Build prompt for clause improvement"""
        context_texts = []
        for clause in context_clauses:
            title = clause.get("contract_title", "")
            text = clause.get("text", "")
            # Use full_context if text is too short
            if len(text) < 100:
                text = clause.get("full_context", "")[:500]
            else:
                text = text[:500]  # Limit text length
            context_texts.append(f"REFERENCE: {title}\n{text}\n")
        
        prompt = f"""You are a legal expert specializing in contract drafting. Improve the provided clause using best practices and reference materials.

REFERENCE CLAUSES:
{chr(10).join(context_texts)}

CLAUSE TO IMPROVE:
{query}

INSTRUCTIONS:
1. Identify areas for improvement (clarity, completeness, enforceability)
2. Provide an enhanced version with explanations
3. Reference best practices from the context clauses
4. Keep improvements practical and legally sound

IMPROVED CLAUSE:"""
        
        return self._truncate_prompt(prompt, config.max_prompt_length)
    
    def _build_analysis_prompt(self, query: str, context_clauses: List[Dict], config: QueryConfig) -> str:
        """Build prompt for clause analysis"""
        context_texts = []
        for clause in context_clauses:
            title = clause.get("contract_title", "")
            text = clause.get("text", "")
            # Use full_context if text is too short
            if len(text) < 100:
                text = clause.get("full_context", "")[:400]
            else:
                text = text[:400]
            context_texts.append(f"REFERENCE: {title}\n{text}\n")
        
        prompt = f"""You are a legal analyst. Analyze the following query using the provided reference clauses.

REFERENCE CLAUSES:
{chr(10).join(context_texts)}

ANALYSIS REQUEST:
{query}

INSTRUCTIONS:
1. Provide comprehensive legal analysis
2. Reference relevant aspects from context clauses
3. Identify key legal considerations
4. Offer practical insights and recommendations

ANALYSIS:"""
        
        return self._truncate_prompt(prompt, config.max_prompt_length)
    
    def _build_comparison_prompt(self, query: str, context_clauses: List[Dict], config: QueryConfig) -> str:
        """Build prompt for clause comparison"""
        context_texts = []
        for clause in context_clauses:
            title = clause.get("contract_title", "")
            text = clause.get("text", "")
            # Use full_context if text is too short
            if len(text) < 100:
                text = clause.get("full_context", "")[:300]
            else:
                text = text[:300]
            context_texts.append(f"CLAUSE: {title}\n{text}\n")
        
        prompt = f"""You are a legal expert. Compare and analyze the following clauses based on the query.

CLAUSES TO COMPARE:
{chr(10).join(context_texts)}

COMPARISON REQUEST:
{query}

INSTRUCTIONS:
1. Identify key differences and similarities
2. Analyze strengths and weaknesses
3. Provide practical recommendations
4. Consider legal implications

COMPARISON ANALYSIS:"""
        
        return self._truncate_prompt(prompt, config.max_prompt_length)
    
    def _build_general_prompt(self, query: str, context_clauses: List[Dict], config: QueryConfig) -> str:
        """Build general search prompt"""
        context_texts = []
        for clause in context_clauses:
            title = clause.get("contract_title", "")
            text = clause.get("text", "")
            # Use full_context if text is too short
            if len(text) < 100:
                text = clause.get("full_context", "")[:300]
            else:
                text = text[:300]
            context_texts.append(f"RELEVANT: {title}\n{text}\n")
        
        prompt = f"""You are a legal assistant. Answer the question using the provided reference clauses.

REFERENCE CLAUSES:
{chr(10).join(context_texts)}

QUESTION:
{query}

ANSWER:"""
        
        return self._truncate_prompt(prompt, config.max_prompt_length)
    
    def _truncate_prompt(self, prompt: str, max_words: int) -> str:
        """Truncate prompt to fit within word limit"""
        words = prompt.split()
        if len(words) <= max_words:
            return prompt
        
        # Truncate while preserving structure
        truncated_words = words[:max_words]
        truncated_prompt = " ".join(truncated_words)
        
        # Ensure we end with a complete instruction
        if not truncated_prompt.endswith((":", "INSTRUCTIONS:", "ANSWER:", "ANALYSIS:")):
            truncated_prompt += "\n\n[Context truncated for length]"
        
        return truncated_prompt
    
    def query_ollama(self, prompt: str, config: QueryConfig) -> str:
        """Query Ollama with optimized parameters"""
        url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.num_predict,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=config.timeout)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            if not answer or answer.lower() in ["unknown", "i don't know", "i cannot answer"]:
                return "I couldn't generate a meaningful response. Please try rephrasing your question or providing more specific details."
            
            return answer
            
        except requests.exceptions.Timeout:
            return f"Request timed out after {config.timeout}s. The query may be too complex. Try simplifying your question."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}. Please check if Ollama is running."
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    def process_complex_query(self, query: str) -> Dict[str, Any]:
        """Process a complex query with memory optimization"""
        start_time = time.time()
        
        # Classify query type
        query_type = self.classify_query(query)
        config = self.configs[query_type]
        
        print(f"🔍 Query type: {query_type}")
        print(f"📊 Using model: {config.model}")
        print(f"📝 Max context docs: {config.max_context_docs}")
        
        # Find relevant clauses
        relevant_clauses = self.find_relevant_clauses(query, config.max_context_docs)
        
        if not relevant_clauses:
            return {
                "answer": "I couldn't find any relevant clauses to help answer your question.",
                "context_docs": [],
                "processing_time": time.time() - start_time,
                "model_used": config.model,
                "query_type": query_type
            }
        
        # Build optimized prompt
        prompt = self.build_optimized_prompt(query, relevant_clauses, query_type)
        
        print(f"📝 Built {query_type} prompt: {len(prompt.split())} words")
        print(f"🔗 Using {len(relevant_clauses)} context documents")
        
        # Query Ollama
        print(f"🚀 Sending request to Ollama...")
        answer = self.query_ollama(prompt, config)
        
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "context_docs": relevant_clauses,
            "processing_time": processing_time,
            "model_used": config.model,
            "query_type": query_type,
            "prompt_length": len(prompt.split())
        }

def main():
    """Test the complex query handler"""
    handler = ComplexQueryHandler()
    
    # Test queries
    test_queries = [
        "Improve this confidentiality clause: The parties agree to keep confidential information secret.",
        "Analyze the risks in intellectual property clauses",
        "Compare termination clauses across different contract types",
        "Find clauses related to data protection"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print(f"{'='*60}")
        
        result = handler.process_complex_query(query)
        
        print(f"Query Type: {result['query_type']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Prompt Length: {result['prompt_length']} words")
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['context_docs']:
            print(f"\nContext Documents Used:")
            for i, doc in enumerate(result['context_docs'][:2], 1):
                print(f"  {i}. {doc['contract_title']} (relevance: {doc['relevance_score']:.2f})")

if __name__ == "__main__":
    main() 