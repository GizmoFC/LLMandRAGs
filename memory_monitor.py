"""
Memory Monitor for RAG System Optimization
Tracks memory usage and provides recommendations for optimization
"""

import psutil
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import os

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.memory_threshold = 0.85  # 85% memory usage threshold
        
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info().rss
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=memory.used,
            memory_percent=memory.percent,
            process_memory=process_memory,
            process_memory_percent=(process_memory / memory.total) * 100
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_current_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        snapshot = self.take_snapshot()
        return {
            "memory_usage_gb": round(snapshot.used_memory / (1024**3), 2),
            "available_memory_gb": round(snapshot.available_memory / (1024**3), 2),
            "total_memory_gb": round(snapshot.total_memory / (1024**3), 2),
            "memory_percent": round(snapshot.memory_percent, 1),
            "process_memory_mb": round(snapshot.process_memory / (1024**2), 2),
            "is_critical": snapshot.memory_percent > 90,
            "is_warning": snapshot.memory_percent > self.memory_threshold * 100
        }
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        status = self.get_current_memory_status()
        recommendations = []
        
        if status["memory_percent"] > 90:
            recommendations.append("🚨 CRITICAL: Memory usage is very high (>90%). Consider:")
            recommendations.append("   - Close unnecessary applications")
            recommendations.append("   - Use smaller quantized models (gemma3:2b-instruct)")
            recommendations.append("   - Reduce context size in RAG queries")
            recommendations.append("   - Restart the system if needed")
        
        elif status["memory_percent"] > 85:
            recommendations.append("⚠️ WARNING: Memory usage is high (>85%). Consider:")
            recommendations.append("   - Switch to smaller models for complex queries")
            recommendations.append("   - Limit context documents to 2-3")
            recommendations.append("   - Monitor memory usage during queries")
        
        elif status["memory_percent"] > 70:
            recommendations.append("📊 MODERATE: Memory usage is moderate (>70%). You can:")
            recommendations.append("   - Use standard models for most queries")
            recommendations.append("   - Use 3-4 context documents safely")
            recommendations.append("   - Monitor for memory spikes during complex queries")
        
        else:
            recommendations.append("✅ GOOD: Memory usage is healthy (<70%). You can:")
            recommendations.append("   - Use larger models for complex analysis")
            recommendations.append("   - Use 4-5 context documents")
            recommendations.append("   - Run multiple queries simultaneously")
        
        return recommendations
    
    def can_run_model(self, model_name: str) -> Dict[str, Any]:
        """Check if a specific model can run given current memory"""
        status = self.get_current_memory_status()
        
        # Model memory requirements (approximate)
        model_requirements = {
            "gemma3:2b-instruct": 1.5,  # GB
            "gemma3:8b-instruct": 4.5,  # GB
            "llama3.1:8b-instruct": 4.5,  # GB
            "gemma3": 4.5,  # GB (default)
        }
        
        required_gb = model_requirements.get(model_name, 4.5)
        available_gb = status["available_memory_gb"]
        
        can_run = available_gb >= required_gb
        safety_margin = available_gb - required_gb
        
        return {
            "can_run": can_run,
            "required_gb": required_gb,
            "available_gb": available_gb,
            "safety_margin_gb": safety_margin,
            "recommendation": self._get_model_recommendation(model_name, can_run, safety_margin)
        }
    
    def _get_model_recommendation(self, model_name: str, can_run: bool, safety_margin: float) -> str:
        """Get recommendation for model usage"""
        if not can_run:
            return f"❌ Cannot run {model_name} - insufficient memory. Try gemma3:2b-instruct instead."
        
        if safety_margin < 1.0:
            return f"⚠️ {model_name} can run but with limited safety margin. Consider closing other applications."
        
        if safety_margin < 2.0:
            return f"✅ {model_name} can run safely with moderate memory usage."
        
        return f"✅ {model_name} can run safely with plenty of memory available."
    
    def get_optimal_config(self, query_type: str) -> Dict[str, Any]:
        """Get optimal configuration based on current memory status"""
        status = self.get_current_memory_status()
        
        if status["memory_percent"] > 90:
            # Critical memory - use minimal configuration
            return {
                "model": "gemma3:2b-instruct",
                "max_context_docs": 1,
                "max_prompt_length": 800,
                "timeout": 60,
                "temperature": 0.2,
                "num_predict": 256
            }
        
        elif status["memory_percent"] > 85:
            # High memory - use conservative configuration
            return {
                "model": "gemma3:2b-instruct",
                "max_context_docs": 2,
                "max_prompt_length": 1200,
                "timeout": 90,
                "temperature": 0.3,
                "num_predict": 384
            }
        
        elif status["memory_percent"] > 70:
            # Moderate memory - use balanced configuration
            return {
                "model": "gemma3:8b-instruct",
                "max_context_docs": 3,
                "max_prompt_length": 1500,
                "timeout": 120,
                "temperature": 0.3,
                "num_predict": 512
            }
        
        else:
            # Good memory - use optimal configuration
            return {
                "model": "gemma3:8b-instruct",
                "max_context_docs": 4,
                "max_prompt_length": 2000,
                "timeout": 150,
                "temperature": 0.4,
                "num_predict": 640
            }
    
    def monitor_query_execution(self, query_func, *args, **kwargs):
        """Monitor memory during query execution"""
        print("🔍 Starting memory monitoring...")
        
        # Pre-query snapshot
        pre_snapshot = self.take_snapshot()
        print(f"📊 Pre-query memory: {pre_snapshot.memory_percent:.1f}% used")
        
        start_time = time.time()
        
        try:
            # Execute query
            result = query_func(*args, **kwargs)
            
            # Post-query snapshot
            post_snapshot = self.take_snapshot()
            execution_time = time.time() - start_time
            
            print(f"📊 Post-query memory: {post_snapshot.memory_percent:.1f}% used")
            print(f"⏱️ Query execution time: {execution_time:.2f}s")
            
            # Memory analysis
            memory_delta = post_snapshot.memory_percent - pre_snapshot.memory_percent
            if memory_delta > 5:
                print(f"⚠️ Memory spike detected: +{memory_delta:.1f}%")
            elif memory_delta > 2:
                print(f"📈 Moderate memory increase: +{memory_delta:.1f}%")
            else:
                print(f"✅ Stable memory usage: +{memory_delta:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None
    
    def save_memory_report(self, filename: str = "memory_report.json"):
        """Save memory monitoring report"""
        report = {
            "timestamp": time.time(),
            "current_status": self.get_current_memory_status(),
            "recommendations": self.get_memory_recommendations(),
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots[-10:]],  # Last 10 snapshots
            "model_capabilities": {
                model: self.can_run_model(model) 
                for model in ["gemma3:2b-instruct", "gemma3:8b-instruct", "llama3.1:8b-instruct"]
            }
        }
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Memory report saved to {filename}")

def main():
    """Test the memory monitor"""
    monitor = MemoryMonitor()
    
    print("🧠 Memory Monitor Test")
    print("=" * 50)
    
    # Current memory status
    status = monitor.get_current_memory_status()
    print(f"Current Memory Usage: {status['memory_percent']}%")
    print(f"Available Memory: {status['available_memory_gb']} GB")
    print(f"Process Memory: {status['process_memory_mb']} MB")
    
    # Recommendations
    print("\n📋 Recommendations:")
    for rec in monitor.get_memory_recommendations():
        print(rec)
    
    # Model capabilities
    print("\n🤖 Model Capabilities:")
    for model in ["gemma3:2b-instruct", "gemma3:8b-instruct", "llama3.1:8b-instruct"]:
        capability = monitor.can_run_model(model)
        print(f"  {model}: {capability['recommendation']}")
    
    # Optimal configurations
    print("\n⚙️ Optimal Configurations:")
    for query_type in ["clause_improvement", "clause_analysis", "general_search"]:
        config = monitor.get_optimal_config(query_type)
        print(f"  {query_type}: {config['model']} with {config['max_context_docs']} docs")
    
    # Save report
    monitor.save_memory_report()

if __name__ == "__main__":
    main() 