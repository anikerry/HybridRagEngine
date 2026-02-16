#!/usr/bin/env python3
"""
Advanced RAG Engine Demo Runner

This script demonstrates the key capabilities of the Advanced Hybrid RAG Engine.
Perfect for showcasing to recruiters or during interviews.

Features demonstrated:
- Multi-modal LLM integration  
- Real-time streaming responses
- Performance analytics
- Evaluation metrics
- Production API capabilities
"""

import asyncio
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from advanced_ask import AdvancedHybridRAG, QueryRequest
from evaluation import RAGEvaluator, TestCase, SAMPLE_TEST_CASES

class RAGDemo:
    """Demo orchestrator for the Advanced RAG Engine"""
    
    def __init__(self):
        self.rag_engine = AdvancedHybridRAG()
        self.evaluator = RAGEvaluator()
        
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Advanced Hybrid RAG Engine...")
        await self.rag_engine.initialize()
        print("✅ RAG Engine ready!")
        
    async def demo_basic_capabilities(self):
        """Demonstrate basic RAG capabilities"""
        print("\n" + "="*60)
        print("🧠 DEMO 1: Basic RAG Capabilities")
        print("="*60)
        
        questions = [
            "What are the main compliance requirements?",
            "How do I submit a thesis proposal?",
            "What are the Thor project specifications?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 50)
            
            request = QueryRequest(
                question=question,
                llm_provider="ollama",
                model="llama3",
                enable_reranking=True
            )
            
            start_time = time.time()
            response = await self.rag_engine.query(request)
            processing_time = time.time() - start_time
            
            print(f"🤖 Answer: {response.answer[:200]}...")
            print(f"📚 Citations: {len(response.citations)} sources")
            print(f"⚡ Processing Time: {processing_time:.2f}s")
            print(f"🔧 Chunks Used: {response.metadata['chunks_used']}")
    
    async def demo_advanced_features(self):
        """Demonstrate advanced features"""
        print("\n" + "="*60)
        print("🎯 DEMO 2: Advanced RAG Features")
        print("="*60)
        
        question = "What are the compliance requirements for documentation?"
        
        # Test different configurations
        configs = [
            {"name": "Basic", "reranking": False, "expansion": False},
            {"name": "With Reranking", "reranking": True, "expansion": False},
            {"name": "With Query Expansion", "reranking": False, "expansion": True},
            {"name": "Full Advanced", "reranking": True, "expansion": True}
        ]
        
        results = []
        
        for config in configs:
            print(f"\n🔧 Testing: {config['name']}")
            print("-" * 30)
            
            request = QueryRequest(
                question=question,
                llm_provider="ollama",
                enable_reranking=config["reranking"],
                enable_query_expansion=config["expansion"]
            )
            
            start_time = time.time()
            response = await self.rag_engine.query(request)
            processing_time = time.time() - start_time
            
            results.append({
                "config": config["name"],
                "processing_time": processing_time,
                "chunks_used": response.metadata['chunks_used'],
                "answer_length": len(response.answer)
            })
            
            print(f"⚡ Time: {processing_time:.2f}s")
            print(f"📊 Chunks: {response.metadata['chunks_used']}")
            print(f"📝 Answer Length: {len(response.answer)} chars")
        
        # Compare results
        print("\n📊 PERFORMANCE COMPARISON:")
        print("-" * 40)
        for result in results:
            print(f"{result['config']:<20} | {result['processing_time']:.2f}s | {result['chunks_used']} chunks")
    
    async def demo_evaluation_framework(self):
        """Demonstrate evaluation capabilities"""
        print("\n" + "="*60)
        print("🧪 DEMO 3: Evaluation Framework")
        print("="*60)
        
        print("Running comprehensive evaluation on test cases...")
        
        # Initialize evaluator
        await self.evaluator.initialize()
        
        # Run evaluation on sample test cases
        results = await self.evaluator.run_evaluation(
            SAMPLE_TEST_CASES[:2],  # Use first 2 test cases for demo
            output_file="demo_evaluation_results.json"
        )
        
        # Display key metrics
        overall = results["aggregate_metrics"]["overall_performance"]
        
        print("\n📈 EVALUATION RESULTS:")
        print("-" * 30)
        print(f"🎯 Avg Relevance Score: {overall['avg_relevance_score']:.3f}")
        print(f"🧠 Semantic Similarity: {overall['avg_semantic_similarity']:.3f}")
        print(f"📚 Citation Accuracy: {overall['avg_citation_accuracy']:.3f}")
        print(f"⚡ Avg Processing Time: {overall['avg_processing_time']:.2f}s")
        print(f"🔍 Context Precision: {overall['avg_context_precision']:.3f}")
        print(f"🎪 Context Recall: {overall['avg_context_recall']:.3f}")
        
        # Generate report
        report_file = self.evaluator.generate_report(
            "demo_evaluation_results.json",
            "demo_evaluation_report"
        )
        print(f"\n📊 Detailed report: {report_file}")
    
    async def demo_streaming_capabilities(self):
        """Demonstrate streaming response capabilities"""  
        print("\n" + "="*60)
        print("📡 DEMO 4: Streaming Responses")
        print("="*60)
        
        question = "Explain the thesis submission process step by step."
        print(f"📝 Question: {question}")
        print("\n🤖 Streaming Response:")
        print("-" * 30)
        
        request = QueryRequest(
            question=question,
            use_streaming=True,
            llm_provider="ollama"
        )
        
        print("🔄 Starting stream...")
        async for chunk in self.rag_engine.stream_query(request):
            # Parse the streaming data
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data["type"] == "content":
                        print(data["content"], end=" ", flush=True)
                    elif data["type"] == "end":
                        print(f"\n\n✅ Stream completed in {data['processing_time']:.2f}s")
                except:
                    pass
    
    def demo_api_capabilities(self):
        """Demonstrate API server capabilities"""
        print("\n" + "="*60)
        print("🌐 DEMO 5: Production API Server")
        print("="*60)
        
        print("📡 API Server Features:")
        print("• RESTful API with FastAPI")
        print("• Automatic OpenAPI documentation")
        print("• Health monitoring endpoints")
        print("• Real-time analytics")
        print("• Response caching")
        print("• Multi-LLM support")
        
        print("\n🚀 To start the API server:")
        print("   python src/advanced_ask.py --server")
        print("   API: http://localhost:8000")
        print("   Docs: http://localhost:8000/docs")
        
        print("\n🎨 To start the Streamlit UI:")
        print("   streamlit run src/streamlit_app.py")
        print("   UI: http://localhost:8501")
        
        # Check if Qdrant is running
        try:
            import requests
            response = requests.get("http://localhost:6333", timeout=2)
            print("\n✅ Qdrant is running (required for demos)")
        except:
            print("\n⚠️  Qdrant not detected. Run: docker compose up -d")
    
    def show_project_highlights(self):
        """Show key project highlights for recruiters"""
        print("\n" + "="*60)
        print("🏆 PROJECT HIGHLIGHTS FOR GENAI ROLES")
        print("="*60)
        
        highlights = [
            "🧠 Advanced RAG Architecture",
            "   • Hybrid retrieval (semantic + keyword)",
            "   • Cross-encoder reranking",
            "   • Query expansion techniques",
            "   • Multi-LLM integration",
            "",
            "🚀 Production Engineering",
            "   • FastAPI with async operations",
            "   • Real-time streaming responses",
            "   • Response caching & optimization",
            "   • Health monitoring & analytics",
            "",
            "🧪 ML Engineering & Evaluation",
            "   • Comprehensive evaluation framework",
            "   • Automated testing with metrics",
            "   • Performance benchmarking",
            "   • A/B testing capabilities",
            "",
            "🎨 Full-Stack AI Development", 
            "   • Backend API development",
            "   • Frontend UI (Streamlit)",
            "   • Database integration (Qdrant)",
            "   • Analytics dashboards",
            "",
            "📊 Modern AI Practices",
            "   • Pydantic for data validation",
            "   • Structured JSON responses",
            "   • Error handling & logging",
            "   • Configuration management"
        ]
        
        for highlight in highlights:
            print(highlight)
    
    async def run_full_demo(self):
        """Run the complete demo sequence"""
        print("🎯 ADVANCED HYBRID RAG ENGINE - COMPREHENSIVE DEMO")
        print("=" * 60)
        print("Showcasing production-ready GenAI engineering capabilities")
        
        await self.initialize()
        
        # Run all demos
        await self.demo_basic_capabilities()
        await self.demo_advanced_features()
        await self.demo_evaluation_framework()
        await self.demo_streaming_capabilities()
        self.demo_api_capabilities()
        self.show_project_highlights()
        
        print("\n🎉 Demo complete!")
        print("💼 This project demonstrates modern GenAI engineering including:")
        print("   • Advanced RAG techniques")
        print("   • Production API development")  
        print("   • ML evaluation frameworks")
        print("   • Real-time streaming systems")
        print("   • Full-stack AI applications")

async def main():
    """Main demo runner"""
    demo = RAGDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    asyncio.run(main())