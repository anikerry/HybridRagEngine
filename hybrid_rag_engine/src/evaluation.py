"""
RAG Evaluation Framework for Advanced Hybrid RAG Engine

This module provides comprehensive evaluation metrics for RAG systems including:
- Semantic similarity metrics
- Citation accuracy
- Response completeness 
- Query-response relevance
- Performance benchmarks
"""

import json
import time
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from enhanced_config import SETTINGS
from advanced_ask import AdvancedHybridRAG, QueryRequest

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    semantic_similarity: float
    citation_accuracy: float  
    response_completeness: float
    relevance_score: float
    processing_time: float
    context_precision: float
    context_recall: float

@dataclass
class TestCase:
    """Test case for evaluation"""
    question: str
    expected_answer: str
    relevant_sources: List[str]
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"

class RAGEvaluator:
    """Comprehensive RAG system evaluator"""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag_engine = AdvancedHybridRAG()
        self.results = []
        
    async def initialize(self):
        """Initialize the RAG engine"""
        await self.rag_engine.initialize()
        
    def calculate_semantic_similarity(self, generated: str, expected: str) -> float:
        """Calculate semantic similarity between generated and expected answers"""
        if not generated or not expected:
            return 0.0
            
        embeddings = self.sentence_model.encode([generated, expected])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, similarity)
    
    def calculate_citation_accuracy(self, citations: List[Dict], relevant_sources: List[str]) -> float:
        """Calculate how many citations match relevant sources"""
        if not citations or not relevant_sources:
            return 0.0
            
        cited_sources = {c.get('source', '').lower() for c in citations}
        relevant_sources_lower = {s.lower() for s in relevant_sources}
        
        # Intersection over union
        intersection = len(cited_sources.intersection(relevant_sources_lower))
        union = len(cited_sources.union(relevant_sources_lower))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_response_completeness(self, generated: str, expected: str) -> float:
        """Calculate how complete the response is compared to expected"""
        if not generated or not expected:
            return 0.0
            
        # Simple heuristic: ratio of lengths with semantic similarity weighting
        length_ratio = min(len(generated) / len(expected), 1.0) if len(expected) > 0 else 0.0
        semantic_sim = self.calculate_semantic_similarity(generated, expected)
        
        return (length_ratio + semantic_sim) / 2
    
    def calculate_context_metrics(self, question: str, retrieved_chunks: List[str], 
                                 relevant_sources: List[str]) -> Tuple[float, float]:
        """Calculate context precision and recall"""
        if not retrieved_chunks or not relevant_sources:
            return 0.0, 0.0
            
        # Simple implementation - in production would use more sophisticated matching
        relevant_retrieved = 0
        for chunk in retrieved_chunks:
            for source in relevant_sources:
                if source.lower() in chunk.lower():
                    relevant_retrieved += 1
                    break
        
        precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0.0
        recall = relevant_retrieved / len(relevant_sources) if relevant_sources else 0.0
        
        return precision, recall
    
    async def evaluate_single(self, test_case: TestCase) -> EvaluationMetrics:
        """Evaluate a single test case"""
        start_time = time.time()
        
        # Create query request
        request = QueryRequest(
            question=test_case.question,
            use_streaming=False,
            llm_provider="ollama",
            enable_reranking=True,
            enable_query_expansion=False
        )
        
        # Get response from RAG system
        try:
            response = await self.rag_engine.query(request)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            semantic_sim = self.calculate_semantic_similarity(
                response.answer, test_case.expected_answer
            )
            
            citation_acc = self.calculate_citation_accuracy(
                response.citations, test_case.relevant_sources
            )
            
            completeness = self.calculate_response_completeness(
                response.answer, test_case.expected_answer
            )
            
            # Extract chunks for context metrics
            retrieved_chunks = [f"{c['source']} p.{c.get('page', '')}" for c in response.citations]
            context_precision, context_recall = self.calculate_context_metrics(
                test_case.question, retrieved_chunks, test_case.relevant_sources
            )
            
            # Calculate overall relevance (weighted combination)
            relevance_score = (
                semantic_sim * 0.4 + 
                citation_acc * 0.3 + 
                completeness * 0.2 +
                (context_precision + context_recall) / 2 * 0.1
            )
            
            return EvaluationMetrics(
                semantic_similarity=semantic_sim,
                citation_accuracy=citation_acc,
                response_completeness=completeness,
                relevance_score=relevance_score,
                processing_time=processing_time,
                context_precision=context_precision,
                context_recall=context_recall
            )
            
        except Exception as e:
            print(f"Error evaluating test case: {e}")
            return EvaluationMetrics(
                semantic_similarity=0.0,
                citation_accuracy=0.0,
                response_completeness=0.0,
                relevance_score=0.0,
                processing_time=time.time() - start_time,
                context_precision=0.0,
                context_recall=0.0
            )
    
    async def run_evaluation(self, test_cases: List[TestCase], 
                           output_file: str = "evaluation_results.json") -> Dict[str, Any]:
        """Run full evaluation suite"""
        print(f"🧪 Starting evaluation with {len(test_cases)} test cases...")
        
        all_metrics = []
        detailed_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"  Evaluating case {i}/{len(test_cases)}: {test_case.category}")
            
            metrics = await self.evaluate_single(test_case)
            all_metrics.append(metrics)
            
            detailed_results.append({
                "question": test_case.question,
                "category": test_case.category,
                "difficulty": test_case.difficulty,
                "metrics": metrics.__dict__
            })
            
            # Brief progress update
            if i % 5 == 0:
                avg_relevance = np.mean([m.relevance_score for m in all_metrics])
                print(f"    Progress: {i}/{len(test_cases)} | Avg Relevance: {avg_relevance:.3f}")
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            "overall_performance": {
                "avg_semantic_similarity": np.mean([m.semantic_similarity for m in all_metrics]),
                "avg_citation_accuracy": np.mean([m.citation_accuracy for m in all_metrics]),
                "avg_response_completeness": np.mean([m.response_completeness for m in all_metrics]),
                "avg_relevance_score": np.mean([m.relevance_score for m in all_metrics]),
                "avg_processing_time": np.mean([m.processing_time for m in all_metrics]),
                "avg_context_precision": np.mean([m.context_precision for m in all_metrics]),
                "avg_context_recall": np.mean([m.context_recall for m in all_metrics])
            },
            "performance_by_category": self._group_by_category(test_cases, all_metrics),
            "performance_by_difficulty": self._group_by_difficulty(test_cases, all_metrics)
        }
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "settings": SETTINGS.dict(),
            "aggregate_metrics": aggregate_metrics,
            "detailed_results": detailed_results
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Evaluation complete! Results saved to {output_file}")
        return results
    
    def _group_by_category(self, test_cases: List[TestCase], metrics: List[EvaluationMetrics]) -> Dict:
        """Group results by category"""
        category_metrics = {}
        for test_case, metric in zip(test_cases, metrics):
            if test_case.category not in category_metrics:
                category_metrics[test_case.category] = []
            category_metrics[test_case.category].append(metric)
        
        return {
            category: {
                "count": len(metrics),
                "avg_relevance": np.mean([m.relevance_score for m in metrics]),
                "avg_processing_time": np.mean([m.processing_time for m in metrics])
            }
            for category, metrics in category_metrics.items()
        }
    
    def _group_by_difficulty(self, test_cases: List[TestCase], metrics: List[EvaluationMetrics]) -> Dict:
        """Group results by difficulty"""
        difficulty_metrics = {}
        for test_case, metric in zip(test_cases, metrics):
            if test_case.difficulty not in difficulty_metrics:
                difficulty_metrics[test_case.difficulty] = []
            difficulty_metrics[test_case.difficulty].append(metric)
        
        return {
            difficulty: {
                "count": len(metrics),
                "avg_relevance": np.mean([m.relevance_score for m in metrics]),
                "avg_processing_time": np.mean([m.processing_time for m in metrics])
            }
            for difficulty, metrics in difficulty_metrics.items()
        }
    
    def generate_report(self, results_file: str = "evaluation_results.json", 
                       output_dir: str = "evaluation_report") -> str:
        """Generate comprehensive evaluation report with visualizations"""
        
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        self._create_performance_charts(results, output_dir)
        
        # Generate HTML report
        report_html = self._generate_html_report(results)
        
        report_file = Path(output_dir) / "evaluation_report.html"
        with open(report_file, 'w') as f:
            f.write(report_html)
        
        print(f"📊 Evaluation report generated: {report_file}")
        return str(report_file)
    
    def _create_performance_charts(self, results: Dict, output_dir: str):
        """Create performance visualization charts"""
        detailed_results = results["detailed_results"]
        
        # Performance metrics comparison
        metrics_data = []
        for result in detailed_results:
            metrics = result["metrics"]
            metrics_data.append({
                "Category": result["category"],
                "Difficulty": result["difficulty"],
                "Semantic Similarity": metrics["semantic_similarity"],
                "Citation Accuracy": metrics["citation_accuracy"],
                "Response Completeness": metrics["response_completeness"],
                "Relevance Score": metrics["relevance_score"],
                "Processing Time": metrics["processing_time"]
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance by category
        category_metrics = df.groupby('Category')[['Semantic Similarity', 'Citation Accuracy', 
                                                  'Response Completeness', 'Relevance Score']].mean()
        category_metrics.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Performance by Category')
        axes[0,0].set_ylabel('Score')
        
        # Performance by difficulty
        difficulty_metrics = df.groupby('Difficulty')[['Semantic Similarity', 'Citation Accuracy', 
                                                      'Response Completeness', 'Relevance Score']].mean()
        difficulty_metrics.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Performance by Difficulty')
        axes[0,1].set_ylabel('Score')
        
        # Processing time distribution
        df['Processing Time'].hist(bins=20, ax=axes[1,0])
        axes[1,0].set_title('Processing Time Distribution')
        axes[1,0].set_xlabel('Processing Time (seconds)')
        
        # Correlation heatmap
        correlation_matrix = df[['Semantic Similarity', 'Citation Accuracy', 
                                'Response Completeness', 'Relevance Score']].corr()
        sns.heatmap(correlation_matrix, annot=True, ax=axes[1,1])
        axes[1,1].set_title('Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "performance_charts.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, results: Dict) -> str:
        """Generate HTML evaluation report"""
        aggregate = results["aggregate_metrics"]["overall_performance"]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG System Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🧪 RAG System Evaluation Report</h1>
            <p><strong>Generated:</strong> {results["timestamp"]}</p>
            
            <h2>📊 Overall Performance</h2>
            <div class="metric">
                <strong>Average Relevance Score:</strong>
                <span class="score">{aggregate["avg_relevance_score"]:.3f}</span>
            </div>
            <div class="metric">
                <strong>Semantic Similarity:</strong>
                <span class="score">{aggregate["avg_semantic_similarity"]:.3f}</span>
            </div>
            <div class="metric">
                <strong>Citation Accuracy:</strong>
                <span class="score">{aggregate["avg_citation_accuracy"]:.3f}</span>
            </div>
            <div class="metric">
                <strong>Response Completeness:</strong>
                <span class="score">{aggregate["avg_response_completeness"]:.3f}</span>
            </div>
            <div class="metric">
                <strong>Average Processing Time:</strong>
                <span class="score">{aggregate["avg_processing_time"]:.2f}s</span>
            </div>
            
            <h2>📈 Performance Charts</h2>
            <img src="performance_charts.png" alt="Performance Charts" style="max-width: 100%;">
            
            <h2>📋 Performance by Category</h2>
            <table>
                <tr><th>Category</th><th>Count</th><th>Avg Relevance</th><th>Avg Processing Time</th></tr>
        """
        
        for category, metrics in results["aggregate_metrics"]["performance_by_category"].items():
            html += f"""
                <tr>
                    <td>{category}</td>
                    <td>{metrics["count"]}</td>
                    <td>{metrics["avg_relevance"]:.3f}</td>
                    <td>{metrics["avg_processing_time"]:.2f}s</td>
                </tr>
            """
        
        html += """
            </table>
            
            <footer style="margin-top: 50px; text-align: center; color: #666;">
                <p>Generated by Advanced Hybrid RAG Engine Evaluation Framework</p>
            </footer>
        </body>
        </html>
        """
        
        return html

# Sample test cases for demo
SAMPLE_TEST_CASES = [
    TestCase(
        question="What are the main compliance requirements?",
        expected_answer="The main compliance requirements include regulatory standards, documentation requirements, and audit procedures.",
        relevant_sources=["Thesis Regulations at Bosch.pdf"],
        difficulty="medium",
        category="compliance"
    ),
    TestCase(
        question="How do I submit a thesis proposal?",
        expected_answer="To submit a thesis proposal, you need to follow the documented submission process including proper formatting and required approvals.",
        relevant_sources=["Thesis Regulations at Bosch.pdf"],
        difficulty="easy", 
        category="process"
    ),
    TestCase(
        question="What are the Thor project specifications?",
        expected_answer="Thor project specifications include technical requirements, performance criteria, and implementation guidelines.",
        relevant_sources=["Thor.pdf"],
        difficulty="medium",
        category="technical"
    )
]

async def run_demo_evaluation():
    """Run a demo evaluation"""
    evaluator = RAGEvaluator()
    await evaluator.initialize()
    
    results = await evaluator.run_evaluation(SAMPLE_TEST_CASES)
    report_file = evaluator.generate_report()
    
    print(f"\n🎯 Evaluation Summary:")
    print(f"Overall Relevance Score: {results['aggregate_metrics']['overall_performance']['avg_relevance_score']:.3f}")
    print(f"Average Processing Time: {results['aggregate_metrics']['overall_performance']['avg_processing_time']:.2f}s")
    print(f"Report generated: {report_file}")

if __name__ == "__main__":
    asyncio.run(run_demo_evaluation())