#!/usr/bin/env python3
# src/evaluation/evaluator.py
"""
Evaluation suite for multi-modal RAG system.
Tracks retrieval metrics, answer quality, and latency.
"""

import time
import json
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from ..retriever.rag_pipeline import answer_query, answer_query_grouped_by_modality
from ..config import PROJECT_ROOT


_logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate multi-modal RAG system performance."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize evaluator."""
        self.output_dir = output_dir or PROJECT_ROOT / "evaluation_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def evaluate_query(self, query: str, ground_truth: str = None, 
                      expected_modalities: List[str] = None) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            query: Query to evaluate
            ground_truth: Expected answer (for comparison)
            expected_modalities: Which modalities should be relevant (text/image/table)
            
        Returns:
            Evaluation result dict with metrics
        """
        start_time = time.time()
        
        # Get answer
        result = answer_query(query, top_k=5, use_multi_modal=True)
        retrieval_time = result['retrieval_time']
        total_time = time.time() - start_time
        
        # Extract metrics
        eval_result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": result['answer'],
            "num_retrieved": len(result['retrieved']),
            "retrieval_time": retrieval_time,
            "total_time": total_time,
            "modalities_retrieved": list(set(r['modality'] for r in result['retrieved'])),
            "top_score": result['retrieved'][0]['score'] if result['retrieved'] else 0,
            "ground_truth": ground_truth,
            "llm_attempts": result.get('llm_attempts', 1),
            "llm_retries": result.get('llm_retries', 0),
        }
        
        # Check if expected modalities were retrieved
        if expected_modalities:
            retrieved_modalities = set(r['modality'] for r in result['retrieved'])
            expected_set = set(expected_modalities)
            eval_result["expected_modalities"] = expected_modalities
            eval_result["modality_coverage"] = len(retrieved_modalities & expected_set) / len(expected_set) if expected_set else 0
        
        self.results.append(eval_result)
        return eval_result
    
    def evaluate_batch(self, queries: List[Dict]) -> List[Dict]:
        """
        Evaluate multiple queries.
        
        Args:
            queries: List of dicts with 'query', 'ground_truth' (optional), 'expected_modalities' (optional)
            
        Returns:
            List of evaluation results
        """
        print(f"Evaluating {len(queries)} queries...")
        
        for i, q_dict in enumerate(queries, 1):
            query = q_dict['query'] if isinstance(q_dict, dict) else q_dict
            ground_truth = q_dict.get('ground_truth') if isinstance(q_dict, dict) else None
            expected_mods = q_dict.get('expected_modalities') if isinstance(q_dict, dict) else None
            
            print(f"  [{i}/{len(queries)}] {query[:60]}...", end="", flush=True)
            
            try:
                self.evaluate_query(query, ground_truth, expected_mods)
                last_result = self.results[-1]
                retry_info = last_result.get("llm_retries", 0)
                if retry_info:
                    print(f"ok (retries: {retry_info})")
                else:
                    print("ok")
            except Exception as e:
                _logger.warning("Evaluation failed for query %r: %s", query, e)
                print(f" failed ({str(e)[:30]}...)")
        
        return self.results
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from evaluation results."""
        if not self.results:
            return {}

        retrieval_times = [r['retrieval_time'] for r in self.results]
        total_times = [r['total_time'] for r in self.results]
        num_retrieved = [r['num_retrieved'] for r in self.results]
        top_scores = [r['top_score'] for r in self.results]

        # Modality coverage stats
        all_modalities = {}
        for r in self.results:
            for m in r.get('modalities_retrieved', []):
                all_modalities[m] = all_modalities.get(m, 0) + 1

        return {
            "num_queries": len(self.results),
            "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
            "avg_total_time": sum(total_times) / len(total_times),
            "min_retrieval_time": min(retrieval_times),
            "max_retrieval_time": max(retrieval_times),
            "avg_num_retrieved": sum(num_retrieved) / len(num_retrieved),
            "avg_top_score": sum(top_scores) / len(top_scores),
            "modality_distribution": all_modalities,
            "num_empty_results": sum(1 for r in self.results if r['num_retrieved'] == 0)
        }
    def save_results(self, filename: str = None) -> Path:
        """Save evaluation results to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
        filepath = self.output_dir / filename
        # Prepare output
        output = {
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "summary": self.get_summary_stats()
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath

    def print_summary(self):
        """Print summary statistics."""
        stats = self.get_summary_stats()

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Queries evaluated: {stats.get('num_queries', 0)}")
        print(f"Avg retrieval time: {stats.get('avg_retrieval_time', 0):.3f}s")
        print(f"Avg total time: {stats.get('avg_total_time', 0):.3f}s")
        print(f"Avg results per query: {stats.get('avg_num_retrieved', 0):.1f}")
        print(f"Avg top score: {stats.get('avg_top_score', 0):.3f}")
        print(f"Empty results: {stats.get('num_empty_results', 0)}")

        print("\nModality distribution:")
        for modality, count in stats.get('modality_distribution', {}).items():
            pct = (count / len(self.results) * 100) if self.results else 0
            print(f"  {modality}: {count} ({pct:.1f}%)")
        print("="*60 + "\n")


# ==================== SAMPLE EVALUATION QUERIES ====================

SAMPLE_QUERIES = [
    {
        "query": "What are the main economic indicators discussed?",
        "expected_modalities": ["text", "table"]
    },
    {
        "query": "What does the chart show about trends over time?",
        "expected_modalities": ["image", "text"]
    },
    {
        "query": "What data is presented in the tables?",
        "expected_modalities": ["table", "text"]
    },
    {
        "query": "Describe the key visual elements from the figures.",
        "expected_modalities": ["image"]
    },
    {
        "query": "What are the conclusions stated?",
        "expected_modalities": ["text"]
    },
    {
        "query": "Compare the statistics across different sections.",
        "expected_modalities": ["table", "text"]
    },
    {
        "query": "What information is shown in the appendix?",
        "expected_modalities": ["text", "table"]
    },
    {
        "query": "Are there any charts or graphs? What do they represent?",
        "expected_modalities": ["image"]
    }
]


def run_sample_evaluation():
    """Run evaluation with sample queries."""
    evaluator = RAGEvaluator()

    print("\nStarting Multi-Modal RAG Evaluation...")
    print(f"Sample queries: {len(SAMPLE_QUERIES)}\n")

    evaluator.evaluate_batch(SAMPLE_QUERIES)

    evaluator.print_summary()
    evaluator.save_results()

    return evaluator


if __name__ == "__main__":
    evaluator = run_sample_evaluation()
