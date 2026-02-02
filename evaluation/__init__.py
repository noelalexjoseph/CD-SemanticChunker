"""
Evaluation module for Construction RAG.

Provides RAGAS-based evaluation metrics for assessing RAG pipeline quality.
"""

from .ragas_evaluator import RAGEvaluator, EvaluationResult, TEST_CASES
from .test_cases import CONSTRUCTION_TEST_CASES

__all__ = [
    "RAGEvaluator",
    "EvaluationResult",
    "TEST_CASES",
    "CONSTRUCTION_TEST_CASES",
]
