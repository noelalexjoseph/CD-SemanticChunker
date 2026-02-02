"""
RAGAS Evaluation for Construction Drawing RAG Pipeline.

This module evaluates the RAG pipeline using RAGAS-inspired metrics:
- Context Precision: Are retrieved contexts relevant?
- Context Recall: Are all relevant contexts retrieved?
- Faithfulness: Is the answer grounded in context? (requires LLM)
- Answer Relevancy: Is the answer relevant to the question? (requires LLM)

These metrics help assess the quality of the retrieval and generation
components of the construction document RAG system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from .test_cases import CONSTRUCTION_TEST_CASES

# Export test cases for convenience
TEST_CASES = CONSTRUCTION_TEST_CASES


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: List[str]
    context_precision: float
    context_recall: float
    keyword_coverage: float
    faithfulness: float
    answer_relevancy: float
    relevance_scores: List[float]
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer,
            "retrieved_contexts": self.retrieved_contexts,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "keyword_coverage": self.keyword_coverage,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "relevance_scores": self.relevance_scores
        }


class RAGEvaluator:
    """
    Evaluator for the Construction Drawing RAG pipeline.
    
    Computes retrieval metrics and optionally LLM-based metrics
    for assessing RAG quality.
    
    Example:
        >>> from construction_rag import ConstructionDrawingRAG
        >>> from construction_rag.evaluation import RAGEvaluator, TEST_CASES
        >>> 
        >>> rag = ConstructionDrawingRAG()
        >>> evaluator = RAGEvaluator(rag)
        >>> results = evaluator.evaluate_all(TEST_CASES)
        >>> print(f"F1 Score: {results['aggregate_metrics']['f1_score']:.2%}")
    """
    
    def __init__(self, rag_pipeline, llm=None):
        """
        Initialize the evaluator.
        
        Args:
            rag_pipeline: Initialized ConstructionDrawingRAG instance
            llm: Optional OpenRouterLLM instance for full RAGAS metrics
        """
        self.rag = rag_pipeline
        self.llm = llm
    
    def compute_keyword_coverage(
        self,
        contexts: List[str],
        keywords: List[str]
    ) -> float:
        """
        Compute what percentage of expected keywords appear in retrieved contexts.
        
        Args:
            contexts: Retrieved context strings
            keywords: Expected keywords
            
        Returns:
            Coverage ratio (0-1)
        """
        if not keywords:
            return 1.0
        
        combined_context = " ".join(contexts).lower()
        found = sum(1 for kw in keywords if kw.lower() in combined_context)
        return found / len(keywords)
    
    def compute_type_precision(
        self,
        retrieved_types: List[str],
        expected_types: List[str]
    ) -> float:
        """
        Compute precision based on chunk types.
        
        Args:
            retrieved_types: Types of retrieved chunks
            expected_types: Expected relevant types
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_types:
            return 0.0
        
        relevant = sum(1 for t in retrieved_types if t in expected_types)
        return relevant / len(retrieved_types)
    
    def compute_type_recall(
        self,
        retrieved_types: List[str],
        expected_types: List[str]
    ) -> float:
        """
        Compute recall based on chunk types.
        
        Args:
            retrieved_types: Types of retrieved chunks
            expected_types: Expected relevant types
            
        Returns:
            Recall score (0-1)
        """
        if not expected_types:
            return 1.0
        
        retrieved_set = set(retrieved_types)
        expected_set = set(expected_types)
        
        found = len(retrieved_set.intersection(expected_set))
        return found / len(expected_set)
    
    def compute_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Compute faithfulness: is the answer grounded in the contexts?
        
        Uses LLM to evaluate. Returns 0.0 if LLM not available.
        
        Args:
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Faithfulness score (0-1)
        """
        if not self.llm or not answer:
            return 0.0
        
        context_text = "\n---\n".join(contexts[:3])
        
        prompt = f"""Evaluate if the following answer is factually grounded in the given contexts.
Score from 0.0 (completely unfaithful) to 1.0 (completely faithful).
Only output a number between 0.0 and 1.0.

Contexts:
{context_text[:2000]}

Answer:
{answer}

Faithfulness score (0.0-1.0):"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=10, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0
    
    def compute_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Compute answer relevancy: is the answer relevant to the question?
        
        Uses LLM to evaluate. Returns 0.0 if LLM not available.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevancy score (0-1)
        """
        if not self.llm or not answer:
            return 0.0
        
        prompt = f"""Evaluate if the following answer is relevant to the question.
Score from 0.0 (completely irrelevant) to 1.0 (completely relevant).
Only output a number between 0.0 and 1.0.

Question: {question}

Answer: {answer}

Relevancy score (0.0-1.0):"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=10, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0
    
    def evaluate_single(
        self,
        test_case: dict,
        n_results: int = 5
    ) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Dict with question, ground_truth, relevant_chunk_types, keywords
            n_results: Number of chunks to retrieve
            
        Returns:
            EvaluationResult with all metrics
        """
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        expected_types = test_case["relevant_chunk_types"]
        keywords = test_case["keywords"]
        
        # Query the RAG pipeline
        results = self.rag.query(question, n_results=n_results)
        
        # Extract contexts and metadata
        contexts = [r.content for r in results]
        retrieved_types = [r.metadata.get("chunk_type", "unknown") for r in results]
        relevance_scores = [r.relevance_score for r in results]
        
        # Compute retrieval metrics
        precision = self.compute_type_precision(retrieved_types, expected_types)
        recall = self.compute_type_recall(retrieved_types, expected_types)
        keyword_cov = self.compute_keyword_coverage(contexts, keywords)
        
        # Generate answer and compute LLM metrics (if LLM available)
        generated_answer = ""
        faithfulness = 0.0
        answer_relevancy = 0.0
        
        if self.llm:
            generated_answer = self.llm.answer_query(question, contexts)
            faithfulness = self.compute_faithfulness(generated_answer, contexts)
            answer_relevancy = self.compute_answer_relevancy(question, generated_answer)
        
        return EvaluationResult(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            retrieved_contexts=contexts[:3],
            context_precision=precision,
            context_recall=recall,
            keyword_coverage=keyword_cov,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            relevance_scores=relevance_scores
        )
    
    def evaluate_all(
        self,
        test_cases: List[dict],
        n_results: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate all test cases.
        
        Args:
            test_cases: List of test case dictionaries
            n_results: Number of chunks to retrieve per query
            verbose: Whether to print progress
            
        Returns:
            Dict with individual_results and aggregate_metrics
        """
        results = []
        
        for i, tc in enumerate(test_cases, 1):
            if verbose:
                print(f"Evaluating [{i}/{len(test_cases)}]: {tc['question'][:40]}...")
            result = self.evaluate_single(tc, n_results)
            results.append(result)
        
        # Compute aggregate metrics
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_keyword_cov = sum(r.keyword_coverage for r in results) / len(results)
        
        # F1 score
        if avg_precision + avg_recall > 0:
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        # LLM metrics (if available)
        avg_faithfulness = 0.0
        avg_relevancy = 0.0
        if self.llm:
            avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
            avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        
        return {
            "individual_results": [r.to_dict() for r in results],
            "aggregate_metrics": {
                "mean_context_precision": avg_precision,
                "mean_context_recall": avg_recall,
                "mean_keyword_coverage": avg_keyword_cov,
                "f1_score": f1,
                "mean_faithfulness": avg_faithfulness,
                "mean_answer_relevancy": avg_relevancy,
                "num_test_cases": len(test_cases),
                "llm_enabled": self.llm is not None
            }
        }
