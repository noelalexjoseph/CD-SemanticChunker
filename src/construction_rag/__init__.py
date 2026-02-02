"""
Construction RAG - Open-source library for construction drawing RAG pipelines.

This library transforms construction drawings into searchable chunks for
Retrieval-Augmented Generation (RAG) systems.

Key Components:
- ConstructionRAGPipeline: Unified high-level interface
- ConstructionDrawingChunker: Document parsing and clustering
- ConstructionDrawingRAG: Vector storage and retrieval
- OpenRouterLLM: LLM integration for summaries and answers

Example:
    >>> from construction_rag import ConstructionRAGPipeline
    >>> 
    >>> # Initialize pipeline
    >>> pipeline = ConstructionRAGPipeline()
    >>> 
    >>> # Process a construction drawing
    >>> result = pipeline.process("floor_plan.jpg")
    >>> print(f"Extracted {len(result.chunks)} chunks")
    >>> 
    >>> # Query the indexed content
    >>> results = pipeline.query("door schedule")
    >>> for r in results:
    ...     print(f"{r.relevance_score:.2f}: {r.content[:50]}...")
    >>> 
    >>> # Ask questions with LLM-generated answers
    >>> answer = pipeline.ask("What is the fire rating for the doors?")
    >>> print(answer)

Developed as part of the Master's thesis:
"Exploring RAG Methods and Techniques as One Source of Truth
for Querying Construction Documents"

Author: Joseph Noel
Institution: RWTH Aachen University
Program: M.Sc. Construction & Robotics
Date: February 2026
"""

__version__ = "0.1.0"
__author__ = "Joseph Noel"

# Core pipeline
from .pipeline import ConstructionRAGPipeline

# Individual components
from .chunker import ConstructionDrawingChunker, cluster_text_blocks
from .rag import ConstructionDrawingRAG
from .llm import OpenRouterLLM, generate_summaries_for_chunks

# Data models
from .models import (
    Chunk,
    BoundingBox,
    ProcessingResult,
    QueryResult
)

# Utilities
from .utils import (
    merge_bboxes,
    bbox_distance,
    classify_chunk_by_position
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Main pipeline
    "ConstructionRAGPipeline",
    
    # Components
    "ConstructionDrawingChunker",
    "ConstructionDrawingRAG",
    "OpenRouterLLM",
    
    # Functions
    "cluster_text_blocks",
    "generate_summaries_for_chunks",
    "merge_bboxes",
    "bbox_distance",
    "classify_chunk_by_position",
    
    # Data models
    "Chunk",
    "BoundingBox",
    "ProcessingResult",
    "QueryResult",
]
