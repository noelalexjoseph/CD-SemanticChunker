"""
Unified Construction RAG Pipeline.

This module provides a high-level interface that combines all components:
- Document parsing with IBM Docling
- DBSCAN spatial clustering
- LLM summarization
- Vector storage and retrieval

Example:
    >>> from construction_rag import ConstructionRAGPipeline
    >>> pipeline = ConstructionRAGPipeline()
    >>> pipeline.process("drawing.jpg")
    >>> results = pipeline.query("Where is the door schedule?")
"""

import os
import time
from typing import List, Dict, Optional, Union
from pathlib import Path

from .models import Chunk, ProcessingResult, QueryResult
from .chunker import ConstructionDrawingChunker
from .rag import ConstructionDrawingRAG
from .llm import OpenRouterLLM, generate_summaries_for_chunks


class ConstructionRAGPipeline:
    """
    Unified pipeline for construction drawing RAG.
    
    Combines document parsing, chunking, summarization, and retrieval
    into a simple, easy-to-use interface.
    
    Example:
        >>> pipeline = ConstructionRAGPipeline()
        >>> 
        >>> # Process drawings
        >>> pipeline.process("floor_plan.jpg")
        >>> pipeline.process_batch(["plan1.jpg", "plan2.jpg"])
        >>> 
        >>> # Query the indexed content
        >>> results = pipeline.query("door schedule")
        >>> answer = pipeline.ask("What fire rating do the doors have?")
    """
    
    def __init__(
        self,
        persist_directory: str = "./construction_rag_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        enable_summaries: bool = True,
        cluster_eps: float = 0.02,
        cluster_min_samples: int = 2,
        title_block_threshold: float = 0.85
    ):
        """
        Initialize the pipeline.
        
        Args:
            persist_directory: Directory for vector database persistence
            embedding_model: Sentence transformer model for embeddings
            llm_model: LLM model for summaries and answers (via OpenRouter)
            llm_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            enable_summaries: Whether to generate LLM summaries for chunks
            cluster_eps: DBSCAN epsilon for text clustering
            cluster_min_samples: DBSCAN minimum samples for clustering
            title_block_threshold: X position threshold for title block detection
        """
        self.persist_directory = persist_directory
        self.enable_summaries = enable_summaries
        
        # Initialize chunker
        self.chunker = ConstructionDrawingChunker(
            title_block_threshold=title_block_threshold,
            text_cluster_eps=cluster_eps,
            text_cluster_min_samples=cluster_min_samples
        )
        
        # Initialize RAG
        self.rag = ConstructionDrawingRAG(
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        # Initialize LLM (optional)
        self.llm = None
        if enable_summaries:
            try:
                self.llm = OpenRouterLLM(model=llm_model, api_key=llm_api_key)
            except (ValueError, ImportError) as e:
                print(f"Warning: LLM not available ({e}). Summaries will be disabled.")
                self.enable_summaries = False
    
    def process(
        self,
        image_path: Union[str, Path],
        generate_summaries: Optional[bool] = None
    ) -> ProcessingResult:
        """
        Process a single construction drawing.
        
        Extracts chunks using Docling, optionally generates summaries,
        and indexes the content for retrieval.
        
        Args:
            image_path: Path to the construction drawing image
            generate_summaries: Override for summary generation (None uses default)
        
        Returns:
            ProcessingResult with extracted chunks and metadata
        """
        image_path = str(image_path)
        start_time = time.time()
        
        try:
            # Extract chunks
            chunks = self.chunker.process_image(image_path)
            
            # Generate summaries if enabled
            should_summarize = generate_summaries if generate_summaries is not None else self.enable_summaries
            if should_summarize and self.llm and chunks:
                chunk_dicts = [c.to_dict() for c in chunks]
                chunk_dicts = generate_summaries_for_chunks(chunk_dicts, self.llm, verbose=False)
                chunks = [Chunk.from_dict(cd) for cd in chunk_dicts]
            
            # Index chunks
            if chunks:
                self.rag.add_chunks(chunks)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                source_image=os.path.basename(image_path),
                chunks=chunks,
                processing_time=processing_time,
                success=True
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                source_image=os.path.basename(image_path),
                chunks=[],
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        generate_summaries: Optional[bool] = None,
        verbose: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple construction drawings.
        
        Args:
            image_paths: List of paths to construction drawing images
            generate_summaries: Override for summary generation
            verbose: Whether to print progress
        
        Returns:
            List of ProcessingResult objects
        """
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths, 1):
            if verbose:
                print(f"Processing [{i}/{total}]: {os.path.basename(str(path))}")
            
            result = self.process(path, generate_summaries)
            results.append(result)
            
            if verbose:
                status = "OK" if result.success else f"FAILED: {result.error}"
                print(f"  {len(result.chunks)} chunks, {result.processing_time:.1f}s - {status}")
        
        return results
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_type: Optional[str] = None
    ) -> List[QueryResult]:
        """
        Query the indexed content for relevant chunks.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            filter_type: Optional filter by chunk type ('table', 'text', 'viewport', etc.)
        
        Returns:
            List of QueryResult objects ranked by relevance
        """
        return self.rag.query(query_text, n_results=n_results, filter_type=filter_type)
    
    def ask(
        self,
        question: str,
        n_context: int = 5
    ) -> str:
        """
        Ask a question and get an LLM-generated answer.
        
        Retrieves relevant chunks and uses the LLM to generate
        an answer grounded in the retrieved context.
        
        Args:
            question: Natural language question
            n_context: Number of context chunks to retrieve
        
        Returns:
            Generated answer string
        
        Raises:
            ValueError: If LLM is not available
        """
        if not self.llm:
            raise ValueError("LLM not available. Initialize pipeline with enable_summaries=True and valid API key.")
        
        # Retrieve relevant chunks
        results = self.query(question, n_results=n_context)
        
        if not results:
            return "No relevant content found in the indexed documents."
        
        # Get context from results
        contexts = [r.content for r in results]
        
        # Generate answer
        return self.llm.answer_query(question, contexts)
    
    def get_stats(self) -> Dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dict with indexed chunks count, chunk type distribution, and model info
        """
        rag_stats = self.rag.get_stats()
        
        return {
            **rag_stats,
            "llm_enabled": self.llm is not None,
            "llm_model": self.llm.model if self.llm else None,
            "persist_directory": self.persist_directory
        }
    
    def clear(self) -> None:
        """Clear all indexed content."""
        self.rag.clear()
        self.chunker.reset_counter()
