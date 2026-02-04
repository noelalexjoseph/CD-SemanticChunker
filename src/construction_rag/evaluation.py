"""
Batch Evaluation Module for Construction Drawing RAG Pipeline.

This module provides comprehensive evaluation capabilities for statistical
validation of the chunking and RAG pipeline, matching the thesis experiment
methodology.

Features:
- Per-image, per-dataset, and aggregate metrics
- 95% confidence intervals for statistical validation
- Batch processing with checkpointing
- Support for multiple datasets
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ImageResult:
    """Results from processing a single image."""
    image_name: str
    dataset: str
    success: bool
    processing_time: float
    num_raw_chunks: int
    num_final_chunks: int
    num_text: int
    num_tables: int
    num_viewports: int
    error_message: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DatasetStats:
    """Aggregated statistics for a dataset."""
    dataset_name: str
    images_processed: int
    images_failed: int
    total_chunks: int
    mean_chunks_per_image: float
    std_chunks_per_image: float
    mean_processing_time: float
    std_processing_time: float
    mean_text_ratio: float
    mean_table_ratio: float
    mean_viewport_ratio: float
    confidence_interval_95: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['confidence_interval_95'] = list(self.confidence_interval_95)
        return result


class BatchEvaluator:
    """
    Batch evaluation system for construction drawing RAG pipeline.
    
    Provides statistical validation by processing multiple images and
    calculating aggregate metrics with confidence intervals.
    
    Example:
        >>> evaluator = BatchEvaluator(results_dir="./evaluation_results")
        >>> images = ["drawing1.jpg", "drawing2.jpg", "drawing3.jpg"]
        >>> results = evaluator.evaluate_images(images, "test_dataset")
        >>> stats = evaluator.calculate_stats(results)
        >>> print(f"Mean chunks: {stats.mean_chunks_per_image:.1f} ± {stats.std_chunks_per_image:.1f}")
    """
    
    def __init__(
        self,
        results_dir: str = "./evaluation_results",
        checkpoint_enabled: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the batch evaluator.
        
        Args:
            results_dir: Directory to save results and checkpoints
            checkpoint_enabled: Whether to save progress checkpoints
            verbose: Whether to print progress messages
        """
        self.results_dir = results_dir
        self.checkpoint_enabled = checkpoint_enabled
        self.verbose = verbose
        self.results: List[ImageResult] = []
        self.checkpoint_data = {"processed": [], "results": []}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Lazy initialization
        self._chunker = None
    
    def _get_chunker(self):
        """Lazy initialization of chunker."""
        if self._chunker is None:
            from .chunker import ConstructionDrawingChunker
            self._chunker = ConstructionDrawingChunker()
        return self._chunker
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    @property
    def checkpoint_file(self) -> str:
        """Path to checkpoint file."""
        return os.path.join(self.results_dir, "evaluation_checkpoint.json")
    
    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if it exists.
        
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                self._log(f"Loaded checkpoint: {len(self.checkpoint_data['processed'])} images already processed")
                return True
            except (json.JSONDecodeError, IOError):
                return False
        return False
    
    def save_checkpoint(self):
        """Save current progress to checkpoint."""
        if self.checkpoint_enabled:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2)
    
    def clear_checkpoint(self):
        """Remove checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
    
    def process_single_image(
        self,
        image_path: str,
        dataset_name: str = "default"
    ) -> ImageResult:
        """
        Process a single image through the chunking pipeline.
        
        Args:
            image_path: Path to the image file
            dataset_name: Name of the source dataset
        
        Returns:
            ImageResult with processing statistics
        """
        chunker = self._get_chunker()
        image_name = os.path.basename(image_path)
        start_time = time.time()
        
        result = ImageResult(
            image_name=image_name,
            dataset=dataset_name,
            success=False,
            processing_time=0,
            num_raw_chunks=0,
            num_final_chunks=0,
            num_text=0,
            num_tables=0,
            num_viewports=0
        )
        
        try:
            # Step 1: Extract raw chunks
            raw_chunks, img_width, img_height = chunker.extract_raw_chunks(image_path)
            result.num_raw_chunks = len(raw_chunks)
            
            # Count raw types
            result.num_text = sum(1 for c in raw_chunks if c.chunk_type == 'text')
            result.num_tables = sum(1 for c in raw_chunks if c.chunk_type == 'table')
            result.num_viewports = sum(1 for c in raw_chunks if c.chunk_type == 'viewport')
            
            # Step 2: Cluster and finalize
            final_chunks = chunker.cluster_and_finalize(raw_chunks, img_width, img_height)
            result.num_final_chunks = len(final_chunks)
            
            result.success = True
            
        except Exception as e:
            result.error_message = str(e)
        
        result.processing_time = time.time() - start_time
        return result
    
    def evaluate_images(
        self,
        image_paths: List[str],
        dataset_name: str = "default",
        resume: bool = True,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> List[ImageResult]:
        """
        Evaluate multiple images from a dataset.
        
        Args:
            image_paths: List of paths to image files
            dataset_name: Name of the dataset
            resume: Whether to resume from checkpoint
            sample_size: If set, randomly sample this many images
            random_seed: Seed for reproducible sampling
        
        Returns:
            List of ImageResult objects
        """
        self._log(f"\n{'='*60}")
        self._log(f"  EVALUATING: {dataset_name}")
        self._log(f"{'='*60}")
        
        # Apply sampling if requested
        images = list(image_paths)
        if sample_size and len(images) > sample_size:
            random.seed(random_seed)
            images = random.sample(images, sample_size)
        
        self._log(f"  Processing {len(images)} images")
        
        # Load checkpoint and filter already processed
        if resume:
            self.load_checkpoint()
            processed_set = set(self.checkpoint_data.get('processed', []))
            images = [img for img in images if os.path.basename(img) not in processed_set]
            self._log(f"  {len(images)} remaining after checkpoint")
        
        results = []
        
        for i, image_path in enumerate(images):
            image_name = os.path.basename(image_path)
            self._log(f"\n  [{i+1}/{len(images)}] {image_name}")
            
            result = self.process_single_image(image_path, dataset_name)
            results.append(result)
            
            # Update checkpoint
            self.checkpoint_data['processed'].append(image_name)
            self.checkpoint_data['results'].append(result.to_dict())
            
            if result.success:
                self._log(f"      OK: {result.num_raw_chunks} raw -> {result.num_final_chunks} final ({result.processing_time:.1f}s)")
            else:
                self._log(f"      FAILED: {result.error_message[:50]}")
            
            # Save checkpoint periodically
            if (i + 1) % 5 == 0:
                self.save_checkpoint()
        
        # Final checkpoint save
        self.save_checkpoint()
        
        return results
    
    def calculate_stats(self, results: List[ImageResult]) -> DatasetStats:
        """
        Calculate aggregate statistics for a set of results.
        
        Args:
            results: List of ImageResult objects
        
        Returns:
            DatasetStats with aggregate metrics and confidence intervals
        """
        if not results:
            return DatasetStats(
                dataset_name="unknown",
                images_processed=0,
                images_failed=0,
                total_chunks=0,
                mean_chunks_per_image=0,
                std_chunks_per_image=0,
                mean_processing_time=0,
                std_processing_time=0,
                mean_text_ratio=0,
                mean_table_ratio=0,
                mean_viewport_ratio=0
            )
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if not successful:
            return DatasetStats(
                dataset_name=results[0].dataset if results else "unknown",
                images_processed=len(results),
                images_failed=len(failed),
                total_chunks=0,
                mean_chunks_per_image=0,
                std_chunks_per_image=0,
                mean_processing_time=0,
                std_processing_time=0,
                mean_text_ratio=0,
                mean_table_ratio=0,
                mean_viewport_ratio=0
            )
        
        chunks_per_image = [r.num_final_chunks for r in successful]
        times = [r.processing_time for r in successful]
        
        # Calculate type ratios
        text_ratios = []
        table_ratios = []
        viewport_ratios = []
        
        for r in successful:
            total = r.num_final_chunks if r.num_final_chunks > 0 else 1
            # Use raw counts for ratios (before clustering)
            raw_total = r.num_raw_chunks if r.num_raw_chunks > 0 else 1
            text_ratios.append(r.num_text / raw_total)
            table_ratios.append(r.num_tables / raw_total)
            viewport_ratios.append(r.num_viewports / raw_total)
        
        # Calculate 95% confidence interval for mean chunks
        mean_chunks = float(np.mean(chunks_per_image))
        std_chunks = float(np.std(chunks_per_image, ddof=1)) if len(chunks_per_image) > 1 else 0.0
        n = len(chunks_per_image)
        ci_margin = 1.96 * (std_chunks / np.sqrt(n)) if n > 0 else 0
        
        return DatasetStats(
            dataset_name=results[0].dataset,
            images_processed=len(results),
            images_failed=len(failed),
            total_chunks=sum(chunks_per_image),
            mean_chunks_per_image=mean_chunks,
            std_chunks_per_image=std_chunks,
            mean_processing_time=float(np.mean(times)),
            std_processing_time=float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            mean_text_ratio=float(np.mean(text_ratios)),
            mean_table_ratio=float(np.mean(table_ratios)),
            mean_viewport_ratio=float(np.mean(viewport_ratios)),
            confidence_interval_95=(mean_chunks - ci_margin, mean_chunks + ci_margin)
        )
    
    def evaluate_directory(
        self,
        directory: str,
        dataset_name: Optional[str] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
        sample_size: Optional[int] = None,
        resume: bool = True
    ) -> Tuple[List[ImageResult], DatasetStats]:
        """
        Evaluate all images in a directory.
        
        Args:
            directory: Path to directory containing images
            dataset_name: Name for the dataset (defaults to directory name)
            extensions: File extensions to include
            sample_size: If set, randomly sample this many images
            resume: Whether to resume from checkpoint
        
        Returns:
            Tuple of (results list, statistics)
        """
        if dataset_name is None:
            dataset_name = os.path.basename(directory)
        
        # Find all images
        images = []
        for f in os.listdir(directory):
            if f.lower().endswith(extensions):
                images.append(os.path.join(directory, f))
        
        if not images:
            self._log(f"No images found in {directory}")
            return [], DatasetStats(
                dataset_name=dataset_name,
                images_processed=0,
                images_failed=0,
                total_chunks=0,
                mean_chunks_per_image=0,
                std_chunks_per_image=0,
                mean_processing_time=0,
                std_processing_time=0,
                mean_text_ratio=0,
                mean_table_ratio=0,
                mean_viewport_ratio=0
            )
        
        # Run evaluation
        results = self.evaluate_images(
            images,
            dataset_name=dataset_name,
            resume=resume,
            sample_size=sample_size
        )
        
        # Include checkpoint results
        checkpoint_results = [
            ImageResult(**r) for r in self.checkpoint_data.get('results', [])
            if r.get('dataset') == dataset_name
        ]
        
        # Combine avoiding duplicates
        seen = set()
        combined = []
        for r in checkpoint_results + results:
            if r.image_name not in seen:
                seen.add(r.image_name)
                combined.append(r)
        
        stats = self.calculate_stats(combined)
        
        return combined, stats
    
    def save_results(
        self,
        results: Dict[str, List[ImageResult]],
        stats: Dict[str, DatasetStats],
        filename: str = "evaluation_results.json"
    ) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Dict mapping dataset name to results list
            stats: Dict mapping dataset name to statistics
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "datasets_evaluated": len(results),
                "total_images": sum(len(r) for r in results.values())
            },
            "per_dataset_stats": {k: v.to_dict() for k, v in stats.items()},
            "detailed_results": {
                k: [r.to_dict() for r in v] for k, v in results.items()
            }
        }
        
        output_path = os.path.join(self.results_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self._log(f"\nResults saved to: {output_path}")
        
        # Clear checkpoint on successful save
        self.clear_checkpoint()
        
        return output_path
    
    def print_summary(self, stats: DatasetStats):
        """
        Print a formatted summary of dataset statistics.
        
        Args:
            stats: DatasetStats object to summarize
        """
        print(f"\n{'='*50}")
        print(f"  {stats.dataset_name} Summary")
        print(f"{'='*50}")
        print(f"  Processed: {stats.images_processed} ({stats.images_failed} failed)")
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  Mean chunks/image: {stats.mean_chunks_per_image:.1f} ± {stats.std_chunks_per_image:.1f}")
        print(f"  95% CI: [{stats.confidence_interval_95[0]:.1f}, {stats.confidence_interval_95[1]:.1f}]")
        print(f"  Mean processing time: {stats.mean_processing_time:.1f}s")
        print(f"  Type distribution:")
        print(f"    Text: {stats.mean_text_ratio*100:.1f}%")
        print(f"    Tables: {stats.mean_table_ratio*100:.1f}%")
        print(f"    Viewports: {stats.mean_viewport_ratio*100:.1f}%")
