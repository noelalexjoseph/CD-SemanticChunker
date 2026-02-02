"""
Construction Drawing Chunker.

This module provides the core functionality for processing construction drawings
through document layout detection and spatial clustering to create semantic chunks.

Key Features:
- IBM Docling integration for layout detection
- DBSCAN spatial clustering for text grouping
- Position-based title block and notes detection
"""

import os
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import DBSCAN

from .models import BoundingBox, Chunk
from .utils import merge_bboxes, classify_chunk_by_position


def cluster_text_blocks(
    text_blocks: List[dict],
    eps: float = 0.02,
    min_samples: int = 2
) -> List[List[dict]]:
    """
    Cluster granular text blocks into semantic regions using DBSCAN.
    
    Docling outputs many small text blocks. This function groups nearby
    blocks into larger semantic regions (e.g., "General Notes", "Specifications").
    
    Args:
        text_blocks: List of text blocks with 'bbox' and 'text' keys
        eps: Maximum distance between samples in a cluster (normalized coordinates).
             Default 0.02 (~2% of image dimensions) groups adjacent elements.
        min_samples: Minimum samples to form a cluster
    
    Returns:
        List of clusters, where each cluster is a list of text blocks
    """
    if len(text_blocks) < min_samples:
        return [text_blocks] if text_blocks else []
    
    # Extract center points for clustering
    centers = []
    valid_blocks = []
    
    for block in text_blocks:
        bbox = block.get('bbox', {})
        if isinstance(bbox, dict) and 'x1' in bbox:
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2
            centers.append([center_x, center_y])
            valid_blocks.append(block)
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append([center_x, center_y])
            valid_blocks.append(block)
    
    if len(centers) < min_samples:
        return [valid_blocks] if valid_blocks else []
    
    # Run DBSCAN clustering
    centers_array = np.array(centers)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers_array)
    
    # Group blocks by cluster label
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:
            # Noise points become their own clusters
            label = f"noise_{i}"
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_blocks[i])
    
    return list(clusters.values())


class ConstructionDrawingChunker:
    """
    Main class for chunking construction drawings.
    
    Processes Docling output and creates structured chunks for RAG retrieval.
    Uses DBSCAN clustering to group granular text blocks into semantic regions.
    
    Example:
        >>> chunker = ConstructionDrawingChunker()
        >>> chunks = chunker.process_image("drawing.jpg")
        >>> print(f"Found {len(chunks)} chunks")
    """
    
    def __init__(
        self,
        title_block_threshold: float = 0.85,
        text_cluster_eps: float = 0.02,
        text_cluster_min_samples: int = 2
    ):
        """
        Initialize the chunker.
        
        Args:
            title_block_threshold: X position threshold for title block detection.
                                   Elements with center_x > threshold are classified
                                   as title block content. Default 0.85 (rightmost 15%).
            text_cluster_eps: DBSCAN epsilon for text clustering. Default 0.02
                              (~2% of image width/height, groups adjacent elements).
            text_cluster_min_samples: DBSCAN min_samples for text clustering.
                                      Default 2 (allows pairs and larger groups).
        """
        self.title_block_threshold = title_block_threshold
        self.text_cluster_eps = text_cluster_eps
        self.text_cluster_min_samples = text_cluster_min_samples
        self.chunk_counter = 0
    
    def _generate_chunk_id(self, prefix: str) -> str:
        """Generate unique chunk ID."""
        self.chunk_counter += 1
        return f"{prefix}_{self.chunk_counter:04d}"
    
    def reset_counter(self) -> None:
        """Reset the chunk counter (useful when processing new document sets)."""
        self.chunk_counter = 0
    
    def process_docling_output(
        self,
        docling_result: dict,
        image_name: str
    ) -> List[Chunk]:
        """
        Process Docling output and create chunks.
        
        Args:
            docling_result: Docling detection result with 'texts', 'tables', 'pictures' keys
            image_name: Name of the source image
        
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Extract different element types from Docling output
        texts = docling_result.get('texts', [])
        tables = docling_result.get('tables', [])
        pictures = docling_result.get('pictures', [])
        
        # 1. Process tables (use directly - 85.7% accuracy from evaluation)
        for i, table in enumerate(tables):
            bbox_data = table.get('bbox', [0, 0, 1, 1])
            if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                bbox = BoundingBox(bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3])
            elif isinstance(bbox_data, dict):
                bbox = BoundingBox.from_dict(bbox_data)
            else:
                bbox = BoundingBox(0, 0, 1, 1)
            
            chunk = Chunk(
                chunk_id=self._generate_chunk_id("table"),
                chunk_type="table",
                content=table.get('text', table.get('content', '')),
                bbox=bbox,
                confidence=table.get('confidence', 0.857),
                metadata={
                    "source_image": image_name,
                    "element_index": i,
                    "rows": table.get('rows', 0),
                    "cols": table.get('cols', 0)
                }
            )
            chunks.append(chunk)
        
        # 2. Process pictures/figures (viewports)
        for i, picture in enumerate(pictures):
            bbox_data = picture.get('bbox', [0, 0, 1, 1])
            if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                bbox = BoundingBox(bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3])
            elif isinstance(bbox_data, dict):
                bbox = BoundingBox.from_dict(bbox_data)
            else:
                bbox = BoundingBox(0, 0, 1, 1)
            
            # Classify as viewport or figure based on size
            chunk_type = "viewport" if bbox.area > 0.1 else "figure"
            
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(chunk_type),
                chunk_type=chunk_type,
                content=picture.get('caption', f"Drawing viewport {i+1}"),
                bbox=bbox,
                confidence=picture.get('confidence', 0.475),
                metadata={
                    "source_image": image_name,
                    "element_index": i,
                    "area": bbox.area
                }
            )
            chunks.append(chunk)
        
        # 3. Cluster text blocks using DBSCAN
        if texts:
            # Convert text elements to clustering format
            text_blocks = []
            for t in texts:
                bbox_data = t.get('bbox', [0, 0, 1, 1])
                if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                    text_blocks.append({
                        'text': t.get('text', ''),
                        'bbox': {
                            'x1': bbox_data[0],
                            'y1': bbox_data[1],
                            'x2': bbox_data[2],
                            'y2': bbox_data[3]
                        }
                    })
                elif isinstance(bbox_data, dict):
                    text_blocks.append({
                        'text': t.get('text', ''),
                        'bbox': bbox_data
                    })
            
            # Cluster nearby text blocks
            clusters = cluster_text_blocks(
                text_blocks,
                eps=self.text_cluster_eps,
                min_samples=self.text_cluster_min_samples
            )
            
            # Create chunk for each cluster
            for cluster in clusters:
                if not cluster:
                    continue
                
                # Merge text content
                content = "\n".join([b.get('text', '') for b in cluster])
                
                # Merge bounding boxes
                boxes = []
                for b in cluster:
                    bbox = b.get('bbox', {})
                    if isinstance(bbox, dict):
                        boxes.append(BoundingBox(
                            bbox.get('x1', 0), bbox.get('y1', 0),
                            bbox.get('x2', 1), bbox.get('y2', 1)
                        ))
                
                merged_bbox = merge_bboxes(boxes) if boxes else BoundingBox(0, 0, 1, 1)
                
                # Classify text region type based on position
                chunk_type = classify_chunk_by_position(
                    merged_bbox,
                    title_block_threshold=self.title_block_threshold
                )
                
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(chunk_type),
                    chunk_type=chunk_type,
                    content=content,
                    bbox=merged_bbox,
                    confidence=0.8,
                    metadata={
                        "source_image": image_name,
                        "num_blocks": len(cluster),
                        "cluster_area": merged_bbox.area
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_image(self, image_path: str) -> List[Chunk]:
        """
        Process a single construction drawing image.
        
        Runs Docling on the image and extracts structured chunks.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of Chunk objects
        """
        from docling.document_converter import DocumentConverter
        
        # Run Docling
        converter = DocumentConverter()
        result = converter.convert(image_path)
        
        # Extract elements from Docling result
        docling_output = {
            'texts': [],
            'tables': [],
            'pictures': []
        }
        
        # Process Docling result
        if hasattr(result, 'document'):
            doc = result.document
            
            # Get text blocks
            if hasattr(doc, 'texts') and doc.texts:
                for text in doc.texts:
                    bbox = [0, 0, 1, 1]
                    if hasattr(text, 'prov') and text.prov:
                        prov = text.prov[0] if isinstance(text.prov, list) else text.prov
                        if hasattr(prov, 'bbox'):
                            b = prov.bbox
                            bbox = [
                                getattr(b, 'l', 0),
                                getattr(b, 't', 0),
                                getattr(b, 'r', 1),
                                getattr(b, 'b', 1)
                            ]
                    docling_output['texts'].append({
                        'text': text.text if hasattr(text, 'text') else str(text),
                        'bbox': bbox
                    })
            
            # Get tables
            if hasattr(doc, 'tables') and doc.tables:
                for table in doc.tables:
                    bbox = [0, 0, 1, 1]
                    if hasattr(table, 'prov') and table.prov:
                        prov = table.prov[0] if isinstance(table.prov, list) else table.prov
                        if hasattr(prov, 'bbox'):
                            b = prov.bbox
                            bbox = [
                                getattr(b, 'l', 0),
                                getattr(b, 't', 0),
                                getattr(b, 'r', 1),
                                getattr(b, 'b', 1)
                            ]
                    content = ""
                    if hasattr(table, 'export_to_markdown'):
                        content = table.export_to_markdown()
                    elif hasattr(table, 'export_to_text'):
                        content = table.export_to_text()
                    else:
                        content = str(table)
                    docling_output['tables'].append({
                        'text': content,
                        'bbox': bbox
                    })
            
            # Get pictures
            if hasattr(doc, 'pictures') and doc.pictures:
                for pic in doc.pictures:
                    bbox = [0, 0, 1, 1]
                    if hasattr(pic, 'prov') and pic.prov:
                        prov = pic.prov[0] if isinstance(pic.prov, list) else pic.prov
                        if hasattr(prov, 'bbox'):
                            b = prov.bbox
                            bbox = [
                                getattr(b, 'l', 0),
                                getattr(b, 't', 0),
                                getattr(b, 'r', 1),
                                getattr(b, 'b', 1)
                            ]
                    docling_output['pictures'].append({
                        'caption': str(pic),
                        'bbox': bbox
                    })
        
        # Process and create chunks
        image_name = os.path.basename(image_path)
        return self.process_docling_output(docling_output, image_name)
