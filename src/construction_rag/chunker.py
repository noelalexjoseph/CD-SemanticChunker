"""
Construction Drawing Chunker - Two-Step Pipeline.

This module implements a two-step pipeline matching the thesis experiments:
1. Raw extraction: Extract texts, tables, pictures from Docling with pixel coords
2. Clustering: DBSCAN clustering, bbox merge, position classification

Key Features:
- IBM Docling integration for layout detection
- Y-coordinate fix (PDF coords → image coords)
- DBSCAN spatial clustering for text grouping
- Position-based title block and notes detection
- Bounding box normalization to 0-1 range for visualization
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image

from .models import BoundingBox, Chunk
from .utils import merge_bboxes, classify_chunk_by_position


@dataclass
class RawChunk:
    """A raw chunk extracted from Docling (pixel coordinates)."""
    chunk_type: str  # 'text', 'table', 'viewport'
    content: str
    bbox: Dict  # Pixel coordinates: {x1, y1, x2, y2}
    confidence: float
    source_image: str
    element_index: int
    
    def to_dict(self) -> dict:
        return asdict(self)


class ConstructionDrawingChunker:
    """
    Two-step chunker for construction drawings.
    
    Matches the thesis experiment pipeline:
    - Step 1: Extract raw chunks from Docling (pixel coords, Y-fixed)
    - Step 2: Cluster text chunks, merge bboxes, normalize for visualization
    
    Example:
        >>> chunker = ConstructionDrawingChunker()
        >>> raw_chunks, img_w, img_h = chunker.extract_raw_chunks("drawing.jpg")
        >>> chunks = chunker.cluster_and_finalize(raw_chunks, img_w, img_h)
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
            title_block_threshold: X position threshold for title block detection (0.85 = rightmost 15%)
            text_cluster_eps: DBSCAN epsilon for text clustering (0.02 = ~2% of image dimensions)
            text_cluster_min_samples: DBSCAN min_samples for clustering
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
        """Reset the chunk counter."""
        self.chunk_counter = 0
    
    # =========================================================================
    # STEP 1: Raw Extraction (like full_pipeline.py)
    # =========================================================================
    
    def extract_raw_chunks(self, image_path: str) -> Tuple[List[RawChunk], int, int]:
        """
        Step 1: Extract raw chunks from Docling.
        
        This matches the experiment's full_pipeline.py approach:
        - Extracts texts, tables, pictures with their bounding boxes
        - Stores PIXEL coordinates (not normalized)
        - Fixes Y-coordinate inversion (PDF coords → image coords)
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Tuple of (raw_chunks, img_width, img_height)
        """
        from docling.document_converter import DocumentConverter
        
        # Get image dimensions for Y-coordinate fix
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Run Docling
        converter = DocumentConverter()
        result = converter.convert(image_path)
        doc = result.document
        
        image_name = os.path.basename(image_path)
        raw_chunks = []
        element_idx = 0
        
        # Extract text items
        if hasattr(doc, 'texts') and doc.texts:
            for text_item in doc.texts:
                # Extract content
                content = ""
                if hasattr(text_item, 'text'):
                    content = text_item.text
                elif hasattr(text_item, 'export_to_text'):
                    content = text_item.export_to_text()
                else:
                    content = str(text_item)
                
                # Skip empty content
                if not content or not content.strip():
                    continue
                
                # Extract bbox and FIX Y-coordinates
                # Docling uses PDF coords (y=0 at bottom), we need image coords (y=0 at top)
                bbox = {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height}
                if hasattr(text_item, 'prov') and text_item.prov:
                    prov = text_item.prov[0] if isinstance(text_item.prov, list) else text_item.prov
                    if hasattr(prov, 'bbox'):
                        b = prov.bbox
                        # Get raw PDF coordinates
                        pdf_l = getattr(b, 'l', 0)
                        pdf_t = getattr(b, 't', 0)  # PDF top (larger Y in PDF coords)
                        pdf_r = getattr(b, 'r', img_width)
                        pdf_b = getattr(b, 'b', img_height)  # PDF bottom (smaller Y in PDF coords)
                        
                        # Convert PDF coords to image coords (invert Y axis)
                        # In PDF: y increases upward, t > b for a box
                        # In image: y increases downward, y1 < y2 for a box
                        bbox = {
                            'x1': pdf_l,
                            'y1': img_height - pdf_t,  # PDF top → image top (invert)
                            'x2': pdf_r,
                            'y2': img_height - pdf_b   # PDF bottom → image bottom (invert)
                        }
                        
                        # Ensure y1 < y2 (in case PDF coords were already image-like)
                        if bbox['y1'] > bbox['y2']:
                            bbox['y1'], bbox['y2'] = bbox['y2'], bbox['y1']
                
                raw_chunks.append(RawChunk(
                    chunk_type='text',
                    content=content.strip(),
                    bbox=bbox,
                    confidence=0.9,
                    source_image=image_name,
                    element_index=element_idx
                ))
                element_idx += 1
        
        # Extract tables
        if hasattr(doc, 'tables') and doc.tables:
            for table in doc.tables:
                # Extract content
                content = ""
                if hasattr(table, 'export_to_markdown'):
                    content = table.export_to_markdown()
                elif hasattr(table, 'export_to_text'):
                    content = table.export_to_text()
                else:
                    content = str(table)
                
                if not content.strip():
                    continue
                
                # Extract bbox with Y-fix
                bbox = {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height}
                if hasattr(table, 'prov') and table.prov:
                    prov = table.prov[0] if isinstance(table.prov, list) else table.prov
                    if hasattr(prov, 'bbox'):
                        b = prov.bbox
                        pdf_l = getattr(b, 'l', 0)
                        pdf_t = getattr(b, 't', 0)
                        pdf_r = getattr(b, 'r', img_width)
                        pdf_b = getattr(b, 'b', img_height)
                        
                        bbox = {
                            'x1': pdf_l,
                            'y1': img_height - pdf_t,
                            'x2': pdf_r,
                            'y2': img_height - pdf_b
                        }
                        if bbox['y1'] > bbox['y2']:
                            bbox['y1'], bbox['y2'] = bbox['y2'], bbox['y1']
                
                raw_chunks.append(RawChunk(
                    chunk_type='table',
                    content=content.strip(),
                    bbox=bbox,
                    confidence=0.857,  # From experiment results
                    source_image=image_name,
                    element_index=element_idx
                ))
                element_idx += 1
        
        # Extract pictures/viewports
        if hasattr(doc, 'pictures') and doc.pictures:
            for i, picture in enumerate(doc.pictures):
                # Extract content
                content = f"Drawing viewport/figure {i+1}"
                if hasattr(picture, 'caption') and picture.caption:
                    content = picture.caption
                elif hasattr(picture, 'text') and picture.text:
                    content = picture.text
                
                # Extract bbox with Y-fix
                bbox = {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height}
                if hasattr(picture, 'prov') and picture.prov:
                    prov = picture.prov[0] if isinstance(picture.prov, list) else picture.prov
                    if hasattr(prov, 'bbox'):
                        b = prov.bbox
                        pdf_l = getattr(b, 'l', 0)
                        pdf_t = getattr(b, 't', 0)
                        pdf_r = getattr(b, 'r', img_width)
                        pdf_b = getattr(b, 'b', img_height)
                        
                        bbox = {
                            'x1': pdf_l,
                            'y1': img_height - pdf_t,
                            'x2': pdf_r,
                            'y2': img_height - pdf_b
                        }
                        if bbox['y1'] > bbox['y2']:
                            bbox['y1'], bbox['y2'] = bbox['y2'], bbox['y1']
                
                raw_chunks.append(RawChunk(
                    chunk_type='viewport',
                    content=content,
                    bbox=bbox,
                    confidence=0.475,  # From experiment results
                    source_image=image_name,
                    element_index=element_idx
                ))
                element_idx += 1
        
        return raw_chunks, img_width, img_height
    
    # =========================================================================
    # STEP 2: Clustering and Finalization (like cluster_and_summarize.py)
    # =========================================================================
    
    def cluster_and_finalize(
        self,
        raw_chunks: List[RawChunk],
        img_width: int,
        img_height: int
    ) -> List[Chunk]:
        """
        Step 2: Cluster text chunks and create final Chunk objects.
        
        This matches the experiment's cluster_and_summarize.py approach:
        - Separates text/table/viewport chunks
        - Clusters text chunks using DBSCAN (normalized coords for clustering only)
        - Merges bboxes using min/max of pixel coords
        - Normalizes final bboxes to 0-1 range for visualization
        - Classifies chunks by position (title_block, notes, text)
        
        Args:
            raw_chunks: List of RawChunk objects from Step 1
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of Chunk objects with normalized bounding boxes
        """
        # Separate by type
        text_chunks = [c for c in raw_chunks if c.chunk_type == 'text']
        table_chunks = [c for c in raw_chunks if c.chunk_type == 'table']
        viewport_chunks = [c for c in raw_chunks if c.chunk_type == 'viewport']
        
        final_chunks = []
        
        # --- Process text chunks with DBSCAN clustering ---
        if text_chunks:
            clustered_text = self._cluster_text_chunks(text_chunks, img_width, img_height)
            
            for cluster_data in clustered_text:
                # Normalize bbox to 0-1 range for visualization
                bbox = cluster_data['bbox']
                normalized_bbox = BoundingBox(
                    bbox['x1'] / img_width,
                    bbox['y1'] / img_height,
                    bbox['x2'] / img_width,
                    bbox['y2'] / img_height
                )
                
                # Classify by position
                chunk_type = classify_chunk_by_position(
                    normalized_bbox,
                    title_block_threshold=self.title_block_threshold
                )
                
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(chunk_type),
                    chunk_type=chunk_type,
                    content=cluster_data['content'],
                    bbox=normalized_bbox,
                    confidence=cluster_data['confidence'],
                    metadata={
                        'source_image': cluster_data['source_image'],
                        'num_blocks': cluster_data['num_blocks'],
                        'cluster_area': normalized_bbox.area
                    }
                )
                final_chunks.append(chunk)
        
        # --- Add tables (no clustering, just normalize bbox) ---
        for tc in table_chunks:
            normalized_bbox = BoundingBox(
                tc.bbox['x1'] / img_width,
                tc.bbox['y1'] / img_height,
                tc.bbox['x2'] / img_width,
                tc.bbox['y2'] / img_height
            )
            
            chunk = Chunk(
                chunk_id=self._generate_chunk_id('table'),
                chunk_type='table',
                content=tc.content,
                bbox=normalized_bbox,
                confidence=tc.confidence,
                metadata={
                    'source_image': tc.source_image,
                    'element_index': tc.element_index
                }
            )
            final_chunks.append(chunk)
        
        # --- Add viewports (no clustering, just normalize bbox) ---
        for vc in viewport_chunks:
            normalized_bbox = BoundingBox(
                vc.bbox['x1'] / img_width,
                vc.bbox['y1'] / img_height,
                vc.bbox['x2'] / img_width,
                vc.bbox['y2'] / img_height
            )
            
            # Classify as viewport or figure based on size
            chunk_type = 'viewport' if normalized_bbox.area > 0.1 else 'figure'
            
            chunk = Chunk(
                chunk_id=self._generate_chunk_id(chunk_type),
                chunk_type=chunk_type,
                content=vc.content,
                bbox=normalized_bbox,
                confidence=vc.confidence,
                metadata={
                    'source_image': vc.source_image,
                    'element_index': vc.element_index,
                    'area': normalized_bbox.area
                }
            )
            final_chunks.append(chunk)
        
        return final_chunks
    
    def _cluster_text_chunks(
        self,
        text_chunks: List[RawChunk],
        img_width: int,
        img_height: int
    ) -> List[Dict]:
        """
        Cluster text chunks using DBSCAN (exactly like experiments).
        
        Args:
            text_chunks: List of text RawChunk objects
            img_width: Image width for normalization
            img_height: Image height for normalization
        
        Returns:
            List of cluster dictionaries with merged content and bbox
        """
        if len(text_chunks) < self.text_cluster_min_samples:
            # Not enough to cluster, return as-is
            return [{
                'content': tc.content,
                'bbox': tc.bbox,
                'confidence': tc.confidence,
                'source_image': tc.source_image,
                'num_blocks': 1
            } for tc in text_chunks]
        
        # Extract centroids (pixel coordinates)
        coords = []
        for tc in text_chunks:
            bbox = tc.bbox
            cx = (bbox['x1'] + bbox['x2']) / 2
            cy = (bbox['y1'] + bbox['y2']) / 2
            coords.append([cx, cy])
        
        coords = np.array(coords)
        
        # Normalize coordinates to 0-1 range FOR CLUSTERING ONLY
        # (exactly like cluster_and_summarize.py lines 111-121)
        min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
        min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
        
        range_x = max(max_x - min_x, 1)  # Avoid division by zero
        range_y = max(max_y - min_y, 1)
        
        normalized_coords = np.copy(coords)
        normalized_coords[:, 0] = (coords[:, 0] - min_x) / range_x
        normalized_coords[:, 1] = (coords[:, 1] - min_y) / range_y
        
        # Run DBSCAN
        db = DBSCAN(eps=self.text_cluster_eps, min_samples=self.text_cluster_min_samples)
        labels = db.fit_predict(normalized_coords)
        
        # Group chunks by cluster label
        # IMPORTANT: Treat each noise point (label=-1) as its own cluster
        # (matching chunking_library.py behavior)
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                # Noise points become their own individual clusters
                label = f"noise_{i}"
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text_chunks[i])
        
        # Create merged cluster data
        clustered_results = []
        source_image = text_chunks[0].source_image if text_chunks else ''
        
        for label, cluster_chunks in clusters.items():
            # Merge content
            merged_content = " ".join([c.content for c in cluster_chunks if c.content.strip()])
            
            # Merge bboxes using min/max of PIXEL coordinates
            # (exactly like cluster_and_summarize.py lines 139-150)
            all_x1 = [c.bbox['x1'] for c in cluster_chunks]
            all_y1 = [c.bbox['y1'] for c in cluster_chunks]
            all_x2 = [c.bbox['x2'] for c in cluster_chunks]
            all_y2 = [c.bbox['y2'] for c in cluster_chunks]
            
            merged_bbox = {
                'x1': min(all_x1),
                'y1': min(all_y1),
                'x2': max(all_x2),
                'y2': max(all_y2)
            }
            
            # Average confidence
            avg_conf = sum(c.confidence for c in cluster_chunks) / len(cluster_chunks)
            
            clustered_results.append({
                'content': merged_content.strip(),
                'bbox': merged_bbox,
                'confidence': avg_conf,
                'source_image': source_image,
                'num_blocks': len(cluster_chunks)
            })
        
        return clustered_results
    
    # =========================================================================
    # Convenience method (combines both steps)
    # =========================================================================
    
    def process_image(self, image_path: str) -> List[Chunk]:
        """
        Process a single image through the full two-step pipeline.
        
        This is a convenience method that combines:
        - Step 1: extract_raw_chunks()
        - Step 2: cluster_and_finalize()
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of Chunk objects with normalized bounding boxes
        """
        # Step 1: Raw extraction
        raw_chunks, img_width, img_height = self.extract_raw_chunks(image_path)
        
        # Step 2: Cluster and finalize
        chunks = self.cluster_and_finalize(raw_chunks, img_width, img_height)
        
        return chunks
    
    def process_docling_output(
        self,
        docling_data: Dict,
        image_name: str,
        img_width: int = 1000,
        img_height: int = 1000
    ) -> List[Chunk]:
        """
        Process pre-extracted Docling JSON output.
        
        This method allows processing of saved Docling results without
        re-running Docling extraction. Useful for batch processing and caching.
        
        Args:
            docling_data: Dictionary with keys 'texts', 'tables', 'pictures'
                          Each element should have 'content'/'text' and 'bbox' keys.
                          Bbox can be: list [x1,y1,x2,y2], dict {x1,y1,x2,y2}, or
                          normalized 0-1 values.
            image_name: Name of the source image
            img_width: Image width in pixels (for coordinate scaling)
            img_height: Image height in pixels (for coordinate scaling)
        
        Returns:
            List of Chunk objects with normalized bounding boxes
        
        Example:
            >>> chunker = ConstructionDrawingChunker()
            >>> docling_json = {
            ...     'texts': [{'text': 'Door Schedule', 'bbox': [0.1, 0.1, 0.5, 0.2]}],
            ...     'tables': [{'content': 'Table data...', 'bbox': [0.1, 0.3, 0.9, 0.8]}],
            ...     'pictures': []
            ... }
            >>> chunks = chunker.process_docling_output(docling_json, "drawing1.jpg")
        """
        raw_chunks = []
        element_idx = 0
        
        def _parse_bbox(bbox_data, default_pixel=True):
            """Parse bbox from various formats."""
            if bbox_data is None:
                if default_pixel:
                    return {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height}
                return {'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}
            
            # Handle list format [x1, y1, x2, y2]
            if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                x1, y1, x2, y2 = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]
            # Handle dict format {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
            elif isinstance(bbox_data, dict):
                x1 = bbox_data.get('x1', bbox_data.get('l', 0))
                y1 = bbox_data.get('y1', bbox_data.get('t', 0))
                x2 = bbox_data.get('x2', bbox_data.get('r', 1))
                y2 = bbox_data.get('y2', bbox_data.get('b', 1))
            else:
                return {'x1': 0, 'y1': 0, 'x2': img_width, 'y2': img_height}
            
            # Check if already normalized (all values 0-1)
            values = [x1, y1, x2, y2]
            is_normalized = all(0 <= v <= 1 for v in values)
            
            # Convert to pixel coordinates if normalized
            if is_normalized:
                x1 = x1 * img_width
                y1 = y1 * img_height
                x2 = x2 * img_width
                y2 = y2 * img_height
            
            return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        
        # Process texts
        texts = docling_data.get('texts', [])
        for text_item in texts:
            content = text_item.get('text', text_item.get('content', ''))
            if not content or not str(content).strip():
                continue
            
            bbox = _parse_bbox(text_item.get('bbox'))
            
            raw_chunks.append(RawChunk(
                chunk_type='text',
                content=str(content).strip(),
                bbox=bbox,
                confidence=text_item.get('confidence', 0.9),
                source_image=image_name,
                element_index=element_idx
            ))
            element_idx += 1
        
        # Process tables
        tables = docling_data.get('tables', [])
        for table_item in tables:
            content = table_item.get('content', table_item.get('text', ''))
            if not content or not str(content).strip():
                continue
            
            bbox = _parse_bbox(table_item.get('bbox'))
            
            raw_chunks.append(RawChunk(
                chunk_type='table',
                content=str(content).strip(),
                bbox=bbox,
                confidence=table_item.get('confidence', 0.857),
                source_image=image_name,
                element_index=element_idx
            ))
            element_idx += 1
        
        # Process pictures/viewports
        pictures = docling_data.get('pictures', docling_data.get('viewports', []))
        for i, pic_item in enumerate(pictures):
            content = pic_item.get('caption', pic_item.get('content', f'Drawing viewport {i+1}'))
            bbox = _parse_bbox(pic_item.get('bbox'))
            
            raw_chunks.append(RawChunk(
                chunk_type='viewport',
                content=str(content),
                bbox=bbox,
                confidence=pic_item.get('confidence', 0.475),
                source_image=image_name,
                element_index=element_idx
            ))
            element_idx += 1
        
        # Run Step 2: Cluster and finalize
        return self.cluster_and_finalize(raw_chunks, img_width, img_height)
