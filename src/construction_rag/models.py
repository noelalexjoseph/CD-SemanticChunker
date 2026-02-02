"""
Data models for Construction RAG library.

This module contains the core data classes used throughout the library
for representing document chunks and their metadata.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple, List


@dataclass
class BoundingBox:
    """
    Normalized bounding box (0-1 range).
    
    Represents a rectangular region on a document page with coordinates
    normalized to the page dimensions.
    
    Attributes:
        x1: Left edge (0-1)
        y1: Top edge (0-1)
        x2: Right edge (0-1)
        y2: Bottom edge (0-1)
    """
    x1: float  # left
    y1: float  # top
    x2: float  # right
    y2: float  # bottom
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    @classmethod
    def from_dict(cls, data: dict) -> "BoundingBox":
        """Create BoundingBox from dictionary."""
        return cls(
            x1=data.get("x1", 0),
            y1=data.get("y1", 0),
            x2=data.get("x2", 1),
            y2=data.get("y2", 1)
        )
    
    @classmethod
    def from_pixels(cls, bbox: List[float], img_width: int, img_height: int) -> "BoundingBox":
        """
        Create BoundingBox from pixel coordinates.
        
        Args:
            bbox: List of [x1, y1, x2, y2] in pixels
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Normalized BoundingBox
        """
        return cls(
            x1=bbox[0] / img_width,
            y1=bbox[1] / img_height,
            x2=bbox[2] / img_width,
            y2=bbox[3] / img_height
        )


@dataclass
class Chunk:
    """
    A semantic chunk extracted from a construction drawing.
    
    Represents a meaningful region of a document that can be indexed
    and retrieved for RAG applications.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        chunk_type: Type of chunk ('text', 'table', 'viewport', 'title_block', 'notes')
        content: Extracted text content
        bbox: Bounding box of the chunk region
        confidence: Detection confidence score (0-1)
        metadata: Additional metadata (source_image, page_title, etc.)
        summary: Optional LLM-generated summary
    """
    chunk_id: str
    chunk_type: str
    content: str
    bbox: BoundingBox
    confidence: float
    metadata: Dict = field(default_factory=dict)
    summary: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "metadata": self.metadata,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create Chunk from dictionary."""
        bbox_data = data.get("bbox", {})
        if isinstance(bbox_data, dict):
            bbox = BoundingBox.from_dict(bbox_data)
        else:
            bbox = BoundingBox(0, 0, 1, 1)
        
        return cls(
            chunk_id=data["chunk_id"],
            chunk_type=data["chunk_type"],
            content=data["content"],
            bbox=bbox,
            confidence=data.get("confidence", 0.9),
            metadata=data.get("metadata", {}),
            summary=data.get("summary")
        )


@dataclass
class ProcessingResult:
    """
    Result of processing a single document.
    
    Attributes:
        source_image: Path or name of the source image
        chunks: List of extracted chunks
        processing_time: Time taken to process in seconds
        success: Whether processing was successful
        error: Error message if processing failed
    """
    source_image: str
    chunks: List[Chunk]
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "source_image": self.source_image,
            "chunks": [c.to_dict() for c in self.chunks],
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error
        }


@dataclass
class QueryResult:
    """
    Result of a RAG query.
    
    Attributes:
        chunk_id: ID of the matched chunk
        content: Content of the matched chunk
        metadata: Chunk metadata
        distance: Vector distance (lower is more similar)
        relevance_score: Similarity score (higher is more similar)
    """
    chunk_id: str
    content: str
    metadata: Dict
    distance: float
    relevance_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "distance": self.distance,
            "relevance_score": self.relevance_score
        }
