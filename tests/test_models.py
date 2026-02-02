"""Tests for data models."""

import pytest
from construction_rag.models import BoundingBox, Chunk, ProcessingResult, QueryResult


class TestBoundingBox:
    """Tests for BoundingBox class."""
    
    def test_create_bbox(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(0.1, 0.2, 0.5, 0.8)
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.5
        assert bbox.y2 == 0.8
    
    def test_center(self):
        """Test center calculation."""
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        assert bbox.center == (0.5, 0.5)
        
        bbox2 = BoundingBox(0.2, 0.3, 0.6, 0.7)
        assert bbox2.center == (0.4, 0.5)
    
    def test_dimensions(self):
        """Test width and height calculations."""
        bbox = BoundingBox(0.1, 0.2, 0.5, 0.8)
        assert bbox.width == pytest.approx(0.4)
        assert bbox.height == pytest.approx(0.6)
    
    def test_area(self):
        """Test area calculation."""
        bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
        assert bbox.area == pytest.approx(0.25)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
        d = bbox.to_dict()
        assert d == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
        bbox = BoundingBox.from_dict(d)
        assert bbox.x1 == 0.1
        assert bbox.y2 == 0.4
    
    def test_from_dict_defaults(self):
        """Test defaults when creating from incomplete dict."""
        bbox = BoundingBox.from_dict({})
        assert bbox.x1 == 0
        assert bbox.y2 == 1
    
    def test_from_pixels(self):
        """Test creation from pixel coordinates."""
        bbox = BoundingBox.from_pixels([100, 200, 300, 400], 1000, 1000)
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.3
        assert bbox.y2 == 0.4


class TestChunk:
    """Tests for Chunk class."""
    
    def test_create_chunk(self):
        """Test basic chunk creation."""
        bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
        chunk = Chunk(
            chunk_id="text_0001",
            chunk_type="text",
            content="Sample text content",
            bbox=bbox,
            confidence=0.9,
            metadata={"source_image": "test.jpg"}
        )
        assert chunk.chunk_id == "text_0001"
        assert chunk.chunk_type == "text"
        assert chunk.confidence == 0.9
    
    def test_chunk_with_summary(self):
        """Test chunk with summary."""
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        chunk = Chunk(
            chunk_id="table_0001",
            chunk_type="table",
            content="Door | Size | Type",
            bbox=bbox,
            confidence=0.857,
            metadata={},
            summary="Door schedule with dimensions and types"
        )
        assert chunk.summary == "Door schedule with dimensions and types"
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
        chunk = Chunk(
            chunk_id="test_0001",
            chunk_type="text",
            content="Test",
            bbox=bbox,
            confidence=0.8,
            metadata={"key": "value"},
            summary="Summary"
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "test_0001"
        assert d["bbox"]["x1"] == 0.1
        assert d["summary"] == "Summary"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "chunk_id": "text_0001",
            "chunk_type": "text",
            "content": "Test content",
            "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
            "confidence": 0.9,
            "metadata": {"source": "test"},
            "summary": "Test summary"
        }
        chunk = Chunk.from_dict(d)
        assert chunk.chunk_id == "text_0001"
        assert chunk.bbox.x1 == 0.1
        assert chunk.summary == "Test summary"


class TestProcessingResult:
    """Tests for ProcessingResult class."""
    
    def test_successful_result(self):
        """Test successful processing result."""
        bbox = BoundingBox(0, 0, 1, 1)
        chunks = [
            Chunk("c1", "text", "Content", bbox, 0.9, {}),
            Chunk("c2", "table", "Table", bbox, 0.85, {}),
        ]
        result = ProcessingResult(
            source_image="test.jpg",
            chunks=chunks,
            processing_time=2.5,
            success=True
        )
        assert result.success
        assert len(result.chunks) == 2
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed processing result."""
        result = ProcessingResult(
            source_image="bad.jpg",
            chunks=[],
            processing_time=0.1,
            success=False,
            error="File not found"
        )
        assert not result.success
        assert result.error == "File not found"


class TestQueryResult:
    """Tests for QueryResult class."""
    
    def test_query_result(self):
        """Test query result creation."""
        result = QueryResult(
            chunk_id="text_0001",
            content="Relevant content",
            metadata={"chunk_type": "text"},
            distance=0.2,
            relevance_score=0.8
        )
        assert result.chunk_id == "text_0001"
        assert result.relevance_score == 0.8
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = QueryResult(
            chunk_id="c1",
            content="Test",
            metadata={},
            distance=0.3,
            relevance_score=0.7
        )
        d = result.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["relevance_score"] == 0.7
