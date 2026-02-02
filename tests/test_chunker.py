"""Tests for chunker module."""

import pytest
from construction_rag.chunker import (
    ConstructionDrawingChunker,
    cluster_text_blocks
)
from construction_rag.models import BoundingBox


class TestClusterTextBlocks:
    """Tests for DBSCAN text clustering."""
    
    def test_empty_input(self):
        """Test with empty input."""
        result = cluster_text_blocks([])
        assert result == []
    
    def test_single_block(self):
        """Test with single block (below min_samples)."""
        blocks = [{"text": "Single", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}}]
        result = cluster_text_blocks(blocks, min_samples=2)
        assert len(result) == 1
        assert len(result[0]) == 1
    
    def test_nearby_blocks_cluster(self):
        """Test that nearby blocks are clustered together."""
        blocks = [
            {"text": "Block 1", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.15, "y2": 0.15}},
            {"text": "Block 2", "bbox": {"x1": 0.12, "y1": 0.12, "x2": 0.17, "y2": 0.17}},
            {"text": "Block 3", "bbox": {"x1": 0.11, "y1": 0.11, "x2": 0.16, "y2": 0.16}},
        ]
        result = cluster_text_blocks(blocks, eps=0.05, min_samples=2)
        # All blocks should be in one cluster (they're very close)
        assert len(result) == 1
        assert len(result[0]) == 3
    
    def test_distant_blocks_separate(self):
        """Test that distant blocks stay separate."""
        blocks = [
            {"text": "Top left", "bbox": {"x1": 0.0, "y1": 0.0, "x2": 0.1, "y2": 0.1}},
            {"text": "Bottom right", "bbox": {"x1": 0.9, "y1": 0.9, "x2": 1.0, "y2": 1.0}},
        ]
        result = cluster_text_blocks(blocks, eps=0.05, min_samples=1)
        # Each block should be its own cluster
        assert len(result) == 2
    
    def test_list_bbox_format(self):
        """Test with list-format bounding boxes."""
        blocks = [
            {"text": "Block 1", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"text": "Block 2", "bbox": [0.12, 0.12, 0.22, 0.22]},
        ]
        result = cluster_text_blocks(blocks, eps=0.05, min_samples=2)
        assert len(result) >= 1


class TestConstructionDrawingChunker:
    """Tests for ConstructionDrawingChunker class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        chunker = ConstructionDrawingChunker()
        assert chunker.title_block_threshold == 0.85
        assert chunker.text_cluster_eps == 0.02
        assert chunker.text_cluster_min_samples == 2
    
    def test_init_custom_params(self):
        """Test custom parameter initialization."""
        chunker = ConstructionDrawingChunker(
            title_block_threshold=0.9,
            text_cluster_eps=0.05,
            text_cluster_min_samples=3
        )
        assert chunker.title_block_threshold == 0.9
        assert chunker.text_cluster_eps == 0.05
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunker = ConstructionDrawingChunker()
        id1 = chunker._generate_chunk_id("text")
        id2 = chunker._generate_chunk_id("table")
        assert id1 == "text_0001"
        assert id2 == "table_0002"
    
    def test_reset_counter(self):
        """Test counter reset."""
        chunker = ConstructionDrawingChunker()
        chunker._generate_chunk_id("test")
        chunker._generate_chunk_id("test")
        chunker.reset_counter()
        id1 = chunker._generate_chunk_id("text")
        assert id1 == "text_0001"
    
    def test_process_docling_output_tables(self):
        """Test processing tables from Docling output."""
        chunker = ConstructionDrawingChunker()
        docling_output = {
            "texts": [],
            "tables": [
                {"text": "Door | Size | Type\nD1 | 900x2100 | Wood", "bbox": [0.1, 0.5, 0.4, 0.8]}
            ],
            "pictures": []
        }
        chunks = chunker.process_docling_output(docling_output, "test.jpg")
        
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert chunks[0].confidence == 0.857  # Default table confidence
    
    def test_process_docling_output_pictures(self):
        """Test processing pictures from Docling output."""
        chunker = ConstructionDrawingChunker()
        docling_output = {
            "texts": [],
            "tables": [],
            "pictures": [
                {"caption": "Floor Plan", "bbox": [0.1, 0.1, 0.8, 0.8]}  # Large = viewport
            ]
        }
        chunks = chunker.process_docling_output(docling_output, "test.jpg")
        
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "viewport"  # Large area = viewport
    
    def test_process_docling_output_text_clustering(self):
        """Test text block clustering."""
        chunker = ConstructionDrawingChunker(text_cluster_eps=0.1)
        docling_output = {
            "texts": [
                {"text": "GENERAL", "bbox": [0.1, 0.1, 0.15, 0.12]},
                {"text": "NOTES", "bbox": [0.12, 0.1, 0.18, 0.12]},
                {"text": "1. Note one", "bbox": [0.1, 0.13, 0.3, 0.15]},
            ],
            "tables": [],
            "pictures": []
        }
        chunks = chunker.process_docling_output(docling_output, "test.jpg")
        
        # Should cluster nearby text blocks
        assert len(chunks) >= 1
        assert all(c.chunk_type in ["text", "title_block", "notes"] for c in chunks)
    
    def test_title_block_detection(self):
        """Test title block detection by position."""
        chunker = ConstructionDrawingChunker(title_block_threshold=0.85)
        docling_output = {
            "texts": [
                {"text": "Project Name", "bbox": [0.9, 0.8, 0.98, 0.85]},  # Right edge
                {"text": "Sheet A-101", "bbox": [0.9, 0.85, 0.98, 0.9]},
            ],
            "tables": [],
            "pictures": []
        }
        chunks = chunker.process_docling_output(docling_output, "test.jpg")
        
        # Elements on right edge should be classified as title_block
        title_blocks = [c for c in chunks if c.chunk_type == "title_block"]
        assert len(title_blocks) >= 1
    
    def test_metadata_preserved(self):
        """Test that metadata is properly preserved."""
        chunker = ConstructionDrawingChunker()
        docling_output = {
            "texts": [{"text": "Test", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            "tables": [],
            "pictures": []
        }
        chunks = chunker.process_docling_output(docling_output, "my_drawing.jpg")
        
        assert chunks[0].metadata["source_image"] == "my_drawing.jpg"
