"""Tests for RAG pipeline module."""

import pytest
import tempfile
import os
from construction_rag.rag import ConstructionDrawingRAG
from construction_rag.models import BoundingBox, Chunk


class TestConstructionDrawingRAG:
    """Tests for ConstructionDrawingRAG class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
        return [
            Chunk(
                chunk_id="text_0001",
                chunk_type="text",
                content="General notes for the construction project. All work shall comply with local building codes.",
                bbox=bbox,
                confidence=0.9,
                metadata={"source_image": "test1.jpg"},
                summary="General construction notes about code compliance"
            ),
            Chunk(
                chunk_id="table_0001",
                chunk_type="table",
                content="Door Schedule\nD1 | 900x2100 | Wood | 60min fire rating\nD2 | 800x2100 | Metal | None",
                bbox=BoundingBox(0.5, 0.5, 1.0, 1.0),
                confidence=0.857,
                metadata={"source_image": "test1.jpg"},
                summary="Door schedule with sizes and fire ratings"
            ),
            Chunk(
                chunk_id="viewport_0001",
                chunk_type="viewport",
                content="Floor Plan - Level 1",
                bbox=BoundingBox(0.1, 0.1, 0.9, 0.9),
                confidence=0.475,
                metadata={"source_image": "test2.jpg"},
                summary="First floor plan layout"
            ),
        ]
    
    def test_init_with_persist(self, temp_db):
        """Test initialization with persistent storage."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        assert rag.persist_directory == temp_db
        assert rag.embedding_dimension == 384  # all-MiniLM-L6-v2
    
    def test_init_creates_directory(self, temp_db):
        """Test that init creates the persist directory."""
        db_path = os.path.join(temp_db, "new_db")
        rag = ConstructionDrawingRAG(persist_directory=db_path)
        assert os.path.exists(db_path)
    
    def test_add_chunks(self, temp_db, sample_chunks):
        """Test adding chunks to the vector store."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        added = rag.add_chunks(sample_chunks)
        
        assert added == 3
        stats = rag.get_stats()
        assert stats["total_chunks"] == 3
    
    def test_query_basic(self, temp_db, sample_chunks):
        """Test basic query functionality."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        results = rag.query("door schedule fire rating", n_results=3)
        
        assert len(results) == 3
        assert all(hasattr(r, 'relevance_score') for r in results)
        # The door schedule should be most relevant
        assert results[0].chunk_id == "table_0001"
    
    def test_query_with_filter_type(self, temp_db, sample_chunks):
        """Test query with chunk type filter."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        results = rag.query("schedule", n_results=5, filter_type="table")
        
        # Should only return table chunks
        assert all(r.metadata["chunk_type"] == "table" for r in results)
    
    def test_query_with_filter_source(self, temp_db, sample_chunks):
        """Test query with source image filter."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        results = rag.query("floor plan", n_results=5, filter_source="test2.jpg")
        
        # Should only return chunks from test2.jpg
        assert all(r.metadata["source_image"] == "test2.jpg" for r in results)
    
    def test_get_chunk_by_id(self, temp_db, sample_chunks):
        """Test retrieving a specific chunk by ID."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        chunk = rag.get_chunk_by_id("table_0001")
        
        assert chunk is not None
        assert chunk["chunk_id"] == "table_0001"
    
    def test_get_chunk_by_id_not_found(self, temp_db, sample_chunks):
        """Test retrieving non-existent chunk."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        chunk = rag.get_chunk_by_id("nonexistent_0001")
        
        assert chunk is None
    
    def test_get_stats(self, temp_db, sample_chunks):
        """Test statistics retrieval."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        stats = rag.get_stats()
        
        assert stats["total_chunks"] == 3
        assert stats["chunks_by_type"]["text"] == 1
        assert stats["chunks_by_type"]["table"] == 1
        assert stats["chunks_by_type"]["viewport"] == 1
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
    
    def test_clear(self, temp_db, sample_chunks):
        """Test clearing the collection."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        assert rag.get_stats()["total_chunks"] == 3
        
        rag.clear()
        
        assert rag.get_stats()["total_chunks"] == 0
    
    def test_delete_by_source(self, temp_db, sample_chunks):
        """Test deleting chunks by source image."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        deleted = rag.delete_by_source("test1.jpg")
        
        assert deleted == 2  # 2 chunks from test1.jpg
        assert rag.get_stats()["total_chunks"] == 1
    
    def test_empty_query(self, temp_db):
        """Test querying empty collection."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        
        results = rag.query("anything")
        
        assert results == []
    
    def test_relevance_score_range(self, temp_db, sample_chunks):
        """Test that relevance scores are in valid range."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        rag.add_chunks(sample_chunks)
        
        results = rag.query("test query", n_results=3)
        
        for r in results:
            assert 0 <= r.relevance_score <= 1
    
    def test_batch_add_chunks(self, temp_db):
        """Test adding many chunks in batches."""
        rag = ConstructionDrawingRAG(persist_directory=temp_db)
        
        # Create 150 chunks (to test batching with batch_size=100)
        chunks = []
        for i in range(150):
            chunks.append(Chunk(
                chunk_id=f"text_{i:04d}",
                chunk_type="text",
                content=f"Content for chunk {i}",
                bbox=BoundingBox(0, 0, 1, 1),
                confidence=0.9,
                metadata={"source_image": "batch_test.jpg"}
            ))
        
        added = rag.add_chunks(chunks, batch_size=100)
        
        assert added == 150
        assert rag.get_stats()["total_chunks"] == 150
