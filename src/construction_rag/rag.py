"""
RAG Pipeline for Construction Drawing Chunks.

This module provides:
- Embedding generation using sentence-transformers
- Vector storage with ChromaDB
- Semantic retrieval for natural language queries

The pipeline indexes construction drawing chunks and enables
semantic search to find relevant content.
"""

import os
from typing import List, Dict, Optional

from .models import Chunk, BoundingBox, QueryResult


# Default configuration
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "construction_drawings"


class ConstructionDrawingRAG:
    """
    RAG pipeline for construction drawing chunks.
    
    Provides embedding generation, vector storage (ChromaDB), and semantic retrieval
    for construction document content.
    
    Example:
        >>> rag = ConstructionDrawingRAG()
        >>> rag.add_chunks(chunks)
        >>> results = rag.query("door schedule")
        >>> for r in results:
        ...     print(f"{r.relevance_score:.2f}: {r.content[:50]}...")
    """
    
    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            persist_directory: Directory for ChromaDB persistence.
                               Set to None for in-memory storage.
            embedding_model: Sentence transformer model name.
                             Default 'all-MiniLM-L6-v2' produces 384-dim embeddings.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Create persist directory if specified
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        # Initialize ChromaDB
        try:
            import chromadb
            if persist_directory:
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.client = chromadb.Client()
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Construction drawing chunks for RAG"}
        )
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
            batch_size: Number of chunks to process at once (for memory efficiency)
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = [c.chunk_id for c in batch]
            
            # Use summary if available, otherwise content
            documents = [c.summary or c.content for c in batch]
            
            metadatas = [
                {
                    "chunk_type": c.chunk_type,
                    "confidence": c.confidence,
                    "source_image": c.metadata.get("source_image", ""),
                    "page_title": c.metadata.get("page_title", "Unknown Sheet"),
                    "bbox_x1": c.bbox.x1,
                    "bbox_y1": c.bbox.y1,
                    "bbox_x2": c.bbox.x2,
                    "bbox_y2": c.bbox.y2,
                    "content": c.content[:1000],  # Store truncated content in metadata
                    "summary": c.summary or ""
                }
                for c in batch
            ]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            added += len(batch)
        
        return added
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_type: Optional[str] = None,
        filter_source: Optional[str] = None
    ) -> List[QueryResult]:
        """
        Query the vector store for relevant chunks.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            filter_type: Optional filter by chunk_type (e.g., "table", "text", "viewport")
            filter_source: Optional filter by source_image
        
        Returns:
            List of QueryResult objects with matching chunks
        """
        # Build filter
        where_filter = None
        if filter_type or filter_source:
            conditions = []
            if filter_type:
                conditions.append({"chunk_type": filter_type})
            if filter_source:
                conditions.append({"source_image": filter_source})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        query_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                query_results.append(QueryResult(
                    chunk_id=results['ids'][0][i],
                    content=results['metadatas'][0][i].get('content', results['documents'][0][i]),
                    metadata=results['metadatas'][0][i],
                    distance=results['distances'][0][i],
                    relevance_score=1 - results['distances'][0][i]
                ))
        
        return query_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Dict with chunk data, or None if not found
        """
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if result['ids']:
            return {
                "chunk_id": result['ids'][0],
                "content": result['metadatas'][0].get('content', result['documents'][0]),
                "metadata": result['metadatas'][0]
            }
        return None
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dict with total_chunks, chunks_by_type, collection_name, and embedding_model
        """
        count = self.collection.count()
        
        # Get type distribution
        type_counts = {}
        if count > 0:
            all_items = self.collection.get(include=["metadatas"])
            for meta in all_items['metadatas']:
                t = meta.get('chunk_type', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_chunks": count,
            "chunks_by_type": type_counts,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension
        }
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Construction drawing chunks for RAG"}
        )
    
    def delete_by_source(self, source_image: str) -> int:
        """
        Delete all chunks from a specific source image.
        
        Args:
            source_image: The source image name to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        # Get chunks from this source
        results = self.collection.get(
            where={"source_image": source_image},
            include=["metadatas"]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
