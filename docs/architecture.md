# Architecture

## System Overview

Construction RAG processes construction drawings through a five-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSTRUCTION RAG PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Construction│    │  IBM Docling│    │   DBSCAN    │
│   Drawing   │ -> │   Parser    │ -> │  Clustering │
│   (Image)   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                         │ 548 text blocks   │ 98 clusters
                         │ 6 tables          │
                         │ 19 pictures       │
                         v                   v
                   ┌─────────────┐    ┌─────────────┐
                   │ GPT-4o-mini │    │  ChromaDB   │
                   │ Summarizer  │ -> │Vector Store │
                   │             │    │             │
                   └─────────────┘    └─────────────┘
                                            │
                                            v
                                     ┌─────────────┐
                                     │  Semantic   │
                                     │   Query     │
                                     │  Interface  │
                                     └─────────────┘
```

## Components

### 1. Document Parsing (IBM Docling)

IBM Docling provides document layout detection using:
- **DocLayNet**: Layout analysis trained on 80,863 annotated pages
- **TableFormer**: Table structure recognition
- **RT-DETR**: Real-time detection backbone

**Performance on Construction Drawings:**
- Table detection: 85.7% accuracy
- Picture detection: 47.5% accuracy
- Text extraction: Comprehensive

### 2. DBSCAN Spatial Clustering

Docling produces highly granular text output (~548 blocks per drawing).
DBSCAN groups spatially proximate blocks into semantic regions.

**Parameters:**
- `eps=0.02`: ~2% of page dimensions
- `min_samples=2`: Allows pairs and larger groups

**Results:**
- Input: 548 text blocks
- Output: 98 semantic clusters
- Reduction: 5.6×

### 3. LLM Summarization

GPT-4o-mini via OpenRouter generates human-readable summaries
that help with retrieval of noisy OCR content.

**Example Transformation:**
```
OCR: "DOORTAO HECHTETABSHEDPRCFRSTFLOOR GENERALNOTESSECTION"
Summary: "General notes section containing door specifications 
          and first floor schedule references."
```

### 4. Vector Storage (ChromaDB)

ChromaDB stores chunk embeddings with metadata for semantic search.

**Embedding Model:** all-MiniLM-L6-v2
- Dimensions: 384
- Speed: ~2500 sentences/second

**Stored Metadata:**
- chunk_type
- source_image
- page_title
- bounding_box
- summary

### 5. Semantic Retrieval

Natural language queries are embedded and matched against
stored chunks using cosine similarity.

**Features:**
- Type filtering (table, text, viewport)
- Source filtering
- Configurable result count

## Data Flow

```
Image File
    │
    ▼
DocumentConverter.convert()
    │
    ├── texts[]     ─┐
    ├── tables[]     ├── Docling output
    └── pictures[]  ─┘
           │
           ▼
    cluster_text_blocks()
           │
           ▼
    Chunk objects
           │
           ├── generate_summary() ─── LLM
           │
           ▼
    rag.add_chunks()
           │
           ▼
    ChromaDB (persistent)
           │
           ▼
    rag.query() ─── User queries
```

## Module Dependencies

```
construction_rag/
├── models.py       # BoundingBox, Chunk, etc. (no deps)
├── utils.py        # Utilities (depends on models)
├── chunker.py      # Docling + DBSCAN (depends on models, utils)
├── llm.py          # OpenRouter LLM (depends on models)
├── rag.py          # ChromaDB (depends on models)
└── pipeline.py     # Unified API (depends on all above)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Processing time | 5.9s/image average |
| Success rate | 98.1% |
| F1 Score | 66.1% |
| Chunk reduction | 5.6× |
| Embedding dimension | 384 |
