# Architecture

For a comprehensive explanation of the library including usage examples and module details, see the [Comprehensive Guide](guide.md).

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

**Two-Step Pipeline:**

The chunker implements a two-step approach matching the thesis experiments:

1. **Step 1 - Raw Extraction** (`extract_raw_chunks()`):
   - Run Docling on the image
   - Extract texts, tables, pictures with bounding boxes
   - Apply Y-coordinate fix (PDF coords to image coords)
   - Store as `RawChunk` objects with pixel coordinates

2. **Step 2 - Clustering** (`cluster_and_finalize()`):
   - Normalize coordinates to 0-1 range for clustering
   - Run DBSCAN on text block centroids
   - Merge content and bounding boxes for each cluster
   - Classify chunks by position (title_block, notes, text)

**Parameters:**
- `eps=0.02`: ~2% of page dimensions
- `min_samples=2`: Allows pairs and larger groups

**Noise Point Handling:**

DBSCAN labels isolated points as noise (label=-1). The library treats each noise point as its own cluster to prevent unrelated isolated text from being grouped together:

```python
for i, label in enumerate(labels):
    if label == -1:
        label = f"noise_{i}"  # Each noise point becomes its own cluster
    clusters[label].append(text_chunks[i])
```

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

## Key Implementation Details

### Y-Coordinate Inversion

IBM Docling uses PDF coordinate system where Y=0 is at the bottom. The library inverts this to image coordinates (Y=0 at top):

```python
# PDF coordinates from Docling
pdf_top = bbox.t      # Larger Y value in PDF
pdf_bottom = bbox.b   # Smaller Y value in PDF

# Convert to image coordinates
image_y1 = img_height - pdf_top     # Top of box in image
image_y2 = img_height - pdf_bottom  # Bottom of box in image
```

**Location:** `chunker.py`, method `extract_raw_chunks()`

### Chunk Type Classification

Chunks are classified based on their position on the page:

| Type | Condition | Description |
|------|-----------|-------------|
| `title_block` | X center > 0.85 | Rightmost 15% of page |
| `notes` | Y center > 0.85 | Bottom 15% of page |
| `text` | Default | General text content |
| `table` | Docling detection | Tabular data |
| `viewport` | Docling detection, area > 0.1 | Large figures |
| `figure` | Docling detection, area <= 0.1 | Small figures |

### Embedding Strategy

The RAG component uses summaries (when available) for embedding, with fallback to raw content:

```python
# Embedding priority
documents = [c.summary or c.content for c in chunks]
```

This improves retrieval quality because summaries are cleaner than noisy OCR text.

## See Also

- [Comprehensive Guide](guide.md) - Detailed usage and module reference
- [API Reference](api.md) - Complete API documentation
