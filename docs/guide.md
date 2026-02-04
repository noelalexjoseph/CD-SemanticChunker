# Construction RAG - Comprehensive Guide

This guide provides a detailed explanation of the Construction RAG library: what it is, how it works, and how to use it effectively.

## Table of Contents

1. [What is Construction-RAG?](#what-is-construction-rag)
2. [Library Structure](#library-structure)
3. [The Processing Pipeline](#the-processing-pipeline)
4. [Module Reference](#module-reference)
5. [Examples Explained](#examples-explained)
6. [Key Algorithms](#key-algorithms)
7. [Configuration Reference](#configuration-reference)

---

## What is Construction-RAG?

### The Problem

Construction drawings are visual documents containing critical project information: floor plans, schedules, specifications, and notes. However, this information is trapped in images and PDFs that traditional text search cannot access.

**Challenges with construction drawings:**
- Text is embedded in images, not searchable
- Information is spatially distributed across the page
- OCR produces noisy, fragmented text
- No semantic understanding of content

### The Solution

Construction RAG transforms construction drawings into searchable, queryable content for Retrieval-Augmented Generation (RAG) systems. It enables:

- **Natural language queries**: "What is the fire rating for the doors?"
- **Semantic search**: Find relevant content by meaning, not just keywords
- **LLM-powered answers**: Get intelligent responses grounded in document content

### How It Fits Into RAG Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  Documents → Text Extraction → Chunking → Embeddings → Query    │
│     ↑                                                            │
│  (Text files, PDFs with text layers)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  CONSTRUCTION RAG PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│  Images → Layout Detection → Clustering → Summaries → Query     │
│     ↑                                                            │
│  (Construction drawings, floor plans, schedules)                │
└─────────────────────────────────────────────────────────────────┘
```

Construction RAG bridges the gap between visual construction documents and modern RAG systems.

---

## Library Structure

### What Makes This a Library?

Construction RAG is a **Python package** that can be:
- Installed via `pip install construction-rag`
- Imported into any Python project
- Used as building blocks for larger applications

Unlike standalone scripts, a library provides:
- **Reusable components**: Import only what you need
- **Consistent API**: Well-defined interfaces
- **Version management**: Track dependencies and updates
- **Documentation**: API reference and guides

### Directory Structure

```
construction-rag/
├── src/construction_rag/    # The library source code
│   ├── __init__.py          # Package exports and version
│   ├── pipeline.py          # High-level unified API
│   ├── chunker.py           # Document parsing + DBSCAN clustering
│   ├── rag.py               # Vector storage + semantic retrieval
│   ├── llm.py               # LLM integration (OpenRouter)
│   ├── models.py            # Data structures (Chunk, BoundingBox, etc.)
│   ├── utils.py             # Helper functions
│   ├── visualization.py     # Matplotlib plotting utilities
│   └── evaluation.py        # Batch evaluation with statistics
│
├── examples/                 # Demo notebooks and scripts
│   ├── 01_basic_usage.ipynb      # Getting started tutorial
│   ├── 02_full_pipeline.ipynb    # Complete workflow demo
│   ├── 03_evaluation.ipynb       # Quality assessment
│   ├── run_full_pipeline.py      # Command-line script
│   └── sample_images/            # Test images
│
├── evaluation/               # RAGAS evaluation framework
│   ├── ragas_evaluator.py        # Evaluation metrics
│   └── test_cases.py             # Ground truth test cases
│
├── docs/                     # Documentation
│   ├── guide.md                  # This comprehensive guide
│   ├── api.md                    # API reference
│   └── architecture.md           # System architecture
│
├── tests/                    # Unit tests
├── pyproject.toml            # Package configuration
└── README.md                 # Quick start guide
```

### Using the Library

**Option 1: High-level API (recommended for most users)**

```python
from construction_rag import ConstructionRAGPipeline

pipeline = ConstructionRAGPipeline()
pipeline.process("drawing.jpg")
answer = pipeline.ask("What doors are specified?")
```

**Option 2: Individual components (for custom workflows)**

```python
from construction_rag import (
    ConstructionDrawingChunker,
    ConstructionDrawingRAG,
    OpenRouterLLM
)

# Use components separately
chunker = ConstructionDrawingChunker()
chunks = chunker.process_image("drawing.jpg")

rag = ConstructionDrawingRAG()
rag.add_chunks(chunks)
```

---

## The Processing Pipeline

Construction RAG processes drawings through a five-stage pipeline:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   INPUT     │    │   DOCLING   │    │   DBSCAN    │    │     LLM     │    │  CHROMADB   │
│  (Image)    │ -> │  (Parsing)  │ -> │ (Clustering)│ -> │ (Summaries) │ -> │  (Storage)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                  │                  │
                    548 text blocks    98 clusters      Human-readable      Semantic
                    6 tables           Merged bboxes    descriptions        search ready
                    19 pictures
```

### Stage 1: Input

**What happens:** The pipeline accepts construction drawing images.

**Supported formats:** JPG, PNG, PDF (converted to images)

**Recommended resolution:** 1500-2500 pixels on longest side

```python
# Single image
result = pipeline.process("floor_plan.jpg")

# Multiple images
results = pipeline.process_batch(["plan1.jpg", "plan2.jpg", "schedule.jpg"])
```

### Stage 2: IBM Docling Parsing

**What happens:** IBM Docling analyzes the document layout and extracts:
- **Texts**: Individual text blocks with bounding boxes
- **Tables**: Structured table content
- **Pictures**: Drawing viewports and figures

**Technology:**
- DocLayNet: Layout analysis trained on 80,863 annotated pages
- TableFormer: Table structure recognition
- RT-DETR: Real-time detection backbone

**Output:** ~548 granular text blocks, 6 tables, 19 pictures (typical)

**Performance on construction drawings:**
- Table detection: 85.7% accuracy
- Picture detection: 47.5% accuracy
- Text extraction: Comprehensive but fragmented

### Stage 3: DBSCAN Spatial Clustering

**What happens:** Groups spatially proximate text blocks into semantic regions.

**Why needed:** Docling produces very granular output. A single note might be split into 20+ separate text blocks. DBSCAN merges these into coherent chunks.

**Algorithm:**
```
For each text block:
  1. Calculate centroid (center of bounding box)
  2. Normalize coordinates to 0-1 range
  3. Run DBSCAN clustering
  4. Merge content and bounding boxes for each cluster
```

**Parameters:**
- `eps=0.02`: Distance threshold (~2% of page dimensions)
- `min_samples=2`: Minimum blocks to form a cluster

**Result:** 548 text blocks → 98 semantic clusters (5.6× reduction)

### Stage 4: LLM Summarization

**What happens:** GPT-4o-mini generates human-readable summaries for each chunk.

**Why needed:** OCR text is often noisy and fragmented. Summaries provide clean descriptions that improve retrieval.

**Example transformation:**
```
OCR Input:  "DOORTAO HECHTETABSHEDPRCFRSTFLOOR GENERALNOTESSECTION"
Summary:    "General notes section containing door specifications 
             and first floor schedule references."
```

**API:** OpenRouter (supports multiple LLM providers)

**Model:** `openai/gpt-4o-mini` (fast, cost-effective)

### Stage 5: ChromaDB Vector Storage

**What happens:** Chunks are embedded and stored for semantic search.

**Embedding model:** `all-MiniLM-L6-v2`
- Dimensions: 384
- Speed: ~2500 sentences/second
- Quality: Good balance of speed and accuracy

**Stored metadata:**
- `chunk_type`: text, table, viewport, title_block, notes
- `source_image`: Original image filename
- `bbox`: Bounding box coordinates (x1, y1, x2, y2)
- `content`: Full text content
- `summary`: LLM-generated summary

### Query Flow

After processing, users can query the indexed content:

```python
# Semantic search
results = pipeline.query("door schedule", n_results=5)

# Question answering
answer = pipeline.ask("What is the fire rating for the doors?")
```

**How queries work:**
1. Query text is embedded using the same model
2. ChromaDB finds nearest neighbors by cosine similarity
3. For Q&A, relevant chunks are sent to LLM as context
4. LLM generates answer grounded in retrieved content

---

## Module Reference

### models.py - Core Data Structures

Defines the data classes used throughout the library.

**BoundingBox**
```python
from construction_rag import BoundingBox

bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)

# Properties
bbox.center   # (0.3, 0.5) - center point
bbox.width    # 0.4
bbox.height   # 0.6
bbox.area     # 0.24
```

Coordinates are normalized to 0-1 range (percentage of image dimensions).

**Chunk**
```python
from construction_rag import Chunk, BoundingBox

chunk = Chunk(
    chunk_id="text_0001",           # Unique identifier
    chunk_type="text",              # text, table, viewport, title_block, notes
    content="Door schedule...",     # Extracted text content
    bbox=BoundingBox(0.1, 0.2, 0.5, 0.8),
    confidence=0.9,                 # Detection confidence
    metadata={"source_image": "plan.jpg"},
    summary="Door schedule showing fire ratings"  # LLM summary
)
```

**ProcessingResult**
```python
result = pipeline.process("drawing.jpg")

result.source_image      # "drawing.jpg"
result.chunks            # List[Chunk]
result.processing_time   # 5.2 (seconds)
result.success           # True/False
result.error             # Error message if failed
```

**QueryResult**
```python
results = pipeline.query("door schedule")

for r in results:
    r.chunk_id          # "text_0001"
    r.content           # Chunk content
    r.metadata          # Dict with chunk_type, source_image, etc.
    r.distance          # Vector distance (lower = more similar)
    r.relevance_score   # 1 - distance (higher = more relevant)
```

### chunker.py - Document Parsing

Handles the two-step extraction and clustering pipeline.

**ConstructionDrawingChunker**
```python
from construction_rag import ConstructionDrawingChunker

chunker = ConstructionDrawingChunker(
    title_block_threshold=0.85,    # X position for title block detection
    text_cluster_eps=0.02,         # DBSCAN epsilon
    text_cluster_min_samples=2     # DBSCAN min samples
)

# Process image (combines both steps)
chunks = chunker.process_image("drawing.jpg")

# Or use two-step approach for more control
raw_chunks, width, height = chunker.extract_raw_chunks("drawing.jpg")
chunks = chunker.cluster_and_finalize(raw_chunks, width, height)
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `extract_raw_chunks(image_path)` | Run Docling, fix Y-coordinates, return raw chunks |
| `cluster_and_finalize(raw_chunks, w, h)` | DBSCAN clustering, normalize coordinates |
| `process_image(image_path)` | Convenience method combining both steps |
| `process_docling_output(data, name, w, h)` | Process pre-extracted Docling JSON |

**RawChunk**
```python
from construction_rag.chunker import RawChunk

# Intermediate format with pixel coordinates
raw = RawChunk(
    chunk_type='text',
    content='Door schedule',
    bbox={'x1': 100, 'y1': 200, 'x2': 500, 'y2': 300},  # Pixels
    confidence=0.9,
    source_image='drawing.jpg',
    element_index=0
)
```

### rag.py - Vector Storage

Manages ChromaDB for semantic search.

**ConstructionDrawingRAG**
```python
from construction_rag import ConstructionDrawingRAG

rag = ConstructionDrawingRAG(
    persist_directory="./my_db",           # Storage location
    embedding_model="all-MiniLM-L6-v2",    # Sentence transformer
    collection_name="construction_drawings"
)

# Add chunks
rag.add_chunks(chunks)

# Query
results = rag.query(
    "door schedule",
    n_results=5,
    filter_type="table",           # Optional: filter by chunk type
    filter_source="plan.jpg"       # Optional: filter by source image
)

# Get specific chunk
chunk_data = rag.get_chunk_by_id("text_0001")

# Statistics
stats = rag.get_stats()
# {'total_chunks': 98, 'chunks_by_type': {'text': 80, 'table': 6, ...}}

# Delete chunks from a source
deleted = rag.delete_by_source("old_plan.jpg")

# Clear all
rag.clear()
```

### llm.py - LLM Integration

Provides LLM capabilities via OpenRouter.

**OpenRouterLLM**
```python
from construction_rag import OpenRouterLLM

llm = OpenRouterLLM(
    model="openai/gpt-4o-mini",    # Model identifier
    api_key=None                   # Uses OPENROUTER_API_KEY env var
)

# Generate chunk summary
summary = llm.generate_summary(
    content="DOOR SCHEDULE TYPE A...",
    chunk_type="table"
)

# Answer question with context
answer = llm.answer_query(
    query="What is the fire rating?",
    contexts=["Door Type A: 60 min fire rating", "Door Type B: 90 min"]
)

# Extract page title from summaries
title = llm.extract_page_title(["Floor plan showing...", "General notes..."])

# Usage statistics
stats = llm.get_stats()
# {'total_calls': 50, 'total_tokens': 12500, 'model': 'openai/gpt-4o-mini'}
```

### pipeline.py - Unified Interface

High-level API combining all components.

**ConstructionRAGPipeline**
```python
from construction_rag import ConstructionRAGPipeline

pipeline = ConstructionRAGPipeline(
    persist_directory="./db",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="openai/gpt-4o-mini",
    llm_api_key=None,              # Or pass directly
    enable_summaries=True,
    cluster_eps=0.02,
    cluster_min_samples=2,
    title_block_threshold=0.85
)

# Process single image
result = pipeline.process("drawing.jpg")

# Process batch
results = pipeline.process_batch(
    ["plan1.jpg", "plan2.jpg"],
    verbose=True
)

# Semantic search
results = pipeline.query("door schedule", n_results=5)

# Question answering
answer = pipeline.ask("What doors are specified?")

# With context chunks returned
answer, context_chunks = pipeline.ask(
    "What is the fire rating?",
    return_context=True
)

# Statistics
stats = pipeline.get_stats()

# Clear database
pipeline.clear()
```

### visualization.py - Plotting Utilities

Matplotlib-based visualization functions.

**draw_bounding_boxes**
```python
from construction_rag import draw_bounding_boxes
import matplotlib.pyplot as plt

fig = draw_bounding_boxes(
    image_path="drawing.jpg",
    chunks=chunks,
    highlight_chunks=["text_0001", "text_0002"],  # Highlight specific chunks
    show_labels=True,
    figsize=(12, 8)
)
plt.show()
plt.close(fig)
```

**crop_chunk_region**
```python
from construction_rag.visualization import crop_chunk_region

# Get cropped image of a chunk's region
cropped_image = crop_chunk_region(
    image_path="drawing.jpg",
    chunk=chunk,
    max_size=400,    # Max dimension
    padding=10       # Padding around bbox
)
```

**display_chunk_with_summary**
```python
from construction_rag.visualization import display_chunk_with_summary

# Display chunk image alongside its summary
fig = display_chunk_with_summary(
    image_path="drawing.jpg",
    chunk=chunk,
    figsize=(10, 4)
)
plt.show()
plt.close(fig)
```

### evaluation.py - Batch Evaluation

Statistical validation for large-scale testing.

**BatchEvaluator**
```python
from construction_rag import BatchEvaluator

evaluator = BatchEvaluator(
    results_dir="./evaluation_results",
    checkpoint_enabled=True,
    verbose=True
)

# Evaluate a directory of images
results, stats = evaluator.evaluate_directory(
    directory="./test_images",
    dataset_name="floor_plans",
    sample_size=50
)

# Print summary
evaluator.print_summary(stats)

# Save results
evaluator.save_results(
    {"floor_plans": results},
    {"floor_plans": stats}
)
```

**DatasetStats**
```python
stats.dataset_name              # "floor_plans"
stats.images_processed          # 50
stats.images_failed             # 2
stats.total_chunks              # 4800
stats.mean_chunks_per_image     # 96.0
stats.std_chunks_per_image      # 12.5
stats.mean_processing_time      # 5.9
stats.confidence_interval_95    # (93.5, 98.5)
```

---

## Examples Explained

### 01_basic_usage.ipynb - Getting Started

**Purpose:** Introduction to the library for new users.

**What it covers:**
1. Installing and importing the library
2. Initializing the pipeline
3. Processing a single construction drawing
4. Examining extracted chunks (types, content, bounding boxes)
5. Running basic semantic search queries
6. Simple question answering

**When to use:** First-time users learning the library.

**Key code:**
```python
from construction_rag import ConstructionRAGPipeline

pipeline = ConstructionRAGPipeline(enable_summaries=False)
result = pipeline.process("sample_images/floor_plan.jpg")
results = pipeline.query("door schedule")
```

### 02_full_pipeline.ipynb - Complete Workflow

**Purpose:** Demonstrates the full capabilities of the library.

**What it covers:**
1. Batch processing multiple drawings
2. Visualizing bounding boxes on images
3. LLM-generated summaries
4. Semantic search with result visualization
5. Question answering with context display
6. Cropped chunk visualization
7. Pipeline statistics

**When to use:** Understanding the complete workflow, building applications.

**Key code:**
```python
# Batch processing
results = pipeline.process_batch(image_paths, verbose=True)

# Visualization
fig = draw_bounding_boxes(image_path, chunks, highlight_chunks=relevant_ids)

# Q&A with context
answer, context = pipeline.ask("What doors are specified?", return_context=True)
```

### 03_evaluation.ipynb - Quality Assessment

**Purpose:** Evaluate RAG pipeline quality using RAGAS-inspired metrics.

**What it covers:**
1. Understanding evaluation metrics
2. Loading predefined test cases
3. Running evaluation
4. Interpreting results (precision, recall, F1)
5. Comparing different configurations

**When to use:** Assessing retrieval quality, comparing configurations.

**Metrics explained:**
- **Context Precision:** Are retrieved contexts relevant?
- **Context Recall:** Are all relevant contexts retrieved?
- **F1 Score:** Harmonic mean of precision and recall
- **Keyword Coverage:** Do results contain expected keywords?

### run_full_pipeline.py - Command-Line Script

**Purpose:** Run the full pipeline from command line for testing.

**Usage:**
```bash
cd examples
python run_full_pipeline.py
```

**What it does:**
1. Checks API key configuration
2. Initializes pipeline
3. Processes all sample images
4. Runs semantic search tests
5. Runs Q&A tests
6. Prints statistics

**When to use:** Automated testing, CI/CD pipelines, quick validation.

---

## Key Algorithms

### DBSCAN Spatial Clustering

**Problem:** IBM Docling produces highly granular text output. A single paragraph might be split into dozens of separate text blocks, each with its own bounding box.

**Solution:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups spatially proximate blocks.

**How it works:**
```
1. Extract centroid (center point) of each text block's bounding box
2. Normalize coordinates to 0-1 range (per-image normalization)
3. Run DBSCAN with eps=0.02, min_samples=2
4. For each cluster:
   - Merge text content (space-separated)
   - Merge bounding boxes (min/max of all corners)
   - Average confidence scores
```

**Parameters:**
- `eps=0.02`: Two blocks within 2% of page dimensions are neighbors
- `min_samples=2`: A cluster needs at least 2 blocks

**Results:**
- Input: ~548 text blocks per drawing
- Output: ~98 semantic clusters
- Reduction: 5.6×

**Code location:** `chunker.py`, method `_cluster_text_chunks()`

### Y-Coordinate Inversion

**Problem:** IBM Docling uses PDF coordinate system where Y=0 is at the bottom. Image coordinates have Y=0 at the top.

**Solution:** Invert Y-coordinates during extraction.

**Formula:**
```python
# PDF coordinates (from Docling)
pdf_top = bbox.t      # Larger Y value
pdf_bottom = bbox.b   # Smaller Y value

# Image coordinates (what we need)
image_y1 = img_height - pdf_top     # Top of box in image
image_y2 = img_height - pdf_bottom  # Bottom of box in image
```

**Code location:** `chunker.py`, method `extract_raw_chunks()`, lines 146-158

### Noise Point Handling

**Problem:** DBSCAN labels isolated points (not near any cluster) as noise (label=-1). By default, all noise points would be grouped together.

**Solution:** Treat each noise point as its own cluster.

**Implementation:**
```python
for i, label in enumerate(labels):
    if label == -1:
        # Each noise point becomes its own cluster
        label = f"noise_{i}"
    clusters[label].append(text_chunks[i])
```

**Why it matters:** Isolated text blocks (like standalone labels or notes) should remain separate, not be merged with other unrelated isolated text.

**Code location:** `chunker.py`, method `_cluster_text_chunks()`, lines 429-439

---

## Configuration Reference

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist_directory` | `"./construction_rag_db"` | ChromaDB storage location |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence transformer model for embeddings |
| `llm_model` | `"openai/gpt-4o-mini"` | OpenRouter LLM model identifier |
| `llm_api_key` | `None` | OpenRouter API key (or use env var) |
| `enable_summaries` | `True` | Generate LLM summaries for chunks |
| `cluster_eps` | `0.02` | DBSCAN epsilon (~2% of page dimensions) |
| `cluster_min_samples` | `2` | DBSCAN minimum samples per cluster |
| `title_block_threshold` | `0.85` | X position threshold for title block detection |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for OpenRouter LLM access |
| `CUDA_VISIBLE_DEVICES` | Set to "" to force CPU mode |

### Chunk Types

| Type | Description | Detection |
|------|-------------|-----------|
| `text` | General text content | Default for clustered text |
| `table` | Tabular data | Docling table detection |
| `viewport` | Drawing viewports/figures | Docling picture detection (large) |
| `figure` | Small figures | Docling picture detection (small) |
| `title_block` | Title block text | X position > 0.85 |
| `notes` | Notes section | Y position > 0.85 |

### Recommended Image Sizes

| Resolution | Processing Time | Quality |
|------------|-----------------|---------|
| 1000x700 | ~15-20s | Good for testing |
| 1500x1000 | ~30-45s | Recommended |
| 2000x1400 | ~60-90s | High quality |
| 4000x3000 | ~3-5 min | Maximum detail |

---

## Troubleshooting

### Common Issues

**"CUDA error: no kernel image is available"**
- Your GPU architecture is not supported by PyTorch
- Solution: Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**"OpenRouter API key not found"**
- Set the environment variable: `export OPENROUTER_API_KEY="your-key"`
- Or pass directly: `ConstructionRAGPipeline(llm_api_key="your-key")`

**Processing is very slow**
- Reduce image resolution (1500-2000px recommended)
- Disable summaries for testing: `enable_summaries=False`
- Use CPU-only PyTorch if GPU is causing issues

**Memory errors during visualization**
- Close figures after displaying: `plt.close(fig)`
- Reduce `max_image_size` in `draw_bounding_boxes()`
- Process fewer images at once

---

## Next Steps

- [API Reference](api.md) - Detailed API documentation
- [Architecture](architecture.md) - System design and data flow
- [Examples](../examples/) - Jupyter notebooks with working code
