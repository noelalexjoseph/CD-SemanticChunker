# Construction RAG

Open-source Python library for transforming construction drawings into searchable chunks for Retrieval-Augmented Generation (RAG) systems.

## About This Project

This library was developed as part of the Master's thesis:

**"Exploring RAG Methods and Techniques as One Source of Truth for Querying Construction Documents"**

- **Author:** Joseph Noel
- **Program:** M.Sc. Construction & Robotics
- **Institution:** RWTH Aachen University, Individualized Production
- **Date:** February 2026

The thesis addresses the challenge of making construction drawing content searchable through RAG, enabling natural language queries over visual technical documents that traditional search cannot access.

## Features

- **IBM Docling Integration** - Document layout detection with 85.7% table detection accuracy
- **DBSCAN Spatial Clustering** - Groups granular text blocks into semantic regions (5.6× reduction: 548 → 98 chunks)
- **LLM Summarization** - GPT-4o-mini generates human-readable chunk summaries via OpenRouter
- **ChromaDB Vector Storage** - Semantic search with sentence-transformers embeddings
- **RAGAS Evaluation** - Built-in evaluation framework for assessing retrieval quality

## Installation

```bash
pip install construction-rag
```

Or install from source:

```bash
git clone https://github.com/jnoel/construction-rag.git
cd construction-rag
pip install -e .
```

### Optional Dependencies

```bash
# For evaluation tools
pip install construction-rag[evaluation]

# For development
pip install construction-rag[dev]

# For Jupyter notebooks
pip install construction-rag[notebooks]

# All extras
pip install construction-rag[all]
```

## Quick Start

### Basic Usage

```python
from construction_rag import ConstructionRAGPipeline

# Initialize pipeline
pipeline = ConstructionRAGPipeline()

# Process a construction drawing
result = pipeline.process("floor_plan.jpg")
print(f"Extracted {len(result.chunks)} chunks in {result.processing_time:.1f}s")

# Query the indexed content
results = pipeline.query("door schedule")
for r in results:
    print(f"{r.relevance_score:.2f}: {r.content[:80]}...")

# Ask questions with LLM-generated answers
answer = pipeline.ask("What is the fire rating for the doors?")
print(answer)
```

### Processing Multiple Drawings

```python
from construction_rag import ConstructionRAGPipeline

pipeline = ConstructionRAGPipeline(
    persist_directory="./my_project_db",
    enable_summaries=True
)

# Process batch
image_paths = ["plan1.jpg", "plan2.jpg", "schedule.jpg"]
results = pipeline.process_batch(image_paths)

# Check statistics
stats = pipeline.get_stats()
print(f"Indexed {stats['total_chunks']} chunks")
print(f"Types: {stats['chunks_by_type']}")
```

### Using Individual Components

```python
from construction_rag import (
    ConstructionDrawingChunker,
    ConstructionDrawingRAG,
    OpenRouterLLM
)

# Just chunking (no LLM or storage)
chunker = ConstructionDrawingChunker(
    text_cluster_eps=0.02,  # DBSCAN epsilon
    text_cluster_min_samples=2
)
chunks = chunker.process_image("drawing.jpg")

# Just RAG storage/retrieval
rag = ConstructionDrawingRAG(persist_directory="./db")
rag.add_chunks(chunks)
results = rag.query("general notes", n_results=5)

# Just LLM
llm = OpenRouterLLM()  # Requires OPENROUTER_API_KEY env var
summary = llm.generate_summary(chunk.content, chunk.chunk_type)
```

## Configuration

### Environment Variables

```bash
# Required for LLM features
export OPENROUTER_API_KEY="your-api-key"
```

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist_directory` | `"./construction_rag_db"` | ChromaDB storage location |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `llm_model` | `"openai/gpt-4o-mini"` | OpenRouter LLM model |
| `enable_summaries` | `True` | Generate LLM chunk summaries |
| `cluster_eps` | `0.02` | DBSCAN epsilon (~2% of page) |
| `cluster_min_samples` | `2` | Min samples per cluster |
| `title_block_threshold` | `0.85` | X position for title block |

## Architecture

```
┌─────────────────┐
│ Construction    │
│ Drawing (Image) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IBM Docling    │  Document parsing
│  Layout Parser  │  (tables, text, figures)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DBSCAN Spatial  │  548 text blocks
│   Clustering    │  → 98 semantic clusters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPT-4o-mini    │  Human-readable
│  Summarization  │  chunk summaries
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │  all-MiniLM-L6-v2
│  Vector Store   │  embeddings (384-dim)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Query  │  Natural language
│   Interface     │  retrieval
└─────────────────┘
```

## Evaluation Results

The library was evaluated on 155 construction drawing images from four datasets:

| Metric | Result |
|--------|--------|
| **F1 Score** | 66.1% |
| **Context Precision** | 80.0% |
| **Context Recall** | 56.2% |
| **Processing Success Rate** | 98.1% (152/155 images) |
| **Average Processing Time** | 5.9 seconds/image |
| **ANOVA p-value** | 0.109 (consistent across document types) |

### Clustering Effectiveness

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Text chunks | 548 | 98 | 5.6× reduction |
| F1 Score | 57.8% | 66.1% | +14% |

## API Reference

### ConstructionRAGPipeline

Main high-level interface combining all components.

```python
pipeline = ConstructionRAGPipeline(
    persist_directory="./db",
    enable_summaries=True
)

# Methods
result = pipeline.process(image_path)      # Process single image
results = pipeline.process_batch(paths)    # Process multiple images
query_results = pipeline.query(text)       # Semantic search
answer = pipeline.ask(question)            # LLM-powered Q&A
stats = pipeline.get_stats()               # Collection statistics
pipeline.clear()                           # Clear all data
```

### ConstructionDrawingChunker

Document parsing and DBSCAN clustering.

```python
chunker = ConstructionDrawingChunker(
    title_block_threshold=0.85,
    text_cluster_eps=0.02,
    text_cluster_min_samples=2
)

chunks = chunker.process_image(path)
chunks = chunker.process_docling_output(docling_result, image_name)
```

### ConstructionDrawingRAG

Vector storage and semantic retrieval.

```python
rag = ConstructionDrawingRAG(
    persist_directory="./db",
    embedding_model="all-MiniLM-L6-v2",
    collection_name="my_collection"
)

rag.add_chunks(chunks)
results = rag.query(text, n_results=5, filter_type="table")
chunk = rag.get_chunk_by_id(chunk_id)
stats = rag.get_stats()
rag.clear()
```

## Evaluation Module

```python
from construction_rag import ConstructionDrawingRAG
from construction_rag.evaluation import RAGEvaluator, TEST_CASES

rag = ConstructionDrawingRAG()
evaluator = RAGEvaluator(rag)

results = evaluator.evaluate_all(TEST_CASES)
print(f"F1 Score: {results['aggregate_metrics']['f1_score']:.2%}")
```

## Citation

If you use this library in your research, please cite:

```bibtex
@mastersthesis{noel2026construction,
  title={Exploring RAG Methods and Techniques as One Source of Truth 
         for Querying Construction Documents},
  author={Noel, Joseph},
  year={2026},
  school={RWTH Aachen University},
  type={Master's Thesis},
  program={M.Sc. Construction \& Robotics}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [IBM Docling](https://github.com/DS4SD/docling) - Document layout detection
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [OpenRouter](https://openrouter.ai/) - LLM API gateway
- [Roboflow Universe](https://universe.roboflow.com/) - Construction drawing datasets (CC BY 4.0)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
