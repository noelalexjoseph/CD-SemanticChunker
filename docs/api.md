# API Reference

## Core Classes

### ConstructionRAGPipeline

The main high-level interface that combines all components.

```python
from construction_rag import ConstructionRAGPipeline

pipeline = ConstructionRAGPipeline(
    persist_directory="./db",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="openai/gpt-4o-mini",
    llm_api_key=None,
    enable_summaries=True,
    cluster_eps=0.02,
    cluster_min_samples=2,
    title_block_threshold=0.85
)
```

**Parameters:**
- `persist_directory` (str): Directory for ChromaDB persistence
- `embedding_model` (str): Sentence transformer model name
- `llm_model` (str): OpenRouter LLM model identifier
- `llm_api_key` (str, optional): OpenRouter API key
- `enable_summaries` (bool): Whether to generate LLM summaries
- `cluster_eps` (float): DBSCAN epsilon for text clustering
- `cluster_min_samples` (int): DBSCAN minimum samples
- `title_block_threshold` (float): X position threshold for title block detection

**Methods:**

#### process(image_path, generate_summaries=None)
Process a single construction drawing.

```python
result = pipeline.process("drawing.jpg")
```

Returns: `ProcessingResult`

#### process_batch(image_paths, generate_summaries=None, verbose=True)
Process multiple construction drawings.

```python
results = pipeline.process_batch(["plan1.jpg", "plan2.jpg"])
```

Returns: `List[ProcessingResult]`

#### query(query_text, n_results=5, filter_type=None)
Query the indexed content.

```python
results = pipeline.query("door schedule", n_results=3, filter_type="table")
```

Returns: `List[QueryResult]`

#### ask(question, n_context=5)
Ask a question and get an LLM-generated answer.

```python
answer = pipeline.ask("What is the fire rating for the doors?")
```

Returns: `str`

#### get_stats()
Get pipeline statistics.

Returns: `Dict`

#### clear()
Clear all indexed content.

---

### ConstructionDrawingChunker

Document parsing and DBSCAN clustering.

```python
from construction_rag import ConstructionDrawingChunker

chunker = ConstructionDrawingChunker(
    title_block_threshold=0.85,
    text_cluster_eps=0.02,
    text_cluster_min_samples=2
)
```

**Methods:**

#### process_image(image_path)
Process a single image through Docling.

```python
chunks = chunker.process_image("drawing.jpg")
```

Returns: `List[Chunk]`

#### process_docling_output(docling_result, image_name)
Process pre-computed Docling output.

Returns: `List[Chunk]`

#### reset_counter()
Reset the chunk ID counter.

---

### ConstructionDrawingRAG

Vector storage and semantic retrieval.

```python
from construction_rag import ConstructionDrawingRAG

rag = ConstructionDrawingRAG(
    persist_directory="./db",
    embedding_model="all-MiniLM-L6-v2",
    collection_name="construction_drawings"
)
```

**Methods:**

#### add_chunks(chunks, batch_size=100)
Add chunks to the vector store.

Returns: `int` (number added)

#### query(query_text, n_results=5, filter_type=None, filter_source=None)
Query for relevant chunks.

Returns: `List[QueryResult]`

#### get_chunk_by_id(chunk_id)
Get a specific chunk by ID.

Returns: `Dict` or `None`

#### get_stats()
Get collection statistics.

Returns: `Dict`

#### clear()
Clear all data.

#### delete_by_source(source_image)
Delete chunks from a specific source.

Returns: `int` (number deleted)

---

### OpenRouterLLM

LLM integration for summaries and Q&A.

```python
from construction_rag import OpenRouterLLM

llm = OpenRouterLLM(
    model="openai/gpt-4o-mini",
    api_key=None,  # Uses OPENROUTER_API_KEY env var
    base_url="https://openrouter.ai/api/v1"
)
```

**Methods:**

#### generate(prompt, max_tokens=200, temperature=0.3)
Generate a response.

Returns: `str`

#### generate_summary(content, chunk_type)
Generate a chunk summary.

Returns: `str`

#### answer_query(query, contexts)
Answer a question based on context.

Returns: `str`

#### get_stats()
Get usage statistics.

Returns: `Dict`

---

## Data Models

### Chunk

```python
from construction_rag import Chunk, BoundingBox

chunk = Chunk(
    chunk_id="text_0001",
    chunk_type="text",
    content="Content text...",
    bbox=BoundingBox(0.1, 0.2, 0.5, 0.8),
    confidence=0.9,
    metadata={"source_image": "test.jpg"},
    summary="Optional summary"
)
```

### BoundingBox

```python
bbox = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)

# Properties
bbox.center  # (0.3, 0.5)
bbox.width   # 0.4
bbox.height  # 0.6
bbox.area    # 0.24
```

### ProcessingResult

```python
result = ProcessingResult(
    source_image="test.jpg",
    chunks=[...],
    processing_time=2.5,
    success=True,
    error=None
)
```

### QueryResult

```python
result = QueryResult(
    chunk_id="text_0001",
    content="...",
    metadata={...},
    distance=0.2,
    relevance_score=0.8
)
```

---

## Utility Functions

### cluster_text_blocks

```python
from construction_rag import cluster_text_blocks

clusters = cluster_text_blocks(
    text_blocks=[{"text": "...", "bbox": {...}}, ...],
    eps=0.02,
    min_samples=2
)
```

### merge_bboxes

```python
from construction_rag import merge_bboxes

merged = merge_bboxes([bbox1, bbox2, bbox3])
```

### classify_chunk_by_position

```python
from construction_rag import classify_chunk_by_position

chunk_type = classify_chunk_by_position(
    bbox,
    title_block_threshold=0.85,
    notes_threshold=0.85
)
# Returns: "title_block", "notes", or "text"
```
