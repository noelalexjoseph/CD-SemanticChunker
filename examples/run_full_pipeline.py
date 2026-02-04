"""
Run the full pipeline notebook cells for testing/debugging.
"""
import os
import sys
import glob

# Add parent directory to path for construction_rag import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 60)
print("CONSTRUCTION RAG - FULL PIPELINE TEST")
print("=" * 60)

# Cell 1: Check API key
from construction_rag import ConstructionRAGPipeline

api_key = os.environ.get("OPENROUTER_API_KEY")
print(f"\n[1] OpenRouter API key: {'Set' if api_key else 'Not set'}")

# Cell 3: Initialize Pipeline
print("\n[2] Initializing pipeline...")
pipeline = ConstructionRAGPipeline(
    persist_directory="./full_pipeline_db",
    enable_summaries=True,
    llm_model="openai/gpt-4o-mini",
    cluster_eps=0.02,
    cluster_min_samples=2
)

print(f"    LLM enabled: {pipeline.llm is not None}")
if pipeline.llm:
    print(f"    LLM model: {pipeline.llm.model}")

# Cell 5: Get sample images
print("\n[3] Finding sample images...")
image_paths = glob.glob("sample_images/*.jpg")
print(f"    Found {len(image_paths)} sample images:")
for path in image_paths:
    print(f"      - {os.path.basename(path)}")

# Cell 6: Process all images
print("\n[4] Processing images (this may take a while)...")
results = pipeline.process_batch(image_paths, verbose=True)

# Cell 7: Summary
successful = sum(1 for r in results if r.success)
total_chunks = sum(len(r.chunks) for r in results)
total_time = sum(r.processing_time for r in results)

print(f"\n[5] Processing Summary:")
print(f"    Images processed: {successful}/{len(results)}")
print(f"    Total chunks: {total_chunks}")
print(f"    Total time: {total_time:.1f}s")
print(f"    Average time/image: {total_time/len(results):.1f}s")

# Cell 9: Show chunks with summaries
all_chunks = [c for r in results for c in r.chunks]

print("\n[6] Sample chunks with summaries:")
for chunk in all_chunks[:5]:
    print(f"\n    [{chunk.chunk_type}] {chunk.chunk_id}")
    content_preview = chunk.content[:80].replace('\n', ' ')
    print(f"    Content: {content_preview}...")
    if chunk.summary:
        print(f"    Summary: {chunk.summary}")

# Cell 11: Semantic search
print("\n[7] Semantic Search Tests:")
queries = [
    "door schedule fire rating",
    "general notes",
    "project information",
    "floor plan layout"
]

for query in queries:
    print(f"\n    Query: '{query}'")
    search_results = pipeline.query(query, n_results=2)
    for r in search_results:
        print(f"      [{r.metadata['chunk_type']}] Score: {r.relevance_score:.3f}")
        print(f"      Source: {r.metadata.get('source_image', 'Unknown')}")

# Cell 13: Question answering
print("\n[8] Question Answering Tests:")
questions = [
    "What types of doors are mentioned in the drawings?",
    "What are the general construction notes?",
]

for question in questions:
    print(f"\n    Q: {question}")
    try:
        answer = pipeline.ask(question)
        print(f"    A: {answer[:200]}..." if len(answer) > 200 else f"    A: {answer}")
    except ValueError as e:
        print(f"    Error: {e}")
        print("    (LLM not available - set OPENROUTER_API_KEY)")

# Cell 15: Statistics
print("\n[9] Final Statistics:")
stats = pipeline.get_stats()
print(f"    Total chunks indexed: {stats['total_chunks']}")
print(f"    Embedding model: {stats['embedding_model']}")
print(f"\n    Chunks by type:")
for chunk_type, count in stats['chunks_by_type'].items():
    print(f"      {chunk_type}: {count}")

print("\n" + "=" * 60)
print("PIPELINE TEST COMPLETE")
print("=" * 60)
