# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-02

### Added

- Initial release of Construction RAG library
- `ConstructionRAGPipeline` - Unified high-level interface
- `ConstructionDrawingChunker` - Document parsing with IBM Docling and DBSCAN clustering
- `ConstructionDrawingRAG` - ChromaDB vector storage and semantic retrieval
- `OpenRouterLLM` - LLM integration for summaries and Q&A
- RAGAS-based evaluation module
- Example Jupyter notebooks
- Comprehensive documentation

### Performance

- 85.7% table detection accuracy
- 66.1% F1 score for RAG retrieval
- 98.1% processing success rate
- 5.9s average processing time per image
- 5.6× chunk reduction through DBSCAN clustering

## [Unreleased]

### Planned

- Custom viewport detector for construction drawings
- Domain-specific OCR post-processing
- Multi-language support
- IFC integration for BIM linking
