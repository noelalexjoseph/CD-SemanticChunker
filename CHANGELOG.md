# Changelog

All notable changes to the construction-rag library.

## [0.1.1] - 2026-02-03

### Fixed

- **Critical: Coordinate normalization for DBSCAN clustering** - Fixed a bug where Docling's pixel coordinates were passed directly to DBSCAN clustering without normalization. The `eps=0.02` parameter expects normalized coordinates (0-1 range), but pixel coordinates (0-2000+) caused all text blocks to be treated as noise, resulting in no effective clustering.

  **Impact before fix:**
  - ~159 chunks per image (each text block as separate chunk)
  - All chunks incorrectly classified as `title_block`
  - Semantic search returned fragmented, unusable results

  **Impact after fix:**
  - ~80-95 chunks per image (proper semantic clustering)
  - Correct chunk type distribution: `text` (62%), `title_block` (30%), `notes` (5%), `figure/table` (3%)
  - Semantic search returns meaningful, grouped content

- **Position-based classification** - Fixed chunk type classification to use normalized bounding box coordinates. Previously, any element with x > 0.85 pixels was classified as `title_block` (essentially everything). Now correctly identifies title blocks in the rightmost 15% of the page.

### Validated

Tested against thesis experiment results:

| Dataset | Experiment Mean | Library Mean | Status |
|---------|-----------------|--------------|--------|
| full_sheet | 77.2 chunks/img | 87.0 chunks/img | PASS |
| FloorVERIFICATION | 88.9 chunks/img | 66.6 chunks/img | PASS |

### Changed

- `cluster_text_blocks()` in `chunker.py` now normalizes center coordinates before DBSCAN
- `process_docling_output()` now calculates page bounds and normalizes bounding boxes for position classification

## [0.1.0] - 2026-01-26

### Added

- Initial release
- IBM Docling integration for document parsing
- DBSCAN spatial clustering for text grouping
- ChromaDB vector storage
- OpenRouter LLM integration for summaries
- RAGAS evaluation support
