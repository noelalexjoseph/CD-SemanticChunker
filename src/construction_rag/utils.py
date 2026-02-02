"""
Utility functions for Construction RAG library.
"""

import sys
import numpy as np
from typing import List

from .models import BoundingBox


def print_safe(message: str) -> None:
    """
    Print with Unicode support on Windows.
    
    Args:
        message: Message to print
    """
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode('utf-8'))
        sys.stdout.flush()


def bbox_distance(b1: BoundingBox, b2: BoundingBox) -> float:
    """
    Calculate Euclidean distance between two bounding boxes (center-to-center).
    
    Args:
        b1: First bounding box
        b2: Second bounding box
        
    Returns:
        Distance between centers
    """
    c1 = b1.center
    c2 = b2.center
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def merge_bboxes(boxes: List[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into one encompassing box.
    
    Args:
        boxes: List of bounding boxes to merge
        
    Returns:
        Single bounding box that encompasses all input boxes
    """
    if not boxes:
        return BoundingBox(0, 0, 0, 0)
    
    x1 = min(b.x1 for b in boxes)
    y1 = min(b.y1 for b in boxes)
    x2 = max(b.x2 for b in boxes)
    y2 = max(b.y2 for b in boxes)
    
    return BoundingBox(x1, y1, x2, y2)


def classify_chunk_by_position(
    bbox: BoundingBox,
    title_block_threshold: float = 0.85,
    notes_threshold: float = 0.85
) -> str:
    """
    Classify a chunk type based on its position on the page.
    
    Construction drawings typically have:
    - Title blocks in the right 15% of the sheet
    - Notes at the bottom 15% of the sheet
    
    Args:
        bbox: Bounding box of the chunk
        title_block_threshold: X position threshold for title block (default 0.85)
        notes_threshold: Y position threshold for notes (default 0.85)
        
    Returns:
        Chunk type string ('title_block', 'notes', or 'text')
    """
    center_x, center_y = bbox.center
    
    if center_x > title_block_threshold:
        return "title_block"
    elif center_y > notes_threshold:
        return "notes"
    else:
        return "text"
