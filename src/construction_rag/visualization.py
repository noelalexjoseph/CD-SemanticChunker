"""
Visualization utilities for Construction RAG library.

This module provides functions to visualize bounding boxes and chunks
on construction drawings for debugging and demonstration purposes.
"""

from typing import List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure

from .models import Chunk


# Color scheme for different chunk types
CHUNK_COLORS = {
    'title_block': '#2196F3',  # Blue
    'text': '#4CAF50',         # Green
    'notes': '#FF9800',        # Orange
    'figure': '#9C27B0',       # Purple
    'table': '#F44336',        # Red
}

HIGHLIGHT_COLOR = '#FFEB3B'  # Bright yellow for highlighted chunks


def draw_bounding_boxes(
    image_path: str,
    chunks: List[Chunk],
    highlight_chunks: Optional[List[str]] = None,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (10, 7),
    max_image_size: int = 1200,
    dpi: int = 72
) -> Figure:
    """
    Draw bounding boxes on image for all chunks.
    
    Args:
        image_path: Path to the source image
        chunks: List of chunks to visualize
        highlight_chunks: Optional list of chunk_ids to highlight (brighter color)
        show_labels: Whether to show chunk type labels
        figsize: Figure size for matplotlib
        max_image_size: Maximum dimension for image (will downsample if larger)
        dpi: DPI for figure rendering (lower = less memory)
        
    Returns:
        matplotlib Figure object
    """
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Downsample image to prevent memory errors
    if max(img_width, img_height) > max_image_size:
        scale_factor = max_image_size / max(img_width, img_height)
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_width, img_height = img.size
    
    # Auto-disable labels if too many chunks to reduce rendering overhead
    if len(chunks) > 50:
        show_labels = False
    
    # Create figure with explicit low DPI to reduce memory usage
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    
    highlight_set = set(highlight_chunks) if highlight_chunks else set()
    
    # Draw bounding boxes for each chunk
    for chunk in chunks:
        # Convert normalized coordinates to pixel coordinates
        x1 = chunk.bbox.x1 * img_width
        y1 = chunk.bbox.y1 * img_height
        x2 = chunk.bbox.x2 * img_width
        y2 = chunk.bbox.y2 * img_height
        
        width = x2 - x1
        height = y2 - y1
        
        # Determine color and line width
        is_highlighted = chunk.chunk_id in highlight_set
        
        if is_highlighted:
            color = HIGHLIGHT_COLOR
            linewidth = 3
            alpha = 0.3
        else:
            color = CHUNK_COLORS.get(chunk.chunk_type, '#808080')  # Gray for unknown types
            linewidth = 1.5
            alpha = 0.15
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(rect)
        
        # Add label if requested
        if show_labels:
            label_text = chunk.chunk_type
            if is_highlighted:
                label_text = f"★ {label_text}"
            
            # Position label at top-left of bbox
            ax.text(
                x1, y1 - 5,
                label_text,
                fontsize=8,
                color=color,
                weight='bold' if is_highlighted else 'normal',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    alpha=0.7,
                    edgecolor=color,
                    linewidth=1
                )
            )
    
    # Adjust layout (suppress warnings if it fails)
    try:
        plt.tight_layout()
    except Exception:
        pass
    
    return fig


def create_chunk_legend() -> None:
    """
    Display a legend showing the color scheme for chunk types.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    y_pos = 0.9
    ax.text(0.1, y_pos, "Chunk Type Legend:", fontsize=14, weight='bold')
    
    y_pos -= 0.15
    for chunk_type, color in CHUNK_COLORS.items():
        # Draw color box
        rect = patches.Rectangle(
            (0.1, y_pos - 0.05), 0.1, 0.08,
            facecolor=color,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(0.25, y_pos, chunk_type, fontsize=12, va='center')
        y_pos -= 0.15
    
    # Add highlight example
    rect = patches.Rectangle(
        (0.1, y_pos - 0.05), 0.1, 0.08,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=3
    )
    ax.add_patch(rect)
    ax.text(0.25, y_pos, "highlighted (query match)", fontsize=12, va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def crop_chunk_region(
    image_path: str,
    chunk: Chunk,
    max_size: int = 400,
    padding: int = 10
) -> Image.Image:
    """
    Crop the bounding box region from the source image.
    
    This extracts the portion of the drawing that corresponds to a chunk's
    bounding box, useful for showing what part of the image was summarized.
    
    Args:
        image_path: Path to the source image
        chunk: Chunk with normalized bbox coordinates (0-1)
        max_size: Maximum dimension for the cropped image
        padding: Pixels of padding around the bbox
    
    Returns:
        PIL Image of the cropped region
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Convert normalized coords to pixels
    x1 = int(chunk.bbox.x1 * img_width) - padding
    y1 = int(chunk.bbox.y1 * img_height) - padding
    x2 = int(chunk.bbox.x2 * img_width) + padding
    y2 = int(chunk.bbox.y2 * img_height) + padding
    
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)
    
    # Ensure valid crop region
    if x2 <= x1 or y2 <= y1:
        # Return a small placeholder if bbox is invalid
        return Image.new('RGB', (100, 100), color='white')
    
    # Crop and resize if needed
    cropped = img.crop((x1, y1, x2, y2))
    if max(cropped.size) > max_size:
        cropped.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return cropped


def display_chunk_with_summary(
    image_path: str,
    chunk: Chunk,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 72
) -> Figure:
    """
    Display a chunk's cropped region alongside its summary.
    
    Shows the visual region from the drawing on the left and the
    LLM-generated summary (or content preview) on the right.
    
    Args:
        image_path: Path to the source image
        chunk: Chunk to display
        figsize: Figure size (width, height) in inches
        dpi: DPI for figure rendering
    
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                    gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Left: Cropped image region
    try:
        cropped = crop_chunk_region(image_path, chunk)
        ax1.imshow(cropped)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Could not load image:\n{str(e)}", 
                 ha='center', va='center', fontsize=8)
    
    # Add chunk type as title with color
    color = CHUNK_COLORS.get(chunk.chunk_type, '#808080')
    ax1.set_title(f"[{chunk.chunk_type}]", fontsize=11, color=color, weight='bold')
    ax1.axis('off')
    
    # Right: Summary text
    summary_text = chunk.summary if chunk.summary else chunk.content
    
    # Truncate if too long
    max_chars = 500
    if len(summary_text) > max_chars:
        summary_text = summary_text[:max_chars] + "..."
    
    # Format the display text
    display_text = f"Summary:\n{summary_text}"
    
    ax2.text(0.02, 0.98, display_text,
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             horizontalalignment='left',
             wrap=True,
             family='monospace')
    ax2.axis('off')
    ax2.set_title(f"Chunk ID: {chunk.chunk_id}", fontsize=9, color='gray')
    
    plt.tight_layout()
    return fig
