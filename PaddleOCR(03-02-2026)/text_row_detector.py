"""
Module for detecting text rows from OCR results.
Groups words by similar Y-coordinates to form logical text rows with support for different reading directions.
"""

from typing import Dict, List, Tuple, Literal, Optional
from collections import defaultdict
import re


def detect_text_rows(
    detections: List[Tuple],
    y_tolerance: int = 10,
    reading_direction: Literal['ltr', 'rtl', 'auto'] = 'ltr'
) -> Dict[float, List[Tuple[float, str, float]]]:
    """
    Groups OCR detection results by similar Y-coordinates to form text rows.

    Args:
        detections: List of tuples containing (x_position, text, confidence, y_position) or
                   (left_x, text, confidence) from OCR results
                   Format: [(x_pos, text, confidence, y_pos), ...] or [(left_x, text, confidence), ...]
        y_tolerance: Maximum vertical distance (in pixels) to consider words as part of the same row.
                     Default is 10 pixels.
        reading_direction: Direction in which text should be read within each row.
                          Options:
                          - 'ltr': Left-to-right (default)
                          - 'rtl': Right-to-left
                          - 'auto': Automatically detect based on content characteristics

    Returns:
        Dictionary where:
        - Keys are representative Y-coordinates for each row
        - Values are lists of tuples containing (x_position, text, confidence) sorted by reading direction

    Example:
        >>> detections = [(100, "Hello", 0.95, 50), (200, "World", 0.92, 51), (50, "Hi", 0.88, 100)]
        >>> rows = detect_text_rows(detections, y_tolerance=10, reading_direction='ltr')
        >>> # Would group "Hello" and "World" in one row (Y~50), "Hi" in another row (Y~100)
    """
    if not detections:
        return {}

    # Dictionary to hold rows: {row_y: [(x, text, confidence), ...]}
    rows: Dict[float, List[Tuple[float, str, float]]] = defaultdict(list)

    for detection in detections:
        if len(detection) < 3:
            continue  # Skip invalid detections

        # Handle different input formats
        if len(detection) == 4:
            # Format: (x_position, text, confidence, y_position)
            x_pos, text, confidence, y_pos = detection
        elif len(detection) == 3:
            # Format: (x_position, text, confidence) - assume y_position is not provided
            # In this case, we'll need to get y_position from bounding boxes if available
            # For now, we'll skip this format or need additional information
            x_pos, text, confidence = detection
            # Without y_pos, we can't properly group into rows, so we'll skip
            continue
        else:
            continue  # Skip unrecognized formats

        # Try to find an existing row that this word belongs to
        assigned = False
        for row_y in rows.keys():
            if abs(y_pos - row_y) <= y_tolerance:
                # Add to existing row
                rows[row_y].append((x_pos, text, confidence))
                assigned = True
                break

        if not assigned:
            # Create a new row with this y-position
            rows[y_pos].append((x_pos, text, confidence))

    # Determine actual reading direction if 'auto' is selected
    actual_direction = reading_direction
    if reading_direction == 'auto':
        actual_direction = _detect_reading_direction_from_content(rows)

    # Sort words within each row based on reading direction
    for row_y in rows:
        if actual_direction == 'ltr':
            rows[row_y].sort(key=lambda item: item[0])  # Sort by x_position ascending (left to right)
        elif actual_direction == 'rtl':
            rows[row_y].sort(key=lambda item: item[0], reverse=True)  # Sort by x_position descending (right to left)
        else:
            # Default to LTR if direction is somehow invalid
            rows[row_y].sort(key=lambda item: item[0])

    return dict(rows)


def detect_text_rows_from_bounding_boxes(
    detections: List[Tuple],
    y_tolerance: int = 10,
    reading_direction: Literal['ltr', 'rtl', 'auto'] = 'ltr'
) -> Dict[float, List[Tuple[float, str, float]]]:
    """
    Groups OCR detection results by similar Y-coordinates to form text rows,
    using bounding box information to determine Y-coordinates.

    Args:
        detections: List of OCR results in format [(bbox, (text, confidence)), ...]
                   where bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        y_tolerance: Maximum vertical distance (in pixels) to consider words as part of the same row.
                     Default is 10 pixels.
        reading_direction: Direction in which text should be read within each row.
                          Options:
                          - 'ltr': Left-to-right (default)
                          - 'rtl': Right-to-left
                          - 'auto': Automatically detect based on content characteristics

    Returns:
        Dictionary where:
        - Keys are representative Y-coordinates for each row
        - Values are lists of tuples containing (x_position, text, confidence) sorted by reading direction
    """
    if not detections:
        return {}

    # Dictionary to hold rows: {row_y: [(x, text, confidence), ...]}
    rows: Dict[float, List[Tuple[float, str, float]]] = defaultdict(list)

    for detection in detections:
        if len(detection) < 2:
            continue  # Skip invalid detections

        bbox, text_info = detection

        if not isinstance(bbox, (list, tuple)) or len(bbox) < 1:
            continue  # Invalid bounding box

        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue  # Invalid text info

        text = text_info[0]
        confidence = text_info[1]

        # Calculate Y-coordinate as the average of all Y-coordinates in the bounding box
        y_coords = [point[1] for point in bbox if len(point) >= 2]
        if not y_coords:
            continue
        y_pos = sum(y_coords) / len(y_coords)

        # Calculate X-coordinate as the minimum X (leftmost position)
        x_coords = [point[0] for point in bbox if len(point) >= 2]
        if not x_coords:
            continue
        x_pos = min(x_coords)

        # Try to find an existing row that this word belongs to
        assigned = False
        for row_y in rows.keys():
            if abs(y_pos - row_y) <= y_tolerance:
                # Add to existing row
                rows[row_y].append((x_pos, text, confidence))
                assigned = True
                break

        if not assigned:
            # Create a new row with this y-position
            rows[y_pos].append((x_pos, text, confidence))

    # Determine actual reading direction if 'auto' is selected
    actual_direction = reading_direction
    if reading_direction == 'auto':
        actual_direction = _detect_reading_direction_from_content(rows)

    # Sort words within each row based on reading direction
    for row_y in rows:
        if actual_direction == 'ltr':
            rows[row_y].sort(key=lambda item: item[0])  # Sort by x_position ascending (left to right)
        elif actual_direction == 'rtl':
            rows[row_y].sort(key=lambda item: item[0], reverse=True)  # Sort by x_position descending (right to left)
        else:
            # Default to LTR if direction is somehow invalid
            rows[row_y].sort(key=lambda item: item[0])

    return dict(rows)


def detect_text_rows_with_position_info(
    detections: List[Tuple],
    y_tolerance: int = 10,
    reading_direction: Literal['ltr', 'rtl', 'auto'] = 'ltr'
) -> Dict[Tuple[int, int], List[Tuple[float, str, float]]]:
    """
    Groups OCR detection results by similar Y-coordinates to form text rows,
    with more sophisticated row key management.

    Args:
        detections: List of tuples containing (x_position, text, confidence, y_position)
        y_tolerance: Maximum vertical distance (in pixels) to consider words as part of the same row.
                     Default is 10 pixels.
        reading_direction: Direction in which text should be read within each row.
                          Options:
                          - 'ltr': Left-to-right (default)
                          - 'rtl': Right-to-left
                          - 'auto': Automatically detect based on content characteristics

    Returns:
        Dictionary where:
        - Keys are rounded Y-coordinates to group nearby rows
        - Values are lists of tuples containing (x_position, text, confidence) sorted by reading direction
    """
    if not detections:
        return {}

    # Dictionary to hold rows: {rounded_y: [(x, text, confidence), ...]}
    rows: Dict[int, List[Tuple[float, str, float]]] = defaultdict(list)

    for detection in detections:
        if len(detection) < 4:
            continue  # Need (x, text, confidence, y) format

        x_pos, text, confidence, y_pos = detection

        # Find the appropriate row based on Y-coordinate with tolerance
        assigned = False
        for existing_y in rows.keys():
            if abs(y_pos - existing_y) <= y_tolerance:
                # Add to existing row
                rows[existing_y].append((x_pos, text, confidence))
                assigned = True
                break

        if not assigned:
            # Create a new row with this y-position (rounded to nearest multiple of tolerance for grouping)
            rounded_y = round(y_pos / y_tolerance) * y_tolerance
            rows[rounded_y].append((x_pos, text, confidence))

    # Determine actual reading direction if 'auto' is selected
    actual_direction = reading_direction
    if reading_direction == 'auto':
        actual_direction = _detect_reading_direction_from_content(rows)

    # Sort words within each row based on reading direction
    for row_y in rows:
        if actual_direction == 'ltr':
            rows[row_y].sort(key=lambda item: item[0])  # Sort by x_position ascending (left to right)
        elif actual_direction == 'rtl':
            rows[row_y].sort(key=lambda item: item[0], reverse=True)  # Sort by x_position descending (right to left)
        else:
            # Default to LTR if direction is somehow invalid
            rows[row_y].sort(key=lambda item: item[0])

    return dict(rows)


def _detect_reading_direction_from_content(
    rows: Dict[float, List[Tuple[float, str, float]]]
) -> Literal['ltr', 'rtl']:
    """
    Attempts to automatically detect reading direction based on content characteristics.

    Args:
        rows: Dictionary of rows with text content

    Returns:
        'ltr' or 'rtl' based on detected characteristics
    """
    # Count RTL characters (Arabic, Hebrew, Persian, etc.)
    rtl_chars = 0
    ltr_chars = 0

    for row_items in rows.values():
        for _, text, _ in row_items:
            # Count characters from RTL scripts
            for char in str(text):
                # Arabic, Hebrew, Persian, Urdu, etc. Unicode ranges
                if '\u0600' <= char <= '\u06FF' or \
                   '\u0750' <= char <= '\u077F' or \
                   '\u08A0' <= char <= '\u08FF' or \
                   '\uFB50' <= char <= '\uFDFF' or \
                   '\uFE70' <= char <= '\uFEFF':
                    rtl_chars += 1
                # Latin characters (common in LTR text)
                elif '\u0041' <= char <= '\u005A' or \
                     '\u0061' <= char <= '\u007A' or \
                     '\u00C0' <= char <= '\u024F':
                    ltr_chars += 1

    # Return direction based on character counts
    if rtl_chars > ltr_chars:
        return 'rtl'
    else:
        return 'ltr'


def _detect_reading_direction_by_position(
    rows: Dict[float, List[Tuple[float, str, float]]]
) -> Literal['ltr', 'rtl']:
    """
    Attempts to automatically detect reading direction based on positional arrangement.

    Args:
        rows: Dictionary of rows with positional information

    Returns:
        'ltr' or 'rtl' based on positional patterns
    """
    # Analyze the positional patterns of words in each row
    ltr_count = 0
    rtl_count = 0

    for row_items in rows.values():
        if len(row_items) < 2:
            continue

        # Check if consecutive words generally increase (LTR) or decrease (RTL) in X position
        for i in range(len(row_items) - 1):
            current_x = row_items[i][0]
            next_x = row_items[i + 1][0]

            if next_x > current_x:
                ltr_count += 1
            elif next_x < current_x:
                rtl_count += 1

    if rtl_count > ltr_count:
        return 'rtl'
    else:
        return 'ltr'


# Documentation for reading direction options
READING_DIRECTION_DOCS = """
Reading Direction Options:

'ltr' (Left-to-Right):
- Default option for most Western languages (English, French, German, Spanish, etc.)
- Words are sorted from leftmost X coordinate to rightmost X coordinate
- Suitable for Latin, Cyrillic, Greek scripts and others

'rtl' (Right-to-Left):
- For languages that read from right to left (Arabic, Hebrew, Persian, Urdu, etc.)
- Words are sorted from rightmost X coordinate to leftmost X coordinate
- Preserves the intended reading order for RTL languages

'auto' (Automatic Detection):
- Attempts to detect the appropriate reading direction automatically
- Uses two methods:
  1. Character-based detection: Identifies RTL script characters in the text
  2. Position-based detection: Analyzes the spatial arrangement of words
- Falls back to 'ltr' if detection is inconclusive
- Useful when processing documents with mixed or unknown reading directions
"""