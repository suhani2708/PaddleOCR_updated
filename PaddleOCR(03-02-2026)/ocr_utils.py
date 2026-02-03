"""
Utility functions for processing PaddleOCR results.
Provides tools for extracting word-level data from OCR results.
"""

from typing import List, Dict, Any, Union, Optional
import logging


def extract_word_data(results: Union[List[Any], Any]) -> List[Dict[str, Union[str, float]]]:
    """
    Extracts individual words with their coordinates and confidence scores from PaddleOCR results.
    
    This function processes the output from PaddleOCR's predict() method (with return_word_box=True)
    and returns structured data for each detected word, which is useful for creating Excel exports
    with proper row/column organization and confidence-based highlighting.
    
    Args:
        results: The output from PaddleOCR's predict() method. This can be:
                 - A list of detection results
                 - Each result may be a dictionary with 'boxes', 'texts', 'scores' keys
                 - Or in the legacy format with bounding box and text/confidence pairs
    
    Returns:
        A list of dictionaries, where each dictionary represents a detected word with:
        - 'text': The recognized text content (str)
        - 'x': The X coordinate (typically center X of the bounding box) (float)
        - 'y': The Y coordinate (typically center Y of the bounding box) (float)
        - 'confidence': The confidence score (0.0 to 1.0) (float)
        
    Raises:
        ValueError: If the input results are malformed or in an unexpected format
        
    Examples:
        >>> # Example with new API format
        >>> results = [{
        ...     'boxes': [[[10, 10], [50, 10], [50, 30], [10, 30]]],
        ...     'texts': ['Hello'],
        ...     'scores': [0.95]
        ... }]
        >>> word_data = extract_word_data(results)
        >>> print(word_data)
        [{'text': 'Hello', 'x': 30.0, 'y': 20.0, 'confidence': 0.95}]
        
        >>> # Example with legacy format
        >>> results = [[[[10, 10], [50, 10], [50, 30], [10, 30]], ('Hello', 0.95)]]
        >>> word_data = extract_word_data(results)
        >>> print(word_data)
        [{'text': 'Hello', 'x': 30.0, 'y': 20.0, 'confidence': 0.95}]
    """
    word_data_list: List[Dict[str, Union[str, float]]] = []
    
    if not results:
        return word_data_list
    
    # Handle different possible input formats
    if not isinstance(results, list):
        results = [results]
    
    for result in results:
        if result is None:
            continue
            
        # Check if result is in the new API format (dictionary with boxes, texts, scores)
        if isinstance(result, dict) and 'boxes' in result and 'texts' in result and 'scores' in result:
            boxes = result.get('boxes', [])
            texts = result.get('texts', [])
            scores = result.get('scores', [])
            
            # Validate that all arrays have the same length
            if not (len(boxes) == len(texts) == len(scores)):
                logging.warning(f"Mismatched lengths in OCR result: boxes={len(boxes)}, texts={len(texts)}, scores={len(scores)}")
                continue
            
            for i in range(len(texts)):
                try:
                    box = boxes[i]
                    text = texts[i]
                    score = scores[i]
                    
                    # Calculate center coordinates from bounding box
                    x_center = sum(point[0] for point in box) / len(box)
                    y_center = sum(point[1] for point in box) / len(box)
                    
                    word_data = {
                        'text': str(text),
                        'x': float(x_center),
                        'y': float(y_center),
                        'confidence': float(score)
                    }
                    
                    word_data_list.append(word_data)
                except (IndexError, TypeError, ValueError) as e:
                    logging.warning(f"Error processing OCR result item: {e}")
                    continue
        
        # Check if result is in legacy format (list of [box, (text, confidence)] pairs)
        elif isinstance(result, (list, tuple)):
            items = result[0] if len(result) == 1 and isinstance(result[0], list) else result
            
            if isinstance(items, (list, tuple)):
                for item in items:
                    if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                        continue
                    
                    box = item[0]
                    text_info = item[1]
                    
                    # Handle different text_info formats
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 1.0  # Default confidence if not provided
                    else:
                        continue
                    
                    try:
                        # Calculate center coordinates from bounding box
                        x_center = sum(point[0] for point in box) / len(box)
                        y_center = sum(point[1] for point in box) / len(box)
                        
                        word_data = {
                            'text': str(text),
                            'x': float(x_center),
                            'y': float(y_center),
                            'confidence': float(confidence)
                        }
                        
                        word_data_list.append(word_data)
                    except (IndexError, TypeError, ValueError) as e:
                        logging.warning(f"Error processing legacy OCR result item: {e}")
                        continue
        else:
            logging.warning(f"Unexpected OCR result format: {type(result)}")
            continue
    
    return word_data_list


def extract_word_data_with_fallback(results: Union[List[Any], Any]) -> List[Dict[str, Union[str, float]]]:
    """
    Enhanced version of extract_word_data with additional fallback mechanisms.
    
    This function attempts multiple strategies to extract word data from OCR results,
    providing more robust handling of different PaddleOCR output formats.
    
    Args:
        results: The output from PaddleOCR's predict() method
        
    Returns:
        A list of dictionaries with word data (text, x, y, confidence)
    """
    try:
        return extract_word_data(results)
    except Exception as e:
        logging.error(f"Primary extraction failed: {e}")
        
        # Fallback: try to handle as a flat list of items
        word_data_list: List[Dict[str, Union[str, float]]] = []
        
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    # Try to extract from dictionary format
                    boxes = item.get('boxes', [])
                    texts = item.get('texts', [])
                    scores = item.get('scores', [])
                    
                    for i in range(min(len(boxes), len(texts), len(scores))):
                        try:
                            box = boxes[i]
                            text = texts[i]
                            score = scores[i]
                            
                            x_center = sum(point[0] for point in box) / len(box)
                            y_center = sum(point[1] for point in box) / len(box)
                            
                            word_data_list.append({
                                'text': str(text),
                                'x': float(x_center),
                                'y': float(y_center),
                                'confidence': float(score)
                            })
                        except Exception:
                            continue
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Try legacy format
                    try:
                        box = item[0]
                        text_info = item[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                        else:
                            text = str(text_info)
                            confidence = 1.0
                        
                        x_center = sum(point[0] for point in box) / len(box)
                        y_center = sum(point[1] for point in box) / len(box)
                        
                        word_data_list.append({
                            'text': str(text),
                            'x': float(x_center),
                            'y': float(y_center),
                            'confidence': float(confidence)
                        })
                    except Exception:
                        continue
        
        return word_data_list


def filter_words_by_confidence(word_data: List[Dict[str, Union[str, float]]], 
                              min_confidence: float = 0.0) -> List[Dict[str, Union[str, float]]]:
    """
    Filters word data by minimum confidence threshold.
    
    Args:
        word_data: List of word data dictionaries from extract_word_data
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        Filtered list of word data dictionaries
    """
    return [word for word in word_data if word.get('confidence', 0.0) >= min_confidence]


def get_word_statistics(word_data: List[Dict[str, Union[str, float]]]) -> Dict[str, float]:
    """
    Calculates statistics for the extracted word data.
    
    Args:
        word_data: List of word data dictionaries from extract_word_data
        
    Returns:
        Dictionary containing statistics like mean confidence, coordinate ranges, etc.
    """
    if not word_data:
        return {}
    
    confidences = [word.get('confidence', 0.0) for word in word_data]
    x_coords = [word.get('x', 0.0) for word in word_data]
    y_coords = [word.get('y', 0.0) for word in word_data]
    
    stats = {
        'total_words': len(word_data),
        'mean_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'min_confidence': min(confidences) if confidences else 0.0,
        'max_confidence': max(confidences) if confidences else 0.0,
        'x_range': (min(x_coords) if x_coords else 0.0, max(x_coords) if x_coords else 0.0),
        'y_range': (min(y_coords) if y_coords else 0.0, max(y_coords) if y_coords else 0.0)
    }
    
    return stats