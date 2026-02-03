"""
Module for converting confidence values to color codes for visualization.
Provides color mapping for confidence levels to aid data entry operators.
"""

from typing import Union


def get_confidence_color(confidence: Union[float, int]) -> str:
    """
    Maps a confidence value to a hex color code for visualization purposes.
    
    The color mapping follows a gradient from red (low confidence) through yellow 
    (medium confidence) to green (high confidence), helping data entry operators
    quickly identify which text needs attention.
    
    Args:
        confidence: A float value between 0.0 and 1.0 representing the confidence level
                   where 0.0 is lowest confidence and 1.0 is highest confidence.
    
    Returns:
        A 6-character hex color string (without # prefix) representing the color
        associated with the confidence level.
        
        Color mapping:
        - 0.0: Red (#FF0000)
        - 0.25: Orange-red (#FF7F00)
        - 0.5: Yellow (#FFFF00)
        - 0.75: Yellow-green (#7FFF00)
        - 1.0: Green (#00FF00)
    
    Raises:
        ValueError: If confidence is not between 0.0 and 1.0 (inclusive)
        
    Examples:
        >>> get_confidence_color(0.0)
        'FF0000'
        >>> get_confidence_color(0.5)
        'FFFF00'
        >>> get_confidence_color(1.0)
        '00FF00'
        >>> get_confidence_color(0.25)
        'FF7F00'
        >>> get_confidence_color(0.75)
        '7FFF00'
    """
    # Input validation
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0 (inclusive), got {confidence}")
    
    # Clamp the value to the valid range in case of floating point precision issues
    confidence = max(0.0, min(1.0, confidence))
    
    # Color mapping based on confidence value
    if confidence <= 0.5:
        # Red to Yellow gradient (0.0 to 0.5)
        # R = 255 (constant), G increases from 0 to 255, B = 0 (constant)
        r = 255
        g = int(255 * (confidence / 0.5))
        b = 0
    else:
        # Yellow to Green gradient (0.5 to 1.0)
        # R decreases from 255 to 0, G = 255 (constant), B = 0 (constant)
        r = int(255 * ((1.0 - confidence) / 0.5))
        g = 255
        b = 0
    
    # Convert RGB values to hex string
    hex_color = f"{r:02X}{g:02X}{b:02X}"
    
    return hex_color


def get_confidence_color_gradient(confidence: Union[float, int], 
                                low_color: str = "FF0000",  # Red
                                mid_color: str = "FFFF00",  # Yellow
                                high_color: str = "00FF00") -> str:  # Green
    """
    Alternative implementation that allows custom gradient colors.
    
    Args:
        confidence: A float value between 0.0 and 1.0 representing the confidence level
        low_color: Hex color for low confidence (default: red)
        mid_color: Hex color for medium confidence (default: yellow)
        high_color: Hex color for high confidence (default: green)
        
    Returns:
        A 6-character hex color string
    """
    # Input validation
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0 (inclusive), got {confidence}")
    
    # Clamp the value to the valid range
    confidence = max(0.0, min(1.0, confidence))
    
    # Parse hex colors
    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert hex colors to RGB
    low_r, low_g, low_b = hex_to_rgb(low_color)
    mid_r, mid_g, mid_b = hex_to_rgb(mid_color)
    high_r, high_g, high_b = hex_to_rgb(high_color)
    
    # Interpolate between colors based on confidence
    if confidence <= 0.5:
        # Interpolate from low to mid color
        t = confidence * 2  # Normalize to 0-1 range for this segment
        r = int(low_r + t * (mid_r - low_r))
        g = int(low_g + t * (mid_g - low_g))
        b = int(low_b + t * (mid_b - low_b))
    else:
        # Interpolate from mid to high color
        t = (confidence - 0.5) * 2  # Normalize to 0-1 range for this segment
        r = int(mid_r + t * (high_r - mid_r))
        g = int(mid_g + t * (high_g - mid_g))
        b = int(mid_b + t * (high_b - mid_b))
    
    # Convert RGB values to hex string
    hex_color = f"{r:02X}{g:02X}{b:02X}"
    
    return hex_color


def get_reverse_confidence_color(confidence: Union[float, int]) -> str:
    """
    Alternative color mapping that emphasizes low confidence values more strongly.
    
    Args:
        confidence: A float value between 0.0 and 1.0 representing the confidence level
        
    Returns:
        A 6-character hex color string
    """
    # Input validation
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0 (inclusive), got {confidence}")
    
    # Clamp the value to the valid range
    confidence = max(0.0, min(1.0, confidence))
    
    # Reverse mapping: emphasize low confidence with stronger colors
    reversed_conf = 1.0 - confidence
    
    if reversed_conf <= 0.5:
        # Red to Yellow gradient for low confidence (0.0 to 0.5)
        r = 255
        g = int(255 * (reversed_conf / 0.5))
        b = 0
    else:
        # Yellow to Green gradient for high confidence (0.5 to 1.0)
        r = int(255 * ((1.0 - reversed_conf) / 0.5))
        g = 255
        b = 0
    
    # Convert RGB values to hex string
    hex_color = f"{r:02X}{g:02X}{b:02X}"
    
    return hex_color


# Predefined color mappings for common use cases
CONFIDENCE_COLOR_MAP = {
    'very_low': 'FF0000',    # Red (< 0.2)
    'low': 'FF4400',         # Orange-red (0.2-0.4)
    'medium': 'FFAA00',      # Orange (0.4-0.6)
    'high': 'FFFF00',        # Yellow (0.6-0.8)
    'very_high': '00FF00'    # Green (0.8-1.0)
}


def get_categorical_confidence_color(confidence: Union[float, int]) -> str:
    """
    Returns a categorical color based on confidence ranges.
    
    Args:
        confidence: A float value between 0.0 and 1.0 representing the confidence level
        
    Returns:
        A 6-character hex color string from predefined categories
    """
    # Input validation
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Confidence must be a number, got {type(confidence).__name__}")
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0.0 and 1.0 (inclusive), got {confidence}")
    
    # Determine category based on confidence
    if confidence < 0.2:
        return CONFIDENCE_COLOR_MAP['very_low']
    elif confidence < 0.4:
        return CONFIDENCE_COLOR_MAP['low']
    elif confidence < 0.6:
        return CONFIDENCE_COLOR_MAP['medium']
    elif confidence < 0.8:
        return CONFIDENCE_COLOR_MAP['high']
    else:
        return CONFIDENCE_COLOR_MAP['very_high']