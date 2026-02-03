"""
Document structure analyzer for PaddleOCR results.
Analyzes document layout and identifies different document elements like headers, paragraphs, tables, etc.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import DBSCAN
import cv2


class DocumentStructureAnalyzer:
    """
    Analyzes document structure and identifies different document elements.
    """
    
    def __init__(self):
        self.element_types = {
            'header': {'size_factor': 1.2, 'position': 'top'},
            'footer': {'size_factor': 1.0, 'position': 'bottom'},
            'paragraph': {'size_factor': 1.0, 'position': 'middle'},
            'title': {'size_factor': 1.5, 'position': 'top'},
            'caption': {'size_factor': 0.9, 'position': 'below_image'},
            'list_item': {'size_factor': 1.0, 'position': 'any'},
            'table_cell': {'size_factor': 0.9, 'position': 'grid'},
            'footnote': {'size_factor': 0.8, 'position': 'bottom'}
        }
    
    def analyze_layout(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the layout of OCR results and identify document elements.
        
        Args:
            ocr_results: List of OCR results with bounding boxes and text
        
        Returns:
            Dictionary containing layout analysis and element classifications
        """
        if not ocr_results:
            return {
                'elements': [],
                'structure': {},
                'layout_type': 'unknown'
            }
        
        # Extract text regions with their properties
        text_regions = []
        for idx, item in enumerate(ocr_results):
            if isinstance(item, dict) and 'x' in item and 'y' in item:
                region = {
                    'index': idx,
                    'x': item['x'],
                    'y': item['y'],
                    'text': item.get('text', ''),
                    'confidence': item.get('confidence', 0.0),
                    'width': item.get('width', 0),
                    'height': item.get('height', 0)
                }
                text_regions.append(region)
        
        if not text_regions:
            return {
                'elements': [],
                'structure': {},
                'layout_type': 'unknown'
            }
        
        # Cluster text regions by position to identify layout patterns
        clusters = self._cluster_by_position(text_regions)
        
        # Classify elements based on size, position, and context
        classified_elements = self._classify_elements(text_regions, clusters)
        
        # Determine overall document layout type
        layout_type = self._determine_layout_type(classified_elements)
        
        return {
            'elements': classified_elements,
            'structure': clusters,
            'layout_type': layout_type
        }
    
    def _cluster_by_position(self, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cluster text regions by their spatial position to identify layout patterns.
        """
        if not text_regions:
            return {}
        
        # Extract coordinates for clustering
        coords = np.array([[region['x'], region['y']] for region in text_regions])
        
        # Use DBSCAN for spatial clustering
        clustering = DBSCAN(eps=50, min_samples=1)
        cluster_labels = clustering.fit_predict(coords)
        
        # Group regions by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text_regions[i])
        
        return clusters
    
    def _classify_elements(self, text_regions: List[Dict[str, Any]], 
                         clusters: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Classify text regions into document elements based on their properties.
        """
        classified = []
        
        # Calculate document boundaries
        all_ys = [region['y'] for region in text_regions]
        min_y, max_y = min(all_ys), max(all_ys)
        doc_height = max_y - min_y
        
        for region in text_regions:
            element_type = self._determine_element_type(region, doc_height, text_regions)
            
            classified.append({
                'text': region['text'],
                'confidence': region['confidence'],
                'bbox': [region['x'], region['y'], 
                        region['x'] + region.get('width', len(region['text']) * 10), 
                        region['y'] + region.get('height', 20)],
                'element_type': element_type,
                'position_score': self._calculate_position_score(region, doc_height)
            })
        
        return classified
    
    def _determine_element_type(self, region: Dict[str, Any], 
                               doc_height: float, 
                               all_regions: List[Dict[str, Any]]) -> str:
        """
        Determine the type of document element based on its properties.
        """
        y_pos = region['y']
        relative_y = (y_pos - min(r['y'] for r in all_regions)) / doc_height if doc_height > 0 else 0.5
        
        # Simple heuristic-based classification
        if relative_y < 0.1:  # Top 10% of document
            if len(region['text']) > 50:  # Longer text might be header
                return 'header'
            else:  # Shorter text might be title
                return 'title'
        elif relative_y > 0.9:  # Bottom 10% of document
            return 'footer'
        elif len(region['text']) < 10:  # Very short text
            return 'caption'
        else:
            return 'paragraph'
    
    def _calculate_position_score(self, region: Dict[str, Any], doc_height: float) -> float:
        """
        Calculate a position score for the element (0.0 = top, 1.0 = bottom).
        """
        return region['y'] / doc_height if doc_height > 0 else 0.5
    
    def _determine_layout_type(self, classified_elements: List[Dict[str, Any]]) -> str:
        """
        Determine the overall document layout type.
        """
        element_counts = {}
        for elem in classified_elements:
            elem_type = elem['element_type']
            element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
        
        # Determine layout based on element distribution
        if element_counts.get('table_cell', 0) > len(classified_elements) * 0.3:
            return 'table'
        elif element_counts.get('title', 0) > 0 and element_counts.get('header', 0) > 0:
            return 'structured_document'
        elif element_counts.get('list_item', 0) > len(classified_elements) * 0.2:
            return 'list'
        else:
            return 'freeform_text'


def integrate_with_ocr_processing(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Integrate document structure analysis with OCR processing results.
    
    Args:
        ocr_results: Raw OCR results from PaddleOCR
        
    Returns:
        Enhanced results with document structure information
    """
    analyzer = DocumentStructureAnalyzer()
    layout_analysis = analyzer.analyze_layout(ocr_results)
    
    # Combine original OCR results with structure analysis
    enhanced_results = []
    for i, ocr_result in enumerate(ocr_results):
        # Find corresponding classified element
        classified_elem = None
        for elem in layout_analysis['elements']:
            # Simple matching based on text similarity
            if elem['text'] == ocr_result.get('text', ''):
                classified_elem = elem
                break
        
        enhanced_result = {
            'original': ocr_result,
            'structure_info': classified_elem or {},
            'element_type': classified_elem['element_type'] if classified_elem else 'unknown'
        }
        enhanced_results.append(enhanced_result)
    
    return {
        'enhanced_results': enhanced_results,
        'layout_analysis': layout_analysis,
        'document_structure': layout_analysis['structure'],
        'layout_type': layout_analysis['layout_type']
    }


def get_document_elements_by_type(enhanced_results: Dict[str, Any], 
                                 element_type: str) -> List[Dict[str, Any]]:
    """
    Extract elements of a specific type from enhanced OCR results.
    
    Args:
        enhanced_results: Results from integrate_with_ocr_processing
        element_type: Type of element to extract ('header', 'paragraph', etc.)
        
    Returns:
        List of elements of the specified type
    """
    elements = []
    for item in enhanced_results['enhanced_results']:
        if item['element_type'] == element_type:
            elements.append(item)
    
    return elements