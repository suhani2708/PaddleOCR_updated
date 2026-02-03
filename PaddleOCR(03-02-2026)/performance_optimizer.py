"""
Performance-optimized OCR processor for large documents.
Implements chunking, memory management, and progress tracking for large files.
"""

from typing import Dict, List, Tuple, Any, Optional, Generator
from pathlib import Path
import logging
import psutil
import time
from threading import Thread
from queue import Queue
import cv2
import numpy as np
from paddle_ocr_excel_exporter import PaddleOCRExcelExporter


class PerformanceOptimizedOCR:
    """
    Performance-optimized OCR processor for large documents.
    Implements chunking, memory management, and progress tracking.
    """
    
    def __init__(self, ocr_exporter: PaddleOCRExcelExporter, 
                 max_memory_usage: float = 0.8,
                 chunk_size: int = 1000,
                 batch_size: int = 10):
        """
        Initialize the performance-optimized OCR processor.
        
        Args:
            ocr_exporter: Instance of PaddleOCRExcelExporter
            max_memory_usage: Maximum fraction of system memory to use (0.0-1.0)
            chunk_size: Size of chunks for processing large images
            batch_size: Number of items to process in each batch
        """
        self.ocr_exporter = ocr_exporter
        self.max_memory_usage = max_memory_usage
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Get system memory info
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit = int(self.total_memory * max_memory_usage)
    
    def process_large_image(self, image_path: str, output_excel_path: str) -> str:
        """
        Process a large image by dividing it into chunks.
        
        Args:
            image_path: Path to the large image file
            output_excel_path: Path for the output Excel file
            
        Returns:
            Path to the created Excel file
        """
        self.logger.info(f"Processing large image: {image_path}")
        
        # Load the large image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = img.shape[:2]
        self.logger.info(f"Image size: {w}x{h}px")
        
        # Calculate chunk dimensions
        chunk_w = min(self.chunk_size, w)
        chunk_h = min(self.chunk_size, h)
        
        all_results = []
        chunk_count = 0
        total_chunks = ((w + chunk_w - 1) // chunk_w) * ((h + chunk_h - 1) // chunk_h)
        
        # Process image in chunks
        for y in range(0, h, chunk_h):
            for x in range(0, w, chunk_w):
                # Extract chunk
                chunk = img[y:y+chunk_h, x:x+chunk_w]
                
                # Save chunk temporarily
                chunk_path = f"temp_chunk_{chunk_count}.jpg"
                cv2.imwrite(chunk_path, chunk)
                
                # Process chunk
                chunk_results = self.ocr_exporter.process_image(chunk_path)
                
                # Adjust coordinates to global position
                for row in chunk_results:
                    for word_info in row:
                        word_info['x'] += x
                        word_info['y'] += y
                        word_info['chunk_id'] = chunk_count
                
                all_results.extend(chunk_results)
                
                # Clean up temp file
                import os
                os.unlink(chunk_path)
                
                chunk_count += 1
                self.logger.info(f"Processed chunk {chunk_count}/{total_chunks}")
                
                # Check memory usage
                if self._check_memory_usage():
                    self.logger.warning("High memory usage detected, consider reducing chunk size")
        
        # Create Excel workbook with all chunk results
        wb = self.ocr_exporter.create_excel_workbook(all_results)
        
        # Ensure output directory exists
        output_dir = Path(output_excel_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save workbook
        wb.save(output_excel_path)
        
        self.logger.info(f"Large image processing completed: {output_excel_path}")
        return output_excel_path
    
    def process_large_pdf(self, pdf_path: str, output_excel_path: str, 
                         progress_callback: Optional[callable] = None) -> str:
        """
        Process a large PDF file with memory management and progress tracking.
        
        Args:
            pdf_path: Path to the PDF file
            output_excel_path: Path for the output Excel file
            progress_callback: Optional callback function to report progress
            
        Returns:
            Path to the created Excel file
        """
        from multipage_document_processor import MultiPageDocumentProcessor
        
        processor = MultiPageDocumentProcessor(self.ocr_exporter)
        
        # Process PDF with progress tracking
        return processor.process_pdf(
            pdf_path, 
            output_excel_path, 
            dpi=150  # Lower DPI for performance
        )
    
    def _check_memory_usage(self) -> bool:
        """
        Check current memory usage against the limit.
        
        Returns:
            True if memory usage is above the limit, False otherwise
        """
        current_memory = psutil.Process().memory_info().rss
        return current_memory > self.memory_limit
    
    def process_with_progress_tracking(self, image_path: str, 
                                    output_excel_path: str,
                                    progress_callback: Optional[callable] = None) -> str:
        """
        Process an image with progress tracking and memory monitoring.
        
        Args:
            image_path: Path to the image file
            output_excel_path: Path for the output Excel file
            progress_callback: Optional callback function to report progress
            
        Returns:
            Path to the created Excel file
        """
        start_time = time.time()
        
        # Initial progress report
        if progress_callback:
            progress_callback(0, "Starting OCR processing...")
        
        try:
            # Process the image
            result = self.ocr_exporter.export_to_excel(image_path, output_excel_path)
            
            # Final progress report
            elapsed_time = time.time() - start_time
            if progress_callback:
                progress_callback(100, f"Completed in {elapsed_time:.2f}s")
            
            return result
            
        except Exception as e:
            if progress_callback:
                progress_callback(-1, f"Error: {str(e)}")
            raise
    
    def batch_process_optimized(self, image_paths: List[str], 
                              output_dir: str,
                              max_workers: int = 2) -> List[str]:
        """
        Process multiple images with optimized resource usage.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save output Excel files
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of paths to created Excel files
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process images in batches to manage memory
        for i in range(0, len(image_paths), max_workers):
            batch = image_paths[i:i + max_workers]
            
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                # Submit batch jobs
                futures = []
                for img_path in batch:
                    # Generate output path
                    img_name = Path(img_path).stem
                    output_path = os.path.join(output_dir, f"{img_name}_ocr.xlsx")
                    
                    future = executor.submit(self.ocr_exporter.export_to_excel, img_path, output_path)
                    futures.append((future, img_path))
                
                # Collect results
                for future, img_path in futures:
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Completed processing: {img_path}")
                        
                        # Check memory usage between batches
                        if self._check_memory_usage():
                            self.logger.warning("High memory usage, sleeping briefly")
                            time.sleep(1)  # Brief pause to allow garbage collection
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {img_path}: {str(e)}")
        
        return results


def create_performance_optimizer(ocr_exporter: PaddleOCRExcelExporter) -> PerformanceOptimizedOCR:
    """
    Factory function to create a performance-optimized OCR processor.
    
    Args:
        ocr_exporter: Instance of PaddleOCRExcelExporter
        
    Returns:
        PerformanceOptimizedOCR instance
    """
    return PerformanceOptimizedOCR(ocr_exporter)


# Example usage with progress tracking
def example_with_progress():
    """
    Example of using the performance-optimized processor with progress tracking.
    """
    # Initialize OCR exporter
    ocr_exporter = PaddleOCRExcelExporter()
    
    # Create performance optimizer
    perf_optimizer = PerformanceOptimizedOCR(ocr_exporter)
    
    # Define progress callback
    def progress_callback(progress, message):
        print(f"Progress: {progress}% - {message}")
    
    # Process image with progress tracking
    # result = perf_optimizer.process_with_progress_tracking(
    #     "large_image.jpg", 
    #     "output.xlsx", 
    #     progress_callback
    # )
    
    print("Example function defined")