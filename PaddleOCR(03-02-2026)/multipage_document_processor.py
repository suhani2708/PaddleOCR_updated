"""
Multi-page document processor for PaddleOCR.
Handles PDF files and multi-page images for comprehensive document processing.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from PIL import Image
import io
import tempfile
import os
from paddle_ocr_excel_exporter import PaddleOCRExcelExporter


class MultiPageDocumentProcessor:
    """
    Processor for multi-page documents (PDFs, TIFFs, etc.) with PaddleOCR.
    """
    
    def __init__(self, ocr_exporter: PaddleOCRExcelExporter):
        """
        Initialize the multi-page document processor.
        
        Args:
            ocr_exporter: Instance of PaddleOCRExcelExporter
        """
        self.ocr_exporter = ocr_exporter
        self.logger = logging.getLogger(__name__)
    
    def process_pdf(self, pdf_path: str, output_excel_path: str, 
                   dpi: int = 200, page_range: Optional[Tuple[int, int]] = None) -> str:
        """
        Process a PDF file and export to Excel with OCR results.
        
        Args:
            pdf_path: Path to the input PDF file
            output_excel_path: Path for the output Excel file
            dpi: DPI for image conversion (higher = better quality but slower)
            page_range: Optional tuple of (start_page, end_page) to process only specific pages
            
        Returns:
            Path to the created Excel file
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Validate input
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert PDF to images
        images = self._convert_pdf_to_images(pdf_path, dpi, page_range)
        
        # Process each page and collect results
        all_page_results = []
        for page_num, image in enumerate(images):
            self.logger.info(f"Processing page {page_num + 1}/{len(images)}")
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name)
                page_results = self.ocr_exporter.process_image(tmp_file.name)
                
                # Add page number to results
                for row in page_results:
                    for word_info in row:
                        word_info['page'] = page_num + 1
                
                all_page_results.extend(page_results)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
        
        # Create Excel workbook with all page results
        wb = self.ocr_exporter.create_excel_workbook(all_page_results)
        
        # Ensure output directory exists
        output_dir = Path(output_excel_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save workbook
        wb.save(output_excel_path)
        
        self.logger.info(f"Multi-page PDF processing completed: {output_excel_path}")
        return output_excel_path
    
    def _convert_pdf_to_images(self, pdf_path: str, dpi: int, 
                              page_range: Optional[Tuple[int, int]] = None) -> List[Image.Image]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for image conversion
            page_range: Optional tuple of (start_page, end_page)
            
        Returns:
            List of PIL Images, one for each page
        """
        try:
            # Use pdf2image to convert PDF to images
            if page_range:
                start_page, end_page = page_range
                # Convert to 0-indexed for pdf2image
                images = convert_from_path(
                    pdf_path, 
                    dpi=dpi,
                    first_page=start_page + 1,  # pdf2image uses 1-indexed
                    last_page=end_page + 1      # pdf2image uses 1-indexed
                )
            else:
                images = convert_from_path(pdf_path, dpi=dpi)
            
            return images
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def process_multipage_tiff(self, tiff_path: str, output_excel_path: str) -> str:
        """
        Process a multi-page TIFF file and export to Excel with OCR results.
        
        Args:
            tiff_path: Path to the input TIFF file
            output_excel_path: Path for the output Excel file
            
        Returns:
            Path to the created Excel file
        """
        self.logger.info(f"Processing multi-page TIFF: {tiff_path}")
        
        # Validate input
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
        
        # Open TIFF file and iterate through pages
        tiff_image = Image.open(tiff_path)
        all_page_results = []
        
        for page_num in range(tiff_image.n_frames):
            tiff_image.seek(page_num)
            page_image = tiff_image.copy()
            
            self.logger.info(f"Processing TIFF page {page_num + 1}")
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                page_image.save(tmp_file.name)
                page_results = self.ocr_exporter.process_image(tmp_file.name)
                
                # Add page number to results
                for row in page_results:
                    for word_info in row:
                        word_info['page'] = page_num + 1
                
                all_page_results.extend(page_results)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
        
        # Create Excel workbook with all page results
        wb = self.ocr_exporter.create_excel_workbook(all_page_results)
        
        # Ensure output directory exists
        output_dir = Path(output_excel_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save workbook
        wb.save(output_excel_path)
        
        self.logger.info(f"Multi-page TIFF processing completed: {output_excel_path}")
        return output_excel_path
    
    def process_batch_documents(self, document_paths: List[str], 
                              output_excel_path: str, 
                              dpi: int = 200) -> str:
        """
        Process a batch of documents (PDFs, images, etc.) and export to Excel.
        
        Args:
            document_paths: List of paths to document files
            output_excel_path: Path for the output Excel file
            dpi: DPI for image conversion (for PDFs)
            
        Returns:
            Path to the created Excel file
        """
        self.logger.info(f"Processing batch of {len(document_paths)} documents")
        
        all_results = []
        
        for doc_path in document_paths:
            self.logger.info(f"Processing document: {doc_path}")
            
            # Determine file type and process accordingly
            file_ext = Path(doc_path).suffix.lower()
            
            if file_ext in ['.pdf']:
                # Process as PDF
                images = self._convert_pdf_to_images(doc_path, dpi)
                
                for page_num, image in enumerate(images):
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        image.save(tmp_file.name)
                        page_results = self.ocr_exporter.process_image(tmp_file.name)
                        
                        # Add document and page info
                        for row in page_results:
                            for word_info in row:
                                word_info['document'] = Path(doc_path).name
                                word_info['page'] = page_num + 1
                        
                        all_results.extend(page_results)
                        os.unlink(tmp_file.name)
                        
            elif file_ext in ['.tiff', '.tif']:
                # Process as multi-page TIFF
                tiff_image = Image.open(doc_path)
                
                for page_num in range(tiff_image.n_frames):
                    tiff_image.seek(page_num)
                    page_image = tiff_image.copy()
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        page_image.save(tmp_file.name)
                        page_results = self.ocr_exporter.process_image(tmp_file.name)
                        
                        # Add document and page info
                        for row in page_results:
                            for word_info in row:
                                word_info['document'] = Path(doc_path).name
                                word_info['page'] = page_num + 1
                        
                        all_results.extend(page_results)
                        os.unlink(tmp_file.name)
                        
            else:
                # Process as single image
                results = self.ocr_exporter.process_image(doc_path)
                
                # Add document info
                for row in results:
                    for word_info in row:
                        word_info['document'] = Path(doc_path).name
                        word_info['page'] = 1
                
                all_results.extend(results)
        
        # Create Excel workbook with all results
        wb = self.ocr_exporter.create_excel_workbook(all_results)
        
        # Ensure output directory exists
        output_dir = Path(output_excel_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save workbook
        wb.save(output_excel_path)
        
        self.logger.info(f"Batch document processing completed: {output_excel_path}")
        return output_excel_path


def create_multipage_processor(ocr_exporter: PaddleOCRExcelExporter) -> MultiPageDocumentProcessor:
    """
    Factory function to create a multi-page document processor.
    
    Args:
        ocr_exporter: Instance of PaddleOCRExcelExporter
        
    Returns:
        MultiPageDocumentProcessor instance
    """
    return MultiPageDocumentProcessor(ocr_exporter)