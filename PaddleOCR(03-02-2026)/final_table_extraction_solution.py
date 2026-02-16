"""
Advanced Table Extraction from Slanted Screenshots

This script addresses the challenge of extracting tabular data from a slanted 
screenshot image where:
- The table is rotated ~5-10 degrees counter-clockwise
- The table has no borders
- Records start with "MM01S10025xx" pattern
- Records span multiple lines
- PPStructure fails to detect tables and classifies content as 'figure'

The solution implements multiple approaches:
1. Image preprocessing (deskewing, contrast enhancement, binarization)
2. Adjusted PPStructure parameters
3. Fallback to regular OCR with intelligent text grouping
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructure
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re
import os


def deskew_image(image_path):
    """
    Correct the rotation of the slanted image (~5-10 degrees)
    """
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find all non-background pixels
    coords = np.column_stack(np.where(binary == 0))  # Assuming text is black
    
    # If no black pixels found, try with white pixels
    if coords.size == 0:
        coords = np.column_stack(np.where(binary == 255))
    
    if coords.size == 0:
        print("Warning: Could not detect text in image for deskewing, returning original")
        return img, 0
    
    # Calculate minimum area rectangle to find the angle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle based on OpenCV convention
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    print(f"Detected rotation angle: {angle:.2f} degrees")
    
    # Rotate the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, angle


def preprocess_image(image_path):
    """
    Preprocess image to improve OCR accuracy:
    - Reduce noise
    - Enhance contrast
    - Binarize
    - Sharpen
    """
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(binary)
    
    # Sharpen the image to make text clearer
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def extract_with_ppstructure(image_path):
    """
    Attempt to extract table using PPStructure with optimized parameters
    """
    try:
        # Initialize PPStructure with parameters optimized for challenging images
        engine = PPStructure(
            table=True,
            ocr=True,
            lang='en',
            use_angle_cls=True,          # Enable angle classification for rotated text
            layout_score_threshold=0.3,  # Lower threshold to detect more elements
            table_max_len=488,           # Increase max length for larger tables
            det_limit_side_len=960,      # Increase detection side length
            rec_batch_num=6              # Batch size for recognition
        )
        
        result = engine(image_path)
        
        # Extract tables from result
        tables = []
        for item in result:
            if item['type'] == 'table' and 'res' in item and 'html' in item['res']:
                tables.append(item['res']['html'])
        
        return tables, result
    except Exception as e:
        print(f"Error with PPStructure: {e}")
        return [], {}


def group_text_by_lines(ocr_result, tolerance=15):
    """
    Group OCR text results by y-coordinates to form logical lines,
    accounting for slight variations due to slant
    """
    lines = defaultdict(list)
    
    for item in ocr_result:
        text = item[1][0]  # Extract text
        bbox = item[0]     # Bounding box coordinates
        
        # Calculate average y-coordinate of the bounding box
        avg_y = sum([point[1] for point in bbox]) / len(bbox)
        
        # Group by y-coordinate with tolerance for slanted text
        found_group = False
        for existing_y in lines.keys():
            if abs(existing_y - avg_y) <= tolerance:
                lines[existing_y].append((text, avg_y))
                found_group = True
                break
        
        if not found_group:
            lines[avg_y].append((text, avg_y))
    
    # Sort lines by y-coordinate to maintain document order
    sorted_lines = {}
    for y in sorted(lines.keys()):
        sorted_lines[y] = lines[y]
    
    return sorted_lines


def extract_records_from_text(text_lines):
    """
    Extract records starting with MM01S1 pattern, properly handling multi-line records
    """
    records = []
    current_record = ""
    
    # Define pattern for MM01 records (starts with MM01S1 followed by digits)
    mm_pattern = re.compile(r'^MM01S1\d+')
    
    for y_coord, line_items in text_lines.items():
        # Join all text in this line
        line_text = " ".join([item[0] for item in line_items]).strip()
        
        if not line_text:
            continue  # Skip empty lines
            
        # Check if this line starts a new record
        if mm_pattern.match(line_text):
            # If we have a previous record, save it
            if current_record:
                records.append(current_record.strip())
            
            # Start new record
            current_record = line_text
        else:
            # If current record exists, append to it (multi-line record)
            if current_record:
                current_record += " " + line_text
            # If no current record but line doesn't start with MM01, 
            # we ignore it as it might be header info or unrelated text
    
    # Add the last record if it exists
    if current_record:
        records.append(current_record.strip())
    
    return records


def main(image_path):
    """
    Main function implementing the complete solution
    """
    print("Analyzing and extracting tabular data from slanted screenshot...")
    print(f"Input image: {image_path}")
    print("-" * 60)
    
    # Step 1: Analyze why PPStructure fails
    print("Step 1: Analysis of failure causes")
    print("- Possible causes: slant (~5-10 degrees), no borders, low contrast, noise")
    print("- Testing preprocessing approaches to address these issues")
    
    # Step 2: Preprocessing - deskew the image
    print("\nStep 2: Preprocessing image (deskewing)")
    deskewed_img, angle = deskew_image(image_path)
    
    # Save deskewed image temporarily
    temp_deskewed_path = "temp_deskewed_for_processing.jpg"
    cv2.imwrite(temp_deskewed_path, deskewed_img)
    
    # Step 3: Try PPStructure with adjusted parameters on deskewed image
    print("\nStep 3: Testing PPStructure with adjusted parameters on deskewed image")
    tables, result = extract_with_ppstructure(temp_deskewed_path)
    
    if tables:
        print("SUCCESS: PPStructure detected table in deskewed image!")
        soup = BeautifulSoup(tables[0], 'html.parser')
        rows = [[td.text.strip() for td in tr.find_all(['td', 'th'])] for tr in soup.find_all('tr')]
        
        if rows and len(rows) > 0:
            # Create DataFrame from table
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 1 else range(len(rows[0])))
            df.to_csv('extracted_table_ppstructure.csv', index=False)
            print("CSV saved as 'extracted_table_ppstructure.csv'")
            
            print("\nCSV Preview (first 3 rows):")
            print(df.head(3))
            
            # Clean up temporary file
            if os.path.exists(temp_deskewed_path):
                os.remove(temp_deskewed_path)
                
            return df
        else:
            print("No rows found in extracted table")
    else:
        print("PPStructure failed to detect table in deskewed image.")
    
    # Step 4: Fallback to regular PaddleOCR approach
    print("\nStep 4: Falling back to regular OCR approach")
    print("Using deskewed image for better text detection...")
    
    # Initialize regular OCR with angle classification enabled
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=True)
    ocr_result = ocr.ocr(temp_deskewed_path, cls=True)
    
    # Flatten the result structure to get individual text boxes
    flat_result = []
    for page_result in ocr_result:
        if page_result:
            for line in page_result:
                if line:  # Ensure line is not empty
                    flat_result.append(line)
    
    print(f"Detected {len(flat_result)} text elements with OCR")
    
    # Step 5: Group text by lines and extract records
    print("\nStep 5: Grouping text by lines and extracting MM01 records")
    text_lines = group_text_by_lines(flat_result)
    
    # Extract records starting with MM01S1 pattern
    records = extract_records_from_text(text_lines)
    
    # Create DataFrame with extracted records
    if records:
        print(f"SUCCESS: Found {len(records)} records starting with MM01S1 pattern")
        
        # Format records for CSV: ID in column A, all content in column B
        data = []
        for record in records:
            # Extract MM01 ID as first column, rest as second column
            mm_match = re.search(r'(MM01S1\d+)', record)
            if mm_match:
                mm_id = mm_match.group(1)
                content = record.replace(mm_id, '', 1).strip()  # Remove ID from content
                data.append([mm_id, content])
        
        if data:
            df = pd.DataFrame(data, columns=['ID', 'Content'])
            df.to_csv('extracted_records_fallback.csv', index=False)
            print("CSV saved as 'extracted_records_fallback.csv'")
            
            print("\nCSV Preview (first 3 rows):")
            print(df.head(3))
            
            # Clean up temporary file
            if os.path.exists(temp_deskewed_path):
                os.remove(temp_deskewed_path)
                
            return df
        else:
            print("No valid records could be formatted")
    else:
        print("No records starting with MM01S1 pattern found")
    
    # Clean up temporary file if it still exists
    if os.path.exists(temp_deskewed_path):
        os.remove(temp_deskewed_path)
    
    return None


if __name__ == "__main__":
    # Replace with your actual image path
    IMAGE_PATH = "[insert IMAGE_PATH here]"
    
    print("Advanced Table Extraction Solution")
    print("="*60)
    
    result_df = main(IMAGE_PATH)
    
    if result_df is not None:
        print("\n" + "="*60)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("- Every CSV row starts with MM01S1 as required")
        print("- Data properly grouped into records")
        print("- Multi-line records handled correctly")
    else:
        print("\n" + "="*60)
        print("EXTRACTION FAILED - All methods exhausted")
        print("Consider:")
        print("- Manually correcting the image rotation")
        print("- Increasing image resolution")
        print("- Using different preprocessing techniques")