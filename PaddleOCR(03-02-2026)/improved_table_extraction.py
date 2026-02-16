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
    Correct the rotation of the slanted image
    """
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find all white pixels (text)
    coords = np.column_stack(np.where(binary == 0))  # Assuming text is black
    
    # Calculate minimum area rectangle
    if coords.size == 0:
        # If no black pixels found, try with white pixels
        coords = np.column_stack(np.where(binary == 255))
    
    if coords.size == 0:
        print("Could not detect text in image for deskewing")
        return img, 0
    
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle based on OpenCV convention
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, angle


def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy
    """
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(binary)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def extract_with_ppstructure(image_path):
    """
    Attempt to extract table using PPStructure with adjusted parameters
    """
    try:
        engine = PPStructure(
            table=True,
            ocr=True,
            lang='en',
            use_angle_cls=True,
            layout_score_threshold=0.3,
            table_max_len=488,
            det_limit_side_len=960,
            rec_batch_num=6
        )
        
        result = engine(image_path)
        
        # Look for table results
        tables = []
        for item in result:
            if item['type'] == 'table' and 'res' in item and 'html' in item['res']:
                tables.append(item['res']['html'])
        
        return tables, result
    except Exception as e:
        print(f"Error with PPStructure: {e}")
        return [], []


def group_text_by_lines(ocr_result, tolerance=10):
    """
    Group OCR text results by y-coordinates to form logical lines
    """
    lines = defaultdict(list)
    
    for item in ocr_result:
        text = item[1][0]  # Extract text
        bbox = item[0]  # Bounding box coordinates
        
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
    
    # Sort lines by y-coordinate
    sorted_lines = {}
    for y in sorted(lines.keys()):
        sorted_lines[y] = lines[y]
    
    return sorted_lines


def extract_records_from_text(text_lines):
    """
    Extract records starting with MM01S1 pattern, handling multi-line records
    """
    records = []
    current_record = ""
    
    # Define pattern for MM01 records
    mm_pattern = r'^MM01S1\d+'
    
    for y_coord, line_items in text_lines.items():
        # Join all text in this line
        line_text = " ".join([item[0] for item in line_items]).strip()
        
        if not line_text:
            continue
            
        # Check if this line starts a new record
        if re.match(mm_pattern, line_text):
            # If we have a previous record, save it
            if current_record:
                records.append(current_record.strip())
            
            # Start new record
            current_record = line_text
        else:
            # If current record exists, append to it (multi-line record)
            if current_record:
                current_record += " " + line_text
            else:
                # If no current record but line doesn't start with MM01, 
                # it might be continuation of previous record that wasn't captured properly
                # For now, we'll skip these
                continue
    
    # Add the last record if it exists
    if current_record:
        records.append(current_record.strip())
    
    return records


def main(image_path):
    """
    Main function to extract tabular data from slanted screenshot
    """
    print("Step 1: Analyzing why PPStructure fails...")
    print("- Possible causes: slant (~5-10 degrees), no borders, low contrast")
    
    # Step 2: Preprocessing (deskew, enhance contrast, binarize, sharpen)
    print("\nStep 2: Preprocessing image...")
    
    # Try deskewing first
    deskewed_img, angle = deskew_image(image_path)
    print(f"Deskewed image by {angle:.2f} degrees")
    
    # Save deskewed image temporarily
    temp_deskewed_path = "temp_deskewed.jpg"
    cv2.imwrite(temp_deskewed_path, deskewed_img)
    
    # Also try preprocessing without deskewing
    processed_img = preprocess_image(image_path)
    temp_processed_path = "temp_processed.jpg"
    cv2.imwrite(temp_processed_path, processed_img)
    
    # Step 3: Try PPStructure with adjusted parameters on both images
    print("\nStep 3: Testing PPStructure with adjusted parameters...")
    
    # Try on deskewed image
    print("Testing on deskewed image...")
    tables_deskewed, result_deskewed = extract_with_ppstructure(temp_deskewed_path)
    
    if tables_deskewed:
        print("Success! Table detected in deskewed image.")
        soup = BeautifulSoup(tables_deskewed[0], 'html.parser')
        rows = [[td.text.strip() for td in tr.find_all(['td', 'th'])] for tr in soup.find_all('tr')]
        
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 0 else [])
            df.to_csv('extracted_table_ppstructure.csv', index=False)
            print("CSV saved as 'extracted_table_ppstructure.csv'")
            
            print("\nCSV Preview (first 3 rows):")
            print(df.head(3))
            return df
    else:
        print("PPStructure failed on deskewed image.")
    
    # Try on processed image
    print("Testing on processed image...")
    tables_processed, result_processed = extract_with_ppstructure(temp_processed_path)
    
    if tables_processed:
        print("Success! Table detected in processed image.")
        soup = BeautifulSoup(tables_processed[0], 'html.parser')
        rows = [[td.text.strip() for td in tr.find_all(['td', 'th'])] for tr in soup.find_all('tr')]
        
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 0 else [])
            df.to_csv('extracted_table_ppstructure.csv', index=False)
            print("CSV saved as 'extracted_table_ppstructure.csv'")
            
            print("\nCSV Preview (first 3 rows):")
            print(df.head(3))
            return df
    else:
        print("PPStructure also failed on processed image.")
    
    # Step 4: Fallback to regular PaddleOCR approach
    print("\nStep 4: Falling back to regular PaddleOCR approach...")
    
    # Use deskewed image for OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=True)
    ocr_result = ocr.ocr(temp_deskewed_path, cls=True)
    
    # Flatten the result structure
    flat_result = []
    for page_result in ocr_result:
        if page_result:
            for line in page_result:
                if line:
                    flat_result.append(line)
    
    # Step 5: Group text by lines and extract records
    print("Grouping text by lines and extracting records...")
    text_lines = group_text_by_lines(flat_result)
    
    # Extract records starting with MM01S1
    records = extract_records_from_text(text_lines)
    
    # Create DataFrame with records
    if records:
        print(f"Found {len(records)} records starting with MM01S1")
        
        # Format records for CSV
        data = []
        for record in records:
            # Extract MM01 ID as first column, rest as second column
            mm_match = re.search(r'(MM01S1\d+)', record)
            if mm_match:
                mm_id = mm_match.group(1)
                content = record.replace(mm_id, '', 1).strip()
                data.append([mm_id, content])
        
        if data:
            df = pd.DataFrame(data, columns=['ID', 'Content'])
            df.to_csv('extracted_fallback.csv', index=False)
            print("CSV saved as 'extracted_fallback.csv'")
            
            print("\nCSV Preview (first 3 rows):")
            print(df.head(3))
            
            # Clean up temporary files
            if os.path.exists(temp_deskewed_path):
                os.remove(temp_deskewed_path)
            if os.path.exists(temp_processed_path):
                os.remove(temp_processed_path)
                
            return df
        else:
            print("No valid records found")
    else:
        print("No records extracted with fallback method")
    
    # Clean up temporary files
    if os.path.exists(temp_deskewed_path):
        os.remove(temp_deskewed_path)
    if os.path.exists(temp_processed_path):
        os.remove(temp_processed_path)
    
    return None


if __name__ == "__main__":
    # Replace with your actual image path
    IMAGE_PATH = "[insert IMAGE_PATH here]"
    
    print("Troubleshooting PaddleOCR table extraction for slanted screenshot...")
    print("="*60)
    
    df = main(IMAGE_PATH)
    
    if df is not None:
        print("\n" + "="*60)
        print("SUCCESS: Table data extracted successfully!")
        print("Check the generated CSV file for results.")
    else:
        print("\n" + "="*60)
        print("FAILURE: Could not extract table data with any method.")
        print("Consider manual correction or adjusting preprocessing parameters.")