import cv2
import numpy as np
import math
from paddleocr import PaddleOCR, PPStructure
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re


def deskew_image(image_path):
    """
    Deskew the image to correct rotation
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find all white pixels
    coords = np.column_stack(np.where(thresh == 0))
    
    # Calculate angle
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
    Preprocess image to improve OCR accuracy
    """
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(binary)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened


def extract_table_with_ppstructure(image_path):
    """
    Try to extract table using PPStructure with adjusted parameters
    """
    try:
        # Initialize PPStructure with adjusted parameters
        engine = PPStructure(
            table=True,
            ocr=True,
            lang='en',
            use_angle_cls=True,  # Enable angle classification
            layout_score_threshold=0.3,  # Lower threshold to detect more elements
            table_max_len=488,  # Increase max length for larger tables
            det_limit_side_len=960,  # Increase detection side length
            rec_batch_num=6  # Batch size for recognition
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
        return [], []


def group_text_by_lines(ocr_result, angle_threshold=5):
    """
    Group OCR text results by y-coordinates to form lines
    """
    lines = defaultdict(list)
    
    for item in ocr_result:
        text = item[1][0]  # Extract text
        bbox = item[0]  # Bounding box coordinates
        
        # Calculate average y-coordinate of the bounding box
        avg_y = sum([point[1] for point in bbox]) / len(bbox)
        
        # Group by rounded y-coordinate to handle slight variations due to slant
        rounded_y = round(avg_y)
        
        # Handle slant by grouping nearby y-coordinates
        found_group = False
        for existing_y in lines.keys():
            if abs(existing_y - avg_y) <= angle_threshold:
                lines[existing_y].append((text, bbox))
                found_group = True
                break
        
        if not found_group:
            lines[rounded_y].append((text, bbox))
    
    # Sort lines by y-coordinate
    sorted_lines = {}
    for y in sorted(lines.keys()):
        sorted_lines[y] = lines[y]
    
    return sorted_lines


def extract_records_from_text(text_lines):
    """
    Extract records starting with MM01S1 pattern
    """
    records = []
    current_record = ""
    
    # Define pattern for MM01 records
    mm_pattern = r'^MM01S1\d+'
    
    for y_coord, line_items in text_lines.items():
        # Join all text in this line
        line_text = " ".join([item[0] for item in line_items])
        
        # Check if this line starts a new record
        if re.match(mm_pattern, line_text.strip()):
            # If we have a previous record, save it
            if current_record:
                records.append(current_record.strip())
            
            # Start new record
            current_record = line_text
        else:
            # Append to current record if it exists
            if current_record:
                current_record += " " + line_text
            else:
                # If no current record but line doesn't start with MM01, skip or handle differently
                continue
    
    # Add the last record if it exists
    if current_record:
        records.append(current_record.strip())
    
    return records


def main(image_path):
    """
    Main function to extract table data from image
    """
    print("Starting table extraction...")
    
    # Step 1: Try preprocessing to deskew and enhance image
    print("Preprocessing image...")
    processed_img = preprocess_image(image_path)
    
    # Save processed image temporarily
    temp_path = "temp_processed.jpg"
    cv2.imwrite(temp_path, processed_img)
    
    # Step 2: Try PPStructure with adjusted parameters
    print("Attempting table extraction with PPStructure...")
    tables, full_result = extract_table_with_ppstructure(temp_path)
    
    if tables:
        print("Table successfully extracted with PPStructure!")
        # Process the HTML table
        soup = BeautifulSoup(tables[0], 'html.parser')
        rows = [[td.text.strip() for td in tr.find_all(['td', 'th'])] for tr in soup.find_all('tr')]
        
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 0 else [])
            df.to_csv('extracted_ppstructure.csv', index=False)
            print("CSV saved as 'extracted_ppstructure.csv'")
            return df
        else:
            print("No rows found in extracted table")
    
    # Step 3: If PPStructure fails, fall back to regular OCR approach
    print("PPStructure failed, falling back to regular OCR approach...")
    
    # Initialize regular OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=True)
    ocr_result = ocr.ocr(temp_path, cls=True)
    
    # Flatten the result structure
    flat_result = []
    for page_result in ocr_result:
        if page_result:
            flat_result.extend(page_result)
    
    # Group text by lines
    text_lines = group_text_by_lines(flat_result)
    
    # Extract records starting with MM01S1
    records = extract_records_from_text(text_lines)
    
    # Create DataFrame with records
    if records:
        # Split each record into ID and content
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
            return df
        else:
            print("No valid records found")
            return None
    else:
        print("No records extracted")
        return None


# Alternative approach using deskewing
def main_with_deskew(image_path):
    """
    Alternative main function that includes deskewing
    """
    print("Starting table extraction with deskewing...")
    
    # Deskew the image
    deskewed_img, angle = deskew_image(image_path)
    print(f"Image deskewed by {angle:.2f} degrees")
    
    # Save deskewed image temporarily
    temp_path = "temp_deskewed.jpg"
    cv2.imwrite(temp_path, deskewed_img)
    
    # Try PPStructure on deskewed image
    print("Attempting table extraction with PPStructure on deskewed image...")
    tables, full_result = extract_table_with_ppstructure(temp_path)
    
    if tables:
        print("Table successfully extracted with PPStructure on deskewed image!")
        # Process the HTML table
        soup = BeautifulSoup(tables[0], 'html.parser')
        rows = [[td.text.strip() for td in tr.find_all(['td', 'th'])] for tr in soup.find_all('tr')]
        
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0] if len(rows) > 0 else [])
            df.to_csv('extracted_deskewed_ppstructure.csv', index=False)
            print("CSV saved as 'extracted_deskewed_ppstructure.csv'")
            return df
        else:
            print("No rows found in extracted table")
    
    # If PPStructure still fails, use fallback approach
    print("Falling back to regular OCR approach on deskewed image...")
    
    # Initialize regular OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=True)
    ocr_result = ocr.ocr(temp_path, cls=True)
    
    # Flatten the result structure
    flat_result = []
    for page_result in ocr_result:
        if page_result:
            flat_result.extend(page_result)
    
    # Group text by lines
    text_lines = group_text_by_lines(flat_result)
    
    # Extract records starting with MM01S1
    records = extract_records_from_text(text_lines)
    
    # Create DataFrame with records
    if records:
        # Split each record into ID and content
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
            df.to_csv('extracted_deskewed_fallback.csv', index=False)
            print("CSV saved as 'extracted_deskewed_fallback.csv'")
            return df
        else:
            print("No valid records found")
            return None
    else:
        print("No records extracted")
        return None


if __name__ == "__main__":
    # Replace with your actual image path
    IMAGE_PATH = "[insert IMAGE_PATH here]"
    
    # Try both approaches
    print("=== Approach 1: Preprocessing only ===")
    df1 = main(IMAGE_PATH)
    
    if df1 is not None and not df1.empty:
        print("\nPreview of extracted data:")
        print(df1.head())
    else:
        print("\n=== Approach 2: Deskewing + Processing ===")
        df2 = main_with_deskew(IMAGE_PATH)
        
        if df2 is not None and not df2.empty:
            print("\nPreview of extracted data:")
            print(df2.head())
        else:
            print("\nBoth approaches failed. Manual inspection needed.")