import csv
import numpy as np
import cv2
from paddleocr import PaddleOCR

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    # Upscale for better resolution
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale and lightly sharpen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened

def get_row_overlap(item, row_items):
    """Calculates if an item belongs to a row based on Y-axis overlap."""
    if not row_items: return 0
    # Use the average Y range of the row
    row_y_min = min(r['y_min'] for r in row_items)
    row_y_max = max(r['y_max'] for r in row_items)
    
    overlap = min(item['y_max'], row_y_max) - max(item['y_min'], row_y_min)
    item_height = item['y_max'] - item['y_min']
    return overlap / item_height if item_height > 0 else 0

def extract_structured_rows(image_path, output_csv):
    img = preprocess_image(image_path)
    if img is None: return

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False,
                    det_db_thresh=0.2, 
                    det_db_box_thresh=0.3,
                    det_limit_side_len=2500)

    result = ocr.ocr(img, cls=True)
    if not result or not result[0]: return

    items = []
    for detection in result[0]:
        bbox, (text, conf) = detection
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        items.append({
            "text": text.strip(),
            "x_start": min(x_coords),
            "y_min": min(y_coords),
            "y_max": max(y_coords),
            "y_center": np.mean(y_coords)
        })

    # 1. Sort by Y first
    items.sort(key=lambda x: x['y_center'])

    # 2. Group into rows based on vertical overlap (> 50%)
    rows = []
    for item in items:
        assigned = False
        for row in rows:
            if get_row_overlap(item, row) > 0.5:
                row.append(item)
                assigned = True
                break
        if not assigned:
            rows.append([item])

    # 3. Sort each row by X-coordinate
    structured_data = []
    for row in rows:
        row.sort(key=lambda x: x['x_start'])
        structured_data.append([item['text'] for item in row])

    # 4. Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(structured_data)

    print(f"✅ Extracted {len(structured_data)} rows.")

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\Statista-FormF-RKJJ-NA-1929755_d----parth.jpg"
    OUTPUT_CSV = "refined_output.csv"
    extract_structured_rows(IMAGE_PATH, OUTPUT_CSV)