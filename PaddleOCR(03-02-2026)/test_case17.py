import cv2
import numpy as np
import xlsxwriter
from paddleocr import PaddleOCR

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    # Upscale for better OCR accuracy
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(gray, -1, kernel)

def get_row_overlap(item, row_items):
    if not row_items: return 0
    row_y_min = min(r['y_min'] for r in row_items)
    row_y_max = max(r['y_max'] for r in row_items)
    overlap = min(item['y_max'], row_y_max) - max(item['y_min'], row_y_min)
    item_height = item['y_max'] - item['y_min']
    return overlap / item_height if item_height > 0 else 0

def extract_to_excel(image_path, output_xlsx):
    img = preprocess_image(image_path)
    if img is None: 
        print("Error: Image not found.")
        return

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False,
                    det_db_thresh=0.1, 
                    det_db_box_thresh=0.2,
                    det_limit_side_len=2500)

    result = ocr.ocr(img, cls=True)
    if not result or result[0] is None: return

    items = []
    for line in result:
        for detection in line:
            bbox, (text, conf) = detection
            y_coords = [p[1] for p in bbox]
            x_coords = [p[0] for p in bbox]
            
            h = max(y_coords) - min(y_coords)
            items.append({
                "text": text.strip(),
                "conf": conf,
                "x_start": min(x_coords),
                "y_min": min(y_coords),
                "y_max": max(y_coords),
                "y_center": np.mean(y_coords),
                "height": h
            })

    # 1. Group items into physical rows
    items.sort(key=lambda x: x['y_center'])
    rows = []
    for item in items:
        assigned = False
        for row in rows:
            if get_row_overlap(item, row) > 0.1:
                row.append(item)
                assigned = True
                break
        if not assigned: rows.append([item])

    # 2. Create Excel with Multi-Color Confidence Logic
    workbook = xlsxwriter.Workbook(output_xlsx)
    worksheet = workbook.add_worksheet("OCR_Output")
    
    # Define Formats
    fmt_red = workbook.add_format({'bg_color': '#FF0000', 'font_color': '#FFFFFF', 'border': 1}) # < 50%
    fmt_green = workbook.add_format({'bg_color': '#00B050', 'font_color': '#FFFFFF', 'border': 1}) # 50% - 80%
    fmt_yellow = workbook.add_format({'bg_color': '#FFFF99', 'border': 1})                       # > 80%

    current_excel_row = 0
    
    for i in range(len(rows)):
        # Sort words in the current row from left to right
        rows[i].sort(key=lambda x: x['x_start'])
        
        # Write the row data with confidence-based coloring
        for c_idx, item in enumerate(rows[i]):
            confidence = item['conf']
            
            # Tiered Logic
            if confidence < 0.50:
                cell_fmt = fmt_red
            elif 0.50 <= confidence <= 0.80:
                cell_fmt = fmt_green
            else: # > 0.80
                cell_fmt = fmt_yellow
                
            worksheet.write(current_excel_row, c_idx, item['text'], cell_fmt)
        
        # LOGIC: Adaptive Gap
        if i < len(rows) - 1:
            curr_row_y = np.mean([item['y_center'] for item in rows[i]])
            next_row_y = np.mean([item['y_center'] for item in rows[i+1]])
            avg_text_height = np.mean([item['height'] for item in rows[i]])
            vertical_distance = next_row_y - curr_row_y
            
            if vertical_distance > (avg_text_height * 2.5):
                current_excel_row += 2  # Skips a row
            else:
                current_excel_row += 1
        else:
            current_excel_row += 1

    workbook.close()
    print(f"✅ Success! Data saved with tiered highlights to {output_xlsx}")

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\Statista-FormF-RKJJ-NA-1929755_d----parth.jpg" # Update this with your actual image path
    OUTPUT_EXCEL = "confidence_highlighted_output.xlsx"
    extract_to_excel(IMAGE_PATH, OUTPUT_EXCEL)