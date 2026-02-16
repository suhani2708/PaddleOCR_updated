from paddleocr import PaddleOCR
import csv
import os
import re
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='en',
    use_gpu=False,
    det_db_score_mode='slow' # Better for dense text
)

IMAGE_PATH = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\Statista-FormF-RKJJ-NA-1929751_d.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ ERROR: Image not found.")
    exit(1)

try:
    result = ocr.ocr(IMAGE_PATH, cls=True)
    
    # 1. Extract Data with full bounding box info
    data_items = []
    for line in result[0]:
        bbox = line[0] # [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        text = line[1][0].strip()
        conf = line[1][1]
        
        if conf < 0.30 or not text: continue
        
        y_min = min(p[1] for p in bbox)
        y_max = max(p[1] for p in bbox)
        x_min = min(p[0] for p in bbox)
        
        data_items.append({
            'y_min': y_min,
            'y_max': y_max,
            'y_center': (y_min + y_max) / 2,
            'x_min': x_min,
            'text': text
        })

    # 2. Group items into rows based on vertical overlap
    # Sort by center Y coordinate
    data_items.sort(key=lambda x: x['y_center'])
    
    grouped_rows = []
    if data_items:
        current_row = [data_items[0]]
        
        for i in range(1, len(data_items)):
            item = data_items[i]
            # Check if current item overlaps vertically with the average of the current row
            row_y_min = np.mean([row_item['y_min'] for row_item in current_row])
            row_y_max = np.mean([row_item['y_max'] for row_item in current_row])
            
            # Calculate overlap
            overlap = min(item['y_max'], row_y_max) - max(item['y_min'], row_y_min)
            item_height = item['y_max'] - item['y_min']
            
            # If overlap is more than 40% of the text height, it's the same row
            if overlap > (item_height * 0.4):
                current_row.append(item)
            else:
                grouped_rows.append(current_row)
                current_row = [item]
        grouped_rows.append(current_row)

    # 3. Process and Filter Rows
    final_output = []
    for row in grouped_rows:
        # Sort items in row from left to right
        row.sort(key=lambda x: x['x_min'])
        row_text = "\t".join([item['text'] for item in row])
        
        # Pattern match for your specific ID format (MM01S1...)
        if re.search(r'MM01S1', row_text, re.IGNORECASE):
            final_output.append([row_text])

    # 4. Save to CSV
    OUTPUT_CSV = 'final_combined_data.csv'
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(final_output)

    print(f"✅ Processed {len(final_output)} rows.")
    print(f"📁 File saved: {os.path.abspath(OUTPUT_CSV)}")

except Exception as e:
    print(f"❌ Error: {e}")