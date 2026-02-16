import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import re

# 📌 Your image path (keep as-is)
img_path = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\image(1).png"

if not os.path.isfile(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# ✅ Critical: Run this ONLY from a folder WITHOUT a local 'paddleocr/' dir
# If you're still getting paddlex error, move script out first!

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

print("🔍 Running OCR...")
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Failed to load image.")

result = ocr.ocr(img, cls=True)

# Extract all text blocks with positions
blocks = []
for line in result:
    if not line:
        continue
    for item in line:
        bbox = np.array(item[0])
        text = item[1][0].strip()
        if not text:
            continue
        xs, ys = bbox[:, 0], bbox[:, 1]
        cx, cy = xs.mean(), ys.mean()
        blocks.append({
            'text': text,
            'cx': cx,
            'cy': cy
        })

if not blocks:
    raise RuntimeError("No text detected.")

# === Step 1: Cluster into logical rows (by Y) ===
def cluster_rows(blocks, y_tol=25):
    sorted_blocks = sorted(blocks, key=lambda b: b['cy'])
    rows = []
    current = [sorted_blocks[0]]
    for b in sorted_blocks[1:]:
        if abs(b['cy'] - current[-1]['cy']) <= y_tol:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return rows

rows = cluster_rows(blocks, y_tol=30)

# === Step 2: Determine column structure from X-coordinates ===
all_cx = [b['cx'] for b in blocks]
if len(all_cx) < 2:
    col_centers = [np.mean(all_cx)]
else:
    # Use k-means-like binning: sort cx, split into ~max_cols bins
    cx_sorted = sorted(all_cx)
    max_possible_cols = min(10, len(cx_sorted))  # cap at 10
    step = len(cx_sorted) // max_possible_cols
    col_centers = []
    for i in range(max_possible_cols):
        start = i * step
        end = (i + 1) * step if i < max_possible_cols - 1 else len(cx_sorted)
        col_centers.append(np.mean(cx_sorted[start:end]))
    # Remove duplicates (close centers)
    col_centers = sorted(set([round(c, 1) for c in col_centers]))

def assign_col(cx):
    dists = [abs(cx - c) for c in col_centers]
    return int(np.argmin(dists))

# === Step 3: Build grid — each row → list of cells (merged vertically) ===
table_data = []
for row_blocks in rows:
    row_blocks = sorted(row_blocks, key=lambda b: b['cx'])
    cells = []
    # Group by column index
    col_dict = {}
    for block in row_blocks:
        col_idx = assign_col(block['cx'])
        if col_idx not in col_dict:
            col_dict[col_idx] = []
        col_dict[col_idx].append(block['text'])
    
    # Merge texts in same column (vertical stacking)
    max_col = max(col_dict.keys()) if col_dict else -1
    for idx in range(max_col + 1):
        if idx in col_dict:
            merged = "\n".join(col_dict[idx])
            cells.append(merged)
        else:
            cells.append("")
    table_data.append(cells)

# === Step 4: Pad all rows to same width (max columns in any row)
max_cols = max(len(row) for row in table_data)
table_data = [row + [""] * (max_cols - len(row)) for row in table_data]

# === Step 5: Convert to DataFrame — NO hardcoded headers
df = pd.DataFrame(table_data)

# Save to Excel (first row = header, as extracted)
output_file = "table_as_in_image.xlsx"
df.to_excel(output_file, index=False, header=False)

print(f" Saved to: {output_file}")
print(f"\nDetected {len(table_data)} rows, {max_cols} columns")
for i, row in enumerate(table_data[:4]):
    print(f"Row {i+1}: {row}")