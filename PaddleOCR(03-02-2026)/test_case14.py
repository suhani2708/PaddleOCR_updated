#!/usr/bin/env python3
import csv
import os
import numpy as np
import cv2
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans

# ==========================================================
# LOAD + UPSCALE + DESKEW
# ==========================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found")
        return None
    # Upscale for small text stability
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ==========================================================
# MAIN EXTRACTION
# ==========================================================
def extract_structured_rows(image_path, output_csv):
    print("=" * 80)
    print("MULTI-COLUMN STRUCTURE-AWARE EXTRACTION (left → right)")
    print("=" * 80)

    img = preprocess_image(image_path)
    if img is None:
        return

    temp_path = "temp_processed.png"
    cv2.imwrite(temp_path, img)

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    result = ocr.ocr(temp_path, cls=True)
    os.remove(temp_path)

    if not result or not result[0]:
        print("❌ No text detected")
        return

    # ─── Collect all detected words + positions ──────────────────────────────
    items = []
    for detection in result[0]:
        bbox = detection[0]
        text = detection[1][0].strip()
        if not text:
            continue
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        items.append({
            "text": text,
            "x_center": np.mean(x_coords),
            "x_left": min(x_coords),
            "y_center": np.mean(y_coords)
        })

    # ─── Column detection with KMeans ────────────────────────────────────────
    print("Detecting columns...")
    x_positions = np.array([[item["x_center"]] for item in items])
    num_columns = 9   # ← change this value according to your table (7–12 usually)
    kmeans = KMeans(n_clusters=num_columns, random_state=0, n_init=10).fit(x_positions)

    for idx, item in enumerate(items):
        item["column"] = kmeans.labels_[idx]

    # ─── Group by column, sort by y and group close rows ─────────────────────
    column_rows = {}
    for col in range(num_columns):
        col_items = [item for item in items if item["column"] == col]
        col_items = sorted(col_items, key=lambda x: x["y_center"])

        rows = []
        current_row = []
        row_threshold = 14   # <--- LOWERED for stricter row separation

        for item in col_items:
            if not current_row:
                current_row.append(item)
                continue
            if abs(item["y_center"] - current_row[0]["y_center"]) < row_threshold:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x["x_left"])
                rows.append(current_row)
                current_row = [item]

        if current_row:
            current_row.sort(key=lambda x: x["x_left"])
            rows.append(current_row)

        column_rows[col] = rows

    # ─── Sort columns LEFT → RIGHT by average x_center ───────────────────────
    column_centers = {}
    for col in column_rows:
        if column_rows[col]:
            avg_x = np.mean([item["x_center"] for row in column_rows[col] for item in row])
            column_centers[col] = avg_x

    sorted_columns = sorted(column_centers.keys(), key=lambda c: column_centers[c])
    print("Detected column order (left to right):", sorted_columns)
    print(f"Number of detected columns: {len(sorted_columns)}")

    # ─── Build table: one visual row → one CSV row ───────────────────────────
    print("Building table (left to right)...")
    max_rows = max((len(rows) for rows in column_rows.values()), default=0)
    table = []

    for row_idx in range(max_rows):
        csv_row = [""] * len(sorted_columns)   # prepare empty row

        for col_idx, col in enumerate(sorted_columns):
            if row_idx < len(column_rows.get(col, [])):
                cells = column_rows[col][row_idx]
                # ─── Smarter joining of multiple detections in same cell ─────
                if len(cells) == 0:
                    csv_row[col_idx] = ""
                elif len(cells) == 1:
                    csv_row[col_idx] = cells[0]["text"].strip()
                elif len(cells) <= 3:
                    # likely continuation of the same value (address, name, code…)
                    csv_row[col_idx] = "".join(item["text"] for item in cells).strip()
                else:
                    # probably separate words / multiple items
                    csv_row[col_idx] = " ".join(item["text"] for item in cells).strip()

        table.append(csv_row)

    # ─── Write to CSV ────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Optional: add header with column numbers
        # writer.writerow([f"Col {i+1}" for i in range(len(sorted_columns))])
        writer.writerows(table)

    print(f"✅ Wrote {len(table)} rows × {len(sorted_columns)} columns")
    print("Saved at:", os.path.abspath(output_csv))


if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\image1.jpg"
    OUTPUT_CSV = "final_perfect_structured_output.csv"

    extract_structured_rows(IMAGE_PATH, OUTPUT_CSV)