#!/usr/bin/env python3
import csv
import os
import numpy as np
import cv2
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    print("="*80)
    print("MULTI-COLUMN STRUCTURE-AWARE EXTRACTION")
    print("="*80)
    img = preprocess_image(image_path)
    if img is None:
        return
    temp_path = "temp_processed.png"
    cv2.imwrite(temp_path, img)
    ocr = PaddleOCR(use_angle_cls=True, lang='en',
                    use_gpu=False, show_log=False)
    result = ocr.ocr(temp_path, cls=True)
    os.remove(temp_path)
    if not result or not result[0]:
        print("❌ No text detected")
        return
    # -------------------------------------------------------
    # Extract bounding box centers and heights
    # -------------------------------------------------------
    items = []
    for detection in result[0]:
        bbox = detection[0]
        text = detection[1][0].strip()
        if not text:
            continue
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        x_left = min(x_coords)
        height = max(y_coords) - min(y_coords)
        items.append({
            "text": text,
            "x_center": x_center,
            "x_left": x_left,
            "y_center": y_center,
            "height": height
        })
    if not items:
        print("❌ No valid text items")
        return
    # -------------------------------------------------------
    # Dynamic row threshold based on average height
    # -------------------------------------------------------
    avg_height = np.mean([item["height"] for item in items])
    row_threshold = avg_height * 1.5  # Adjustable multiplier
    print(f"Average text height: {avg_height:.2f}, Row threshold: {row_threshold:.2f}")
    # -------------------------------------------------------
    # STEP 1: Detect optimal number of columns using silhouette score
    # -------------------------------------------------------
    print("Detecting optimal number of columns...")
    x_positions = np.array([[item["x_center"]] for item in items])
    possible_k = range(5, min(30, len(items) // 2 + 1))  # Reasonable range for columns
    scores = []
    for k in possible_k:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(x_positions)
        score = silhouette_score(x_positions, kmeans.labels_)
        scores.append(score)
    num_columns = possible_k[np.argmax(scores)]
    print(f"Optimal number of columns: {num_columns}")
    # Fit with optimal k
    kmeans = KMeans(n_clusters=num_columns, random_state=0).fit(x_positions)
    for idx, item in enumerate(items):
        item["column"] = kmeans.labels_[idx]
    # Sort columns left to right by cluster centers
    cluster_centers = kmeans.cluster_centers_[:, 0]
    column_order = np.argsort(cluster_centers)
    # -------------------------------------------------------
    # STEP 2: Inside each column → group rows by Y
    # -------------------------------------------------------
    column_rows = {}
    for col in range(num_columns):
        col_items = [item for item in items if item["column"] == col]
        col_items = sorted(col_items, key=lambda x: x["y_center"])
        rows = []
        current_row = []
        for item in col_items:
            if not current_row:
                current_row.append(item)
                continue
            if abs(item["y_center"] - np.mean([i["y_center"] for i in current_row])) < row_threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        if current_row:
            rows.append(current_row)
        column_rows[col] = rows
    # -------------------------------------------------------
    # STEP 3: Build structured rows for CSV
    # -------------------------------------------------------
    max_rows = max(len(column_rows[col]) for col in column_rows) if column_rows else 0
    print(f"Total rows detected: {max_rows}")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row_index in range(max_rows):
            row_data = [""] * num_columns
            for i, col in enumerate(column_order):
                if row_index < len(column_rows[col]):
                    cell_items = column_rows[col][row_index]
                    # Sort items in cell by x_left (in case multi-detection in cell)
                    sorted_cell = sorted(cell_items, key=lambda x: x["x_left"])
                    cell_text = " ".join(item["text"] for item in sorted_cell)
                    row_data[i] = cell_text
            writer.writerow(row_data)
    print("✅ Done")
    print("Saved at:", os.path.abspath(output_csv))

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\91909\Desktop\github new repository\new repo\ftb(30-12-2025)\PaddleOCR_updated\PaddleOCR(03-02-2026)\docs\images\Statista-FormF-RKJJ-NA-1929751_d.jpg"
    OUTPUT_CSV = "final_perfect2_structured_output.csv"
    extract_structured_rows(IMAGE_PATH, OUTPUT_CSV)