import os
import re
import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from openpyxl import Workbook

# =========================
# CONFIGURATION
# =========================
IMAGE_PATH = r"docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"
OUTPUT_FILE = "output.xlsx"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ocr = PaddleOCR(use_angle_cls=True, lang='en')


# =========================
# DESKEW FUNCTION
# =========================
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated


def extract_text(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found. Check path.")
        return []

    image = deskew(image)
    result = ocr.ocr(image)

    lines = []

    for line in result[0]:
        box = line[0]
        x_min = int(min([p[0] for p in box]))
        y_min = int(min([p[1] for p in box]))
        x_max = int(max([p[0] for p in box]))
        y_max = int(max([p[1] for p in box]))

        crop = image[y_min:y_max, x_min:x_max]
        text = pytesseract.image_to_string(crop, config='--psm 6')
        text = text.strip()

        if text:
            lines.append((y_min, text))

    lines = sorted(lines, key=lambda x: x[0])
    return [line[1] for line in lines]


def merge_rows(lines):
    merged_data = []
    current_row = []

    for line in lines:
        if re.match(r"MM\d+", line):
            if current_row:
                merged_data.append(" ".join(current_row))
            current_row = [line]
        else:
            current_row.append(line)

    if current_row:
        merged_data.append(" ".join(current_row))

    return merged_data


# =========================
# MAIN
# =========================
wb = Workbook()
ws = wb.active

lines = extract_text(IMAGE_PATH)
merged_rows = merge_rows(lines)

row_number = 1
for row in merged_rows:
    ws.cell(row=row_number, column=1).value = row
    row_number += 1

wb.save(OUTPUT_FILE)

print("✅ Extraction Completed Successfully!")
