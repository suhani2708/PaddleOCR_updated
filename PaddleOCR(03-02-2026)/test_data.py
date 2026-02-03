# test_data.py
import sys
import os

# 🔥 CRITICAL: Remove current directory from Python path to avoid loading local paddleocr source
if "" in sys.path:
    sys.path.remove("")
if "." in sys.path:
    sys.path.remove(".")
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

# Now import from installed package (not local source)
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
import csv
from pathlib import Path

def validate_and_normalize_path(image_path):
    """Validate and normalize the image path"""
    if not image_path:
        raise ValueError("Image path cannot be empty")

    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(__file__), image_path)

    # Check if file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path

img_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"  # Changed to relative path

try:
    img_path = validate_and_normalize_path(img_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# ✅ Only Text + Confidence (2 decimals)
ocr = PaddleOCR(lang='en', use_angle_cls=False, show_log=False, return_word_box=True)
result = ocr.ocr(img_path, cls=True)  # Correct API usage: using ocr() method

csv_file = f"{Path(img_path).stem}_output.csv"

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Text", "Confidence"])
    if result:
        # Handle PaddleOCR output format: result[0] contains all detected text regions
        # Each region is: [bounding_box, (text, confidence)]
        # bounding_box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # text is a string, confidence is a float
        items = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        for line in items:
            if len(line) < 2:
                continue
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = text_info[0]
                confidence = text_info[1]
                if text.strip():
                    writer.writerow([text, f"{confidence:.2f}"])
                    print(f"Text: '{text}' | Confidence: {confidence:.2f}")
    else:
        print("No text detected.")
        writer.writerow(["No text detected", ""])

print(f"\n✅ Output saved to: {csv_file}")