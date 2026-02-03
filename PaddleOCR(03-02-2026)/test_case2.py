
import sys
import os
import csv
import logging

# --- Clean sys.path (avoid local import conflicts) ---
for p in ["", ".", os.getcwd()]:
    if p in sys.path:
        sys.path.remove(p)

# --- Environment fixes ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# --- Reduce PaddleOCR logs ---
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR

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

# --- Image path ---
img_path = "docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg"  # Changed to relative path

try:
    img_path = validate_and_normalize_path(img_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# --- OCR init ---
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=False,
    show_log=False,
    return_word_box=True
)

# --- Run OCR ---
result = ocr.ocr(img_path, cls=True)  # Correct API usage: using ocr() method

# --- Settings ---
csv_file = "output.csv"
CONFIDENCE_THRESHOLD = 1.00  # Highlight below 90%

detections = []

# --- Normalize PaddleOCR output ---
if result:
    # Handle PaddleOCR output format: result[0] contains all detected text regions
    # Each region is: [bounding_box, (text, confidence)]
    # bounding_box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    # text is a string, confidence is a float
    items = result[0] if len(result) == 1 and isinstance(result[0], list) else result
    for item in items:
        if not item or len(item) < 2:
            continue

        text_info = item[1]

        if (
            isinstance(text_info, (list, tuple))
            and len(text_info) >= 2
        ):
            try:
                text = str(text_info[0]).strip()
                confidence = float(text_info[1])

                if text:
                    detections.append((text, confidence))

            except (ValueError, TypeError):
                continue

# --- Write CSV with Low_Confidence flag ---
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["Text", "Confidence", "Low_Confidence"])

    if not detections:
        print("❌ No text detected")
        writer.writerow(["No text detected", "", ""])
    else:
        for text, confidence in detections:
            # ✅ Add flag for Excel conditional formatting
            low_flag = "YES" if confidence < CONFIDENCE_THRESHOLD else "NO"
            writer.writerow([text, f"{confidence:.2f}", low_flag])
            print(f"Text: '{text}' | Confidence: {confidence:.2f} | Low: {low_flag}")

print(f"\n✅ CSV saved successfully: {csv_file}")