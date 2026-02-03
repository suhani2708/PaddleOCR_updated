import pytesseract
from PIL import Image
import pandas as pd
import re
import os

# --- SET TESSERACT PATH IF NEEDED ---
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def validate_and_normalize_path(image_path):
    """Validate and normalize the image path"""
    if not image_path:
        raise ValueError("Image path cannot be empty")

    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(__file__), image_path)

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path

# --- IMAGE PATH ---
image_path = 'docs/images/Statista-FormF-RKJJ-NA-1929751_d.jpg'  # Changed to relative path

try:
    image_path = validate_and_normalize_path(image_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# --- LOAD AND PREPROCESS IMAGE ---
img = Image.open(image_path)
img = img.convert('L')  # Grayscale
img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize
img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)  # Upscale

# --- PERFORM OCR ---
text = pytesseract.image_to_string(img)

# --- PROCESS ALL LINES ---
lines = text.splitlines()
clean_lines = [line.strip() for line in lines if line.strip()]

if not clean_lines:
    raise ValueError(" No text detected in the image.")

print(f" Detected {len(clean_lines)} non-empty lines. Processing...")

# --- SPLIT EACH LINE INTO INDIVIDUAL WORDS ---
all_word_rows = []
max_words = 0

for i, line in enumerate(clean_lines):
    # Split by any whitespace (including single space)
    words = line.split()  # This handles multiple spaces, tabs, etc.
    all_word_rows.append(words)
    max_words = max(max_words, len(words))
    print(f"  Line {i+1}: {words}")

# --- PAD ROWS TO SAME LENGTH (so DataFrame aligns columns) ---
padded_rows = []
for words in all_word_rows:
    padded_row = words + [''] * (max_words - len(words))
    padded_rows.append(padded_row)

# --- CREATE DATAFRAME ---
df = pd.DataFrame(padded_rows)

# --- SAVE TO EXCEL ---
output_excel = 'all_words_in_columns.xlsx'
df.to_excel(output_excel, index=False, header=False)

print(f"\n Successfully saved all words into '{output_excel}' with each word in a separate column.") 