
# -*- coding: utf-8 -*-
import pytesseract
from PIL import Image
import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows

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

print("Loading and preprocessing image...")

# --- LOAD AND PREPROCESS IMAGE ---
img = Image.open(image_path)
img = img.convert('L')  # Grayscale
img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize
img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)  # Upscale 2x

# --- GET FULL TEXT LINES (for grouping) ---
text = pytesseract.image_to_string(img)
lines = [line.strip() for line in text.splitlines() if line.strip()]

if not lines:
    raise ValueError("No text detected in the image.")

print(f" Total non-empty lines detected: {len(lines)}")

# --- GROUP LINES IN CHUNKS OF 4 ---
chunk_size = 4
groups = []
for i in range(0, len(lines), chunk_size):
    chunk = lines[i:i + chunk_size]
    words_in_group = []
    for line in chunk:
        words_in_group.extend(line.split())
    groups.append(words_in_group)

print(f"\nCreated {len(groups)} groups (each = up to 4 lines)")

# --- PAD TO MAX LENGTH ---
max_len = max(len(g) for g in groups) if groups else 0
padded_groups = [g + [''] * (max_len - len(g)) for g in groups]

# --- CREATE DATAFRAME (for structure only) ---
df = pd.DataFrame(padded_groups)

# --- NOW, GET WORD-LEVEL CONFIDENCE TO KNOW WHICH WORDS TO HIGHLIGHT ---
print("Running OCR with word-level data for confidence...")
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# Build a list of words with their confidence, in reading order
ocr_words_with_conf = []
n_boxes = len(data['level'])
for i in range(n_boxes):
    conf = int(data['conf'][i])
    text_word = data['text'][i].strip()
    if conf >= 0 and text_word:
        ocr_words_with_conf.append((text_word, conf))

# Flatten all words from groups (same order as in df)
all_words_in_order = []
for group in padded_groups:
    for word in group:
        if word != '':
            all_words_in_order.append(word)

# Map each word in `all_words_in_order` to its confidence
# Since Tesseract may split/merge differently, we match by position (approximate)
word_conf_map = {}
min_len = min(len(all_words_in_order), len(ocr_words_with_conf))
for i in range(min_len):
    word_from_group = all_words_in_order[i]
    word_from_ocr, conf = ocr_words_with_conf[i]
    # Even if text differs slightly, we assign confidence by position
    word_conf_map[word_from_group] = conf

# For safety, if lengths differ, remaining words get conf = 0
for i in range(min_len, len(all_words_in_order)):
    word_conf_map[all_words_in_order[i]] = 0

# --- CREATE EXCEL WITH HIGHLIGHTING ---
wb = Workbook()
ws = wb.active

red_font = Font(color="FF0000")  # Red color

# Write rows with conditional formatting
for row_idx, row in enumerate(padded_groups, start=1):
    for col_idx, word in enumerate(row, start=1):
        ws.cell(row=row_idx, column=col_idx, value=word)
        if word and word_conf_map.get(word, 100) < 99:
            ws.cell(row=row_idx, column=col_idx).font = red_font

# Save
output_excel = 'merged_4lines_per_row_highlighted.xlsx'
wb.save(output_excel)

print(f"\n Saved {len(groups)} rows to: {output_excel}")
print(f"   Shape: {len(groups)} rows × {max_len} columns")
print("    Words with confidence < 99% are highlighted in RED.")
