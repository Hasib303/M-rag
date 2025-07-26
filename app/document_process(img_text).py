import easyocr
import glob
import os

# Directory containing images
image_dir = "images"

# Initialize EasyOCR reader for Bangla
reader = easyocr.Reader(['bn'])  # 'bn' for Bangla

# Get all PNG images in the directory, sorted
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))

all_text = ""

for i, image_path in enumerate(image_paths):
    print(f"Processing {image_path} ({i+1}/{len(image_paths)})")
    results = reader.readtext(image_path, detail=0)
    text_bengali = '\n'.join(results)
    all_text += f"\n\n{'='*50}\nPage {i+1}: {os.path.basename(image_path)}\n{'='*50}\n"
    all_text += text_bengali

print(all_text)

# Save to file
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)