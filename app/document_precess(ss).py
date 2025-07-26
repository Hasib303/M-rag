from pdf2image import convert_from_path
import os

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# PDF file path
input_pdf = "data/HSC26-Bangla1st-Paper.pdf"

# Convert PDF to images
print(f"Converting PDF to images...")
pages = convert_from_path(input_pdf, dpi=300)

# Save each page as PNG
for i, page in enumerate(pages):
    # Create filename with zero-padding for proper sorting
    filename = f"images/page_{i+1:03d}.png"
    
    # Save the page as PNG
    page.save(filename, 'PNG')
    print(f"Saved: {filename}")

print(f"\nConversion complete! {len(pages)} pages saved to 'images' directory.")