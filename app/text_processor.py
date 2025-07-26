import os
from typing import List, Dict, Any
from pathlib import Path

class TextProcessor:
    def __init__(self, extracted_text_file: str = "extracted_text.txt"):
        self.extracted_text_file = Path(extracted_text_file)
        
    def load_extracted_text(self) -> str:
        """Load text content from the extracted_text.txt file."""
        text = ""
        try:
            if self.extracted_text_file.exists():
                with open(self.extracted_text_file, 'r', encoding='utf-8') as file:
                    text = file.read()
                print(f"Successfully loaded text from: {self.extracted_text_file}")
            else:
                print(f"File not found: {self.extracted_text_file}")
        except Exception as e:
            print(f"Error reading extracted text file {self.extracted_text_file}: {str(e)}")
        return text
    
    def get_text_metadata(self) -> Dict[str, Any]:
        """Get metadata about the extracted text."""
        text = self.load_extracted_text()
        
        if not text:
            return {}
        
        # Count lines, words, characters
        lines = text.split('\n')
        words = text.split()
        
        metadata = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_lines': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'file_size_bytes': self.extracted_text_file.stat().st_size if self.extracted_text_file.exists() else 0
        }
        
        return metadata
    
    def process_text_content(self) -> Dict[str, Any]:
        """Process the extracted text and return structured data."""
        text_content = self.load_extracted_text()
        
        if not text_content:
            return {
                'content': '',
                'metadata': {},
                'processing_status': 'failed - no content'
            }
        
        metadata = self.get_text_metadata()
        
        return {
            'content': text_content,
            'metadata': metadata,
            'processing_status': 'completed'
        }

if __name__ == "__main__":
    processor = TextProcessor(extracted_text_file="extracted_text.txt")
    
    # Test: Load and process text
    result = processor.process_text_content()
    
    print("=== Text Processing Results ===")
    print(f"Processing Status: {result['processing_status']}")
    print(f"Content Length: {len(result['content'])} characters")
    print(f"Metadata: {result['metadata']}")
    
    if result['content']:
        print(f"\nFirst 200 characters:\n{result['content'][:200]}...")
    
    # Save processed results
    with open("processed_text_output.txt", "w", encoding="utf-8") as out:
        out.write("=== Processed Text Content ===\n")
        out.write(f"Status: {result['processing_status']}\n")
        out.write(f"Metadata: {result['metadata']}\n\n")
        out.write("Content:\n")
        out.write(result['content'])
    
    print(f"\nProcessed results saved to: processed_text_output.txt")