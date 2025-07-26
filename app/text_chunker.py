import re
from typing import List, Dict, Any, Optional
from text_processor import TextProcessor
import json
from pathlib import Path

class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap_size: int = 50, extracted_text_file: str = "extracted_text.txt"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.text_processor = TextProcessor(extracted_text_file)
        
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences with specified size and overlap."""
        # Split by Bengali sentence markers
        sentences = re.split(r'[।\n\.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraphs with specified size."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is too long, split it by sentences
                if len(paragraph) > self.chunk_size:
                    para_chunks = self.chunk_by_sentences(paragraph)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def chunk_by_fixed_size(self, text: str) -> List[str]:
        """Chunk text by fixed character size with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for sentence endings within the next 100 characters
                search_end = min(end + 100, text_length)
                sentence_end = max(
                    text.rfind('।', end, search_end),
                    text.rfind('.', end, search_end),
                    text.rfind('\n', end, search_end)
                )
                if sentence_end > end:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap_size
            
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.overlap_size:
            return text
        
        overlap_start = len(text) - self.overlap_size
        # Try to start overlap at word boundary
        space_pos = text.find(' ', overlap_start)
        if space_pos > overlap_start:
            overlap_start = space_pos + 1
            
        return text[overlap_start:]
    
    def create_chunks_with_metadata(self, chunking_method: str = "sentences") -> List[Dict[str, Any]]:
        """Create chunks with metadata from the text processor output."""
        # Get processed text from text processor
        processed_data = self.text_processor.process_text_content()
        
        if processed_data['processing_status'] != 'completed':
            print(f"Text processing failed: {processed_data['processing_status']}")
            return []
        
        text_content = processed_data['content']
        
        # Choose chunking method
        if chunking_method == "sentences":
            chunks = self.chunk_by_sentences(text_content)
        elif chunking_method == "paragraphs":
            chunks = self.chunk_by_paragraphs(text_content)
        elif chunking_method == "fixed_size":
            chunks = self.chunk_by_fixed_size(text_content)
        else:
            print(f"Unknown chunking method: {chunking_method}. Using sentences.")
            chunks = self.chunk_by_sentences(text_content)
        
        # Create chunks with metadata
        chunks_with_metadata = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': f"chunk_{i+1:04d}",
                'chunk_index': i,
                'chunk_size': len(chunk_text),
                'chunk_words': len(chunk_text.split()),
                'chunking_method': chunking_method,
                'source_file': str(self.text_processor.extracted_text_file),
                'text': chunk_text
            }
            chunks_with_metadata.append(chunk_metadata)
        
        return chunks_with_metadata
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str = "text_chunks.json") -> str:
        """Save chunks to JSON file."""
        output_path = Path(output_file)
        
        chunk_data = {
            'total_chunks': len(chunks),
            'chunking_config': {
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size
            },
            'chunks': chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(chunks)} chunks to: {output_path}")
        return str(output_path)
    
    def get_chunk_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
        chunk_words = [chunk['chunk_words'] for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_words_per_chunk': sum(chunk_words) / len(chunk_words),
            'total_characters': sum(chunk_sizes),
            'total_words': sum(chunk_words)
        }
        
        return stats

if __name__ == "__main__":
    chunker = TextChunker(chunk_size=300, overlap_size=30, extracted_text_file="extracted_text.txt")
    
    print("=== Creating Text Chunks ===")
    
    # Test different chunking methods
    methods = ["sentences", "paragraphs", "fixed_size"]
    
    for method in methods:
        print(f"\n--- Chunking by {method} ---")
        
        chunks = chunker.create_chunks_with_metadata(chunking_method=method)
        
        if chunks:
            # Save chunks
            output_file = f"chunks_{method}.json"
            chunker.save_chunks(chunks, output_file)
            
            # Get statistics
            stats = chunker.get_chunk_statistics(chunks)
            print(f"Statistics: {stats}")
            
            # Show first chunk as example
            print(f"First chunk preview: {chunks[0]['text'][:100]}...")
        else:
            print(f"No chunks created for method: {method}")
    
    print("\n=== Chunking Complete ===")
    print("Check the generated JSON files for detailed chunk data.")