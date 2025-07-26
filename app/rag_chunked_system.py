from typing import List, Dict, Any, Optional
from text_chunker import TextChunker
from task_extractor import TaskExtractor
import json
from pathlib import Path
import re
from collections import defaultdict

class RAGChunkedSystem:
    def __init__(self, extracted_text_file: str = "extracted_text.txt", chunk_size: int = 400, overlap_size: int = 50):
        self.chunker = TextChunker(chunk_size, overlap_size, extracted_text_file)
        self.task_extractor = TaskExtractor()
        self.chunks = []
        self.chunk_index = {}
        self.is_initialized = False
        
    def initialize(self, chunking_method: str = "sentences") -> bool:
        """Initialize the RAG system with chunked data."""
        try:
            print("Initializing RAG system with chunked data...")
            
            # Create chunks
            self.chunks = self.chunker.create_chunks_with_metadata(chunking_method)
            
            if not self.chunks:
                print("No chunks created. Check if extracted_text.txt exists.")
                return False
            
            # Create search index
            self._create_chunk_index()
            
            # Extract structured data from chunks
            self._extract_chunk_tasks()
            
            self.is_initialized = True
            print(f"RAG system initialized with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            return False
    
    def _create_chunk_index(self):
        """Create an index for fast chunk searching."""
        self.chunk_index = {}
        
        for chunk in self.chunks:
            chunk_id = chunk['chunk_id']
            words = chunk['text'].lower().split()
            
            # Index each word to chunk IDs
            for word in words:
                # Clean word of punctuation
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    if clean_word not in self.chunk_index:
                        self.chunk_index[clean_word] = []
                    self.chunk_index[clean_word].append(chunk_id)
    
    def _extract_chunk_tasks(self):
        """Extract structured tasks from each chunk."""
        for chunk in self.chunks:
            chunk_text = chunk['text']
            structured_data = self.task_extractor.extract_structured_data(chunk_text)
            chunk['structured_data'] = structured_data
    
    def search_chunks(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search through chunks based on query."""
        if not self.is_initialized:
            print("RAG system not initialized. Please call initialize() first.")
            return []
        
        query_words = [re.sub(r'[^\w]', '', word.lower()) for word in query.split()]
        query_words = [word for word in query_words if word]
        
        if not query_words:
            return []
        
        # Find chunks that contain query words
        chunk_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.chunk_index:
                for chunk_id in self.chunk_index[word]:
                    chunk_scores[chunk_id] += 1
        
        # Calculate relevance scores
        scored_chunks = []
        for chunk_id, word_matches in chunk_scores.items():
            chunk = next(c for c in self.chunks if c['chunk_id'] == chunk_id)
            
            # Calculate relevance score
            relevance_score = word_matches / len(query_words)
            
            # Boost score if query appears as phrase
            if query.lower() in chunk['text'].lower():
                relevance_score += 0.5
            
            scored_chunks.append({
                'chunk_id': chunk_id,
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'structured_data': chunk.get('structured_data', {}),
                'relevance_score': relevance_score,
                'word_matches': word_matches,
                'chunk_size': chunk['chunk_size']
            })
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_chunks[:max_results]
    
    def answer_query(self, query: str, max_chunks: int = 3) -> Dict[str, Any]:
        """Answer a query using the chunked RAG system."""
        if not self.is_initialized:
            return {"error": "RAG system not initialized"}
        
        # Search for relevant chunks
        relevant_chunks = self.search_chunks(query, max_chunks)
        
        if not relevant_chunks:
            return {
                "query": query,
                "answer": "No relevant information found in the document.",
                "sources": [],
                "chunk_count": 0
            }
        
        # Compile answer from relevant chunks
        answer_parts = []
        sources = []
        
        for chunk_data in relevant_chunks:
            answer_parts.append(chunk_data['text'])
            sources.append({
                'chunk_id': chunk_data['chunk_id'],
                'chunk_index': chunk_data['chunk_index'],
                'relevance_score': chunk_data['relevance_score'],
                'word_matches': chunk_data['word_matches']
            })
        
        return {
            "query": query,
            "answer": "\n\n---\n\n".join(answer_parts),
            "sources": sources,
            "chunk_count": len(relevant_chunks)
        }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
    
    def get_chunks_with_tasks(self, task_type: str = None) -> List[Dict[str, Any]]:
        """Get chunks that contain specific types of tasks."""
        chunks_with_tasks = []
        
        for chunk in self.chunks:
            structured_data = chunk.get('structured_data', {})
            tasks = structured_data.get('tasks', {})
            
            if task_type:
                if task_type in tasks and tasks[task_type]:
                    chunks_with_tasks.append(chunk)
            else:
                # Return chunks with any tasks
                if any(task_list for task_list in tasks.values()):
                    chunks_with_tasks.append(chunk)
        
        return chunks_with_tasks
    
    def save_system_data(self, output_file: str = "rag_system_data.json") -> str:
        """Save the complete RAG system data."""
        system_data = {
            'chunks': self.chunks,
            'system_config': {
                'chunk_size': self.chunker.chunk_size,
                'overlap_size': self.chunker.overlap_size,
                'total_chunks': len(self.chunks)
            },
            'statistics': self.chunker.get_chunk_statistics(self.chunks)
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(system_data, f, ensure_ascii=False, indent=2)
        
        print(f"RAG system data saved to: {output_path}")
        return str(output_path)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self.is_initialized:
            return {"error": "RAG system not initialized"}
        
        chunk_stats = self.chunker.get_chunk_statistics(self.chunks)
        
        # Count chunks with different task types
        task_counts = defaultdict(int)
        for chunk in self.chunks:
            structured_data = chunk.get('structured_data', {})
            tasks = structured_data.get('tasks', {})
            for task_type, task_list in tasks.items():
                if task_list:
                    task_counts[task_type] += 1
        
        return {
            'total_chunks': len(self.chunks),
            'chunk_statistics': chunk_stats,
            'chunks_with_tasks': dict(task_counts),
            'index_size': len(self.chunk_index),
            'system_status': 'initialized' if self.is_initialized else 'not_initialized'
        }

if __name__ == "__main__":
    rag_system = RAGChunkedSystem(
        extracted_text_file="extracted_text.txt",
        chunk_size=400,
        overlap_size=50
    )
    
    print("=== Initializing RAG Chunked System ===")
    
    if rag_system.initialize(chunking_method="sentences"):
        # Save system data
        rag_system.save_system_data()
        
        # Get system statistics
        stats = rag_system.get_system_stats()
        print(f"\nSystem Statistics: {stats}")
        
        # Test queries
        test_queries = [
            "রবীন্দ্রনাথ",
            "অপরিচিতা",
            "প্রশ্ন",
            "চরিত্র"
        ]
        
        print("\n=== Testing Queries ===")
        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            result = rag_system.answer_query(query, max_chunks=2)
            print(f"Found {result['chunk_count']} relevant chunks")
            if result['chunk_count'] > 0:
                print(f"Answer preview: {result['answer'][:150]}...")
        
        print("\n=== RAG System Ready ===")
    else:
        print("Failed to initialize RAG system")