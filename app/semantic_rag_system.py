from typing import List, Dict, Any, Optional, Tuple
from chunk_embedder import ChunkEmbedder
from task_extractor import TaskExtractor
import json
from pathlib import Path
import numpy as np

class SemanticRAGSystem:
    def __init__(self, extracted_text_file: str = "extracted_text.txt", chunk_size: int = 400, overlap_size: int = 50):
        self.embedder = ChunkEmbedder(extracted_text_file, chunk_size, overlap_size)
        self.task_extractor = TaskExtractor()
        self.is_initialized = False
        
    def initialize(self, chunking_method: str = "sentences", embedding_method: str = "tfidf") -> bool:
        """Initialize the semantic RAG system."""
        try:
            print("Initializing Semantic RAG System...")
            
            # Initialize embeddings
            success = self.embedder.initialize_embeddings(chunking_method, embedding_method)
            
            if success:
                # Extract structured data from chunks
                self._extract_chunk_tasks()
                self.is_initialized = True
                print("Semantic RAG System initialized successfully!")
                return True
            else:
                print("Failed to initialize embeddings")
                return False
                
        except Exception as e:
            print(f"Error initializing Semantic RAG system: {str(e)}")
            return False
    
    def load_existing_embeddings(self, embeddings_file: str = "chunk_embeddings.pkl", chunks_file: str = "embedded_chunks.json") -> bool:
        """Load existing embeddings to avoid recomputation."""
        try:
            success = self.embedder.load_embeddings(embeddings_file, chunks_file)
            if success:
                self._extract_chunk_tasks()
                self.is_initialized = True
                print("Loaded existing embeddings successfully!")
                return True
            return False
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False
    
    def _extract_chunk_tasks(self):
        """Extract structured tasks from each chunk."""
        print("Extracting structured data from chunks...")
        
        for i, chunk in enumerate(self.embedder.chunks):
            if 'structured_data' not in chunk:
                chunk_text = chunk['text']
                structured_data = self.task_extractor.extract_structured_data(chunk_text)
                chunk['structured_data'] = structured_data
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(self.embedder.chunks)} chunks")
    
    def semantic_search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        if not self.is_initialized:
            print("Semantic RAG system not initialized")
            return []
        
        return self.embedder.find_similar_chunks(query, top_k, similarity_threshold)
    
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Combine semantic search with keyword-based search."""
        if not self.is_initialized:
            print("Semantic RAG system not initialized")
            return []
        
        # Get semantic search results
        semantic_results = self.semantic_search(query, top_k * 2)  # Get more for combining
        
        # Get keyword-based search results (simple implementation)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and re-rank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, semantic_weight, top_k
        )
        
        return combined_results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        keyword_scores = []
        
        for chunk in self.embedder.chunks:
            chunk_words = set(chunk['text'].lower().split())
            
            # Calculate keyword overlap score
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            
            if union > 0:
                keyword_score = intersection / len(query_words)  # Precision-like score
                
                if keyword_score > 0:
                    chunk_copy = chunk.copy()
                    chunk_copy['keyword_score'] = keyword_score
                    keyword_scores.append(chunk_copy)
        
        # Sort by keyword score
        keyword_scores.sort(key=lambda x: x['keyword_score'], reverse=True)
        return keyword_scores[:top_k]
    
    def _combine_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict], 
                              semantic_weight: float, top_k: int) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        combined_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            chunk_id = result['chunk_id']
            semantic_score = result.get('similarity_score', 0)
            combined_scores[chunk_id] = {
                'chunk': result,
                'semantic_score': semantic_score,
                'keyword_score': 0,
                'combined_score': semantic_score * semantic_weight
            }
        
        # Add keyword scores
        for result in keyword_results:
            chunk_id = result['chunk_id']
            keyword_score = result.get('keyword_score', 0)
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['keyword_score'] = keyword_score
                combined_scores[chunk_id]['combined_score'] += keyword_score * (1 - semantic_weight)
            else:
                combined_scores[chunk_id] = {
                    'chunk': result,
                    'semantic_score': 0,
                    'keyword_score': keyword_score,
                    'combined_score': keyword_score * (1 - semantic_weight)
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # Prepare final results
        final_results = []
        for item in sorted_results[:top_k]:
            chunk = item['chunk'].copy()
            chunk['semantic_score'] = item['semantic_score']
            chunk['keyword_score'] = item['keyword_score']
            chunk['combined_score'] = item['combined_score']
            final_results.append(chunk)
        
        return final_results
    
    def answer_query(self, query: str, search_method: str = "hybrid", max_chunks: int = 3) -> Dict[str, Any]:
        """Answer a query using the semantic RAG system."""
        if not self.is_initialized:
            return {"error": "Semantic RAG system not initialized"}
        
        # Search for relevant chunks
        if search_method == "semantic":
            relevant_chunks = self.semantic_search(query, max_chunks)
        elif search_method == "hybrid":
            relevant_chunks = self.hybrid_search(query, max_chunks)
        else:
            relevant_chunks = self.semantic_search(query, max_chunks)
        
        if not relevant_chunks:
            return {
                "query": query,
                "answer": "No relevant information found in the document.",
                "sources": [],
                "search_method": search_method,
                "chunk_count": 0
            }
        
        # Compile answer from relevant chunks
        answer_parts = []
        sources = []
        
        for chunk_data in relevant_chunks:
            answer_parts.append(chunk_data['text'])
            
            source_info = {
                'chunk_id': chunk_data['chunk_id'],
                'chunk_index': chunk_data['chunk_index']
            }
            
            # Add score information based on search method
            if 'combined_score' in chunk_data:
                source_info.update({
                    'combined_score': chunk_data['combined_score'],
                    'semantic_score': chunk_data.get('semantic_score', 0),
                    'keyword_score': chunk_data.get('keyword_score', 0)
                })
            elif 'similarity_score' in chunk_data:
                source_info['similarity_score'] = chunk_data['similarity_score']
            
            sources.append(source_info)
        
        return {
            "query": query,
            "answer": "\n\n--- Next Relevant Section ---\n\n".join(answer_parts),
            "sources": sources,
            "search_method": search_method,
            "chunk_count": len(relevant_chunks)
        }
    
    def get_chunks_by_category(self, category: str, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """Get chunks related to a specific category using semantic search."""
        category_queries = {
            'questions': 'প্রশ্ন উত্তর',
            'vocabulary': 'শব্দার্থ অর্থ',
            'characters': 'চরিত্র পাত্র',
            'story': 'গল্প কাহিনী',
            'analysis': 'বিশ্লেষণ আলোচনা'
        }
        
        if category not in category_queries:
            return []
        
        query = category_queries[category]
        return self.semantic_search(query, top_k=10, similarity_threshold=min_similarity)
    
    def save_system_data(self, output_file: str = "semantic_rag_data.json") -> str:
        """Save the complete semantic RAG system data."""
        if not self.is_initialized:
            print("System not initialized")
            return ""
        
        # Save embeddings first
        embedding_files = self.embedder.save_embeddings()
        
        # Save system configuration and stats
        system_data = {
            'system_config': {
                'chunk_size': self.embedder.chunker.chunk_size,
                'overlap_size': self.embedder.chunker.overlap_size,
                'total_chunks': len(self.embedder.chunks)
            },
            'embedding_stats': self.embedder.get_embedding_stats(),
            'saved_files': embedding_files
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(system_data, f, ensure_ascii=False, indent=2)
        
        print(f"Semantic RAG system data saved to: {output_path}")
        return str(output_path)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        embedding_stats = self.embedder.get_embedding_stats()
        
        # Count chunks with different content types
        content_type_counts = {
            'with_questions': 0,
            'with_vocabulary': 0,
            'with_characters': 0,
            'with_any_tasks': 0
        }
        
        for chunk in self.embedder.chunks:
            structured_data = chunk.get('structured_data', {})
            tasks = structured_data.get('tasks', {})
            
            if tasks.get('questions'):
                content_type_counts['with_questions'] += 1
            if tasks.get('vocabulary'):
                content_type_counts['with_vocabulary'] += 1
            if tasks.get('characters'):
                content_type_counts['with_characters'] += 1
            if any(task_list for task_list in tasks.values()):
                content_type_counts['with_any_tasks'] += 1
        
        return {
            'embedding_stats': embedding_stats,
            'content_analysis': content_type_counts,
            'system_status': 'initialized'
        }

if __name__ == "__main__":
    rag_system = SemanticRAGSystem(
        extracted_text_file="extracted_text.txt",
        chunk_size=400,
        overlap_size=50
    )
    
    print("=== Initializing Semantic RAG System ===")
    
    if rag_system.initialize(chunking_method="sentences", embedding_method="tfidf"):
        # Save system data
        rag_system.save_system_data()
        
        # Get system statistics
        stats = rag_system.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"Embedding Stats: {stats['embedding_stats']}")
        print(f"Content Analysis: {stats['content_analysis']}")
        
        # Test different search methods
        test_queries = [
            "রবীন্দ্রনাথ ঠাকুর",
            "অপরিচিতা গল্পের চরিত্র",
            "প্রশ্ন ও উত্তর",
            "শব্দার্থ"
        ]
        
        search_methods = ["semantic", "hybrid"]
        
        print("\n=== Testing Search Methods ===")
        for method in search_methods:
            print(f"\n--- {method.upper()} SEARCH ---")
            for query in test_queries:
                print(f"\nQuery: {query}")
                result = rag_system.answer_query(query, search_method=method, max_chunks=2)
                print(f"Found {result['chunk_count']} chunks")
                if result['chunk_count'] > 0:
                    print(f"Answer preview: {result['answer'][:100]}...")
        
        print("\n=== Semantic RAG System Ready ===")
    else:
        print("Failed to initialize Semantic RAG system")