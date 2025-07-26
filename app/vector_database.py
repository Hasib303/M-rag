import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import sqlite3
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # Facebook AI Similarity Search - vector database
from chunk_embedder import ChunkEmbedder

class VectorDatabase:
    def __init__(self, db_path: str = "vector_database", dimension: int = 5000):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.dimension = dimension
        self.index = None
        self.metadata_db = None
        self.chunks_data = []
        self.is_initialized = False
        
        # Initialize metadata database (SQLite)
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for chunk metadata."""
        db_file = self.db_path / "metadata.db"
        self.metadata_db = sqlite3.connect(str(db_file))
        
        # Create table for chunk metadata
        self.metadata_db.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                chunk_index INTEGER,
                text TEXT,
                chunk_size INTEGER,
                chunk_words INTEGER,
                embedding_method TEXT,
                source_file TEXT,
                structured_data TEXT
            )
        ''')
        self.metadata_db.commit()
    
    def create_index(self, index_type: str = "flat") -> bool:
        """Create FAISS index for vector similarity search."""
        try:
            if index_type == "flat":
                # Exact search - slower but most accurate
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product
            elif index_type == "ivf":
                # Approximate search - faster for large datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            elif index_type == "hnsw":
                # Hierarchical Navigable Small World - good balance
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                print(f"Unknown index type: {index_type}")
                return False
            
            print(f"Created FAISS {index_type} index with dimension {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            return False
    
    def add_embeddings(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]) -> bool:
        """Add embeddings and metadata to the vector database."""
        try:
            if self.index is None:
                print("Index not created. Call create_index() first.")
                return False
            
            # Normalize embeddings for cosine similarity (if using Inner Product)
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to FAISS index
            if hasattr(self.index, 'train') and not self.index.is_trained:
                # Train index if needed (for IVF type)
                self.index.train(normalized_embeddings)
            
            self.index.add(normalized_embeddings.astype('float32'))
            
            # Store metadata in SQLite
            for chunk_meta in chunks_metadata:
                self.metadata_db.execute('''
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, chunk_index, text, chunk_size, chunk_words, 
                     embedding_method, source_file, structured_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_meta['chunk_id'],
                    chunk_meta['chunk_index'],
                    chunk_meta['text'],
                    chunk_meta['chunk_size'],
                    chunk_meta['chunk_words'],
                    chunk_meta.get('embedding_method', 'unknown'),
                    chunk_meta.get('source_file', ''),
                    json.dumps(chunk_meta.get('structured_data', {}), ensure_ascii=False)
                ))
            
            self.metadata_db.commit()
            self.chunks_data = chunks_metadata
            self.is_initialized = True
            
            print(f"Added {len(embeddings)} embeddings to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding embeddings to database: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar vectors in the database."""
        if not self.is_initialized:
            print("Vector database not initialized")
            return []
        
        try:
            # Normalize query embedding
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            query_normalized = query_normalized.reshape(1, -1).astype('float32')
            
            # Search in FAISS index
            scores, indices = self.index.search(query_normalized, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= score_threshold and idx < len(self.chunks_data):
                    # Get metadata from SQLite
                    chunk_meta = self._get_chunk_metadata_by_index(idx)
                    
                    if chunk_meta:
                        result = {
                            'chunk_id': chunk_meta['chunk_id'],
                            'chunk_index': chunk_meta['chunk_index'],
                            'text': chunk_meta['text'],
                            'similarity_score': float(score),
                            'rank': i + 1,
                            'structured_data': json.loads(chunk_meta['structured_data']) if chunk_meta['structured_data'] else {}
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching vector database: {str(e)}")
            return []
    
    def _get_chunk_metadata_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by its index in the database."""
        try:
            cursor = self.metadata_db.execute(
                'SELECT * FROM chunks WHERE chunk_index = ?', (index,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'chunk_id': row[1],
                    'chunk_index': row[2],
                    'text': row[3],
                    'chunk_size': row[4],
                    'chunk_words': row[5],
                    'embedding_method': row[6],
                    'source_file': row[7],
                    'structured_data': row[8]
                }
            return None
            
        except Exception as e:
            print(f"Error getting chunk metadata: {str(e)}")
            return None
    
    def save_index(self, index_file: str = "faiss_index.bin") -> bool:
        """Save FAISS index to disk."""
        try:
            index_path = self.db_path / index_file
            faiss.write_index(self.index, str(index_path))
            print(f"FAISS index saved to: {index_path}")
            return True
        except Exception as e:
            print(f"Error saving FAISS index: {str(e)}")
            return False
    
    def load_index(self, index_file: str = "faiss_index.bin") -> bool:
        """Load FAISS index from disk."""
        try:
            index_path = self.db_path / index_file
            if not index_path.exists():
                print(f"Index file not found: {index_path}")
                return False
            
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks metadata
            cursor = self.metadata_db.execute('SELECT * FROM chunks ORDER BY chunk_index')
            rows = cursor.fetchall()
            
            self.chunks_data = []
            for row in rows:
                chunk_data = {
                    'chunk_id': row[1],
                    'chunk_index': row[2],
                    'text': row[3],
                    'chunk_size': row[4],
                    'chunk_words': row[5],
                    'embedding_method': row[6],
                    'source_file': row[7],
                    'structured_data': json.loads(row[8]) if row[8] else {}
                }
                self.chunks_data.append(chunk_data)
            
            self.is_initialized = True
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        if not self.is_initialized:
            return {"error": "Database not initialized"}
        
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__ if self.index else 'None',
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'total_chunks': len(self.chunks_data),
            'database_path': str(self.db_path)
        }
    
    def close(self):
        """Close database connections."""
        if self.metadata_db:
            self.metadata_db.close()

class VectorRAGSystem:
    def __init__(self, extracted_text_file: str = "extracted_text.txt", 
                 chunk_size: int = 400, overlap_size: int = 50,
                 vector_db_path: str = "vector_database"):
        
        self.embedder = ChunkEmbedder(extracted_text_file, chunk_size, overlap_size)
        self.vector_db = VectorDatabase(vector_db_path)
        self.is_initialized = False
    
    def initialize(self, chunking_method: str = "sentences", 
                  embedding_method: str = "tfidf", 
                  index_type: str = "flat") -> bool:
        """Initialize the vector-based RAG system."""
        try:
            print("Initializing Vector RAG System...")
            
            # Create embeddings
            if not self.embedder.initialize_embeddings(chunking_method, embedding_method):
                return False
            
            # Create vector database index
            if not self.vector_db.create_index(index_type):
                return False
            
            # Add embeddings to vector database
            if not self.vector_db.add_embeddings(self.embedder.embeddings, self.embedder.chunks):
                return False
            
            # Save index for persistence
            self.vector_db.save_index()
            
            self.is_initialized = True
            print("Vector RAG System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing Vector RAG system: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector database."""
        if not self.is_initialized:
            print("System not initialized")
            return []
        
        # Get query embedding
        query_embedding = self.embedder.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Search in vector database
        return self.vector_db.search(query_embedding, top_k)
    
    def answer_query(self, query: str, max_chunks: int = 3) -> Dict[str, Any]:
        """Answer query using vector database search."""
        search_results = self.search(query, max_chunks)
        
        if not search_results:
            return {
                "query": query,
                "answer": "No relevant information found.",
                "sources": []
            }
        
        answer_parts = []
        sources = []
        
        for result in search_results:
            answer_parts.append(result['text'])
            sources.append({
                'chunk_id': result['chunk_id'],
                'similarity_score': result['similarity_score'],
                'rank': result['rank']
            })
        
        return {
            "query": query,
            "answer": "\n\n---\n\n".join(answer_parts),
            "sources": sources
        }

if __name__ == "__main__":
    # Test vector database system
    vector_rag = VectorRAGSystem(
        extracted_text_file="extracted_text.txt",
        chunk_size=400,
        overlap_size=50,
        vector_db_path="vector_database"
    )
    
    print("=== Initializing Vector Database RAG System ===")
    
    if vector_rag.initialize(index_type="flat"):
        # Get statistics
        stats = vector_rag.vector_db.get_stats()
        print(f"Vector Database Stats: {stats}")
        
        # Test queries
        test_queries = ["রবীন্দ্রনাথ", "অপরিচিতা", "প্রশ্ন"]
        
        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            result = vector_rag.answer_query(query, max_chunks=2)
            print(f"Answer: {result['answer'][:100]}...")
        
        print("\n=== Vector Database RAG System Ready ===")
    else:
        print("Failed to initialize system")