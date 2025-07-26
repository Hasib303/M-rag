import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from text_chunker import TextChunker
import json
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class ChunkEmbedder:
    def __init__(self, extracted_text_file: str = "extracted_text.txt", chunk_size: int = 400, overlap_size: int = 50):
        self.chunker = TextChunker(chunk_size, overlap_size, extracted_text_file)
        self.chunks = []
        self.embeddings = None
        self.vectorizer = None
        self.is_initialized = False
        
    def preprocess_text_for_embedding(self, text: str) -> str:
        """Preprocess Bengali text for better embedding quality."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Bengali characters
        text = re.sub(r'[^\u0980-\u09FF\s\w]', ' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def create_tfidf_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Create TF-IDF embeddings for chunks."""
        print("Creating TF-IDF embeddings...")
        
        # Prepare texts for embedding
        chunk_texts = []
        for chunk in chunks:
            preprocessed_text = self.preprocess_text_for_embedding(chunk['text'])
            chunk_texts.append(preprocessed_text)
        
        # Create TF-IDF vectorizer with Bengali-friendly settings
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            lowercase=True,
            token_pattern=r'[\u0980-\u09FF\w]+',  # Bengali and English words
            stop_words=None  # No predefined stop words for Bengali
        )
        
        # Fit and transform the texts
        try:
            embeddings = vectorizer.fit_transform(chunk_texts)
            print(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings.toarray(), vectorizer
        except Exception as e:
            print(f"Error creating TF-IDF embeddings: {str(e)}")
            return None, None
    
    def create_simple_word_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Create simple word-based embeddings as fallback."""
        print("Creating simple word embeddings...")
        
        # Build vocabulary
        vocab = set()
        for chunk in chunks:
            words = self.preprocess_text_for_embedding(chunk['text']).split()
            vocab.update(words)
        
        vocab = list(vocab)
        vocab_size = len(vocab)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        print(f"Vocabulary size: {vocab_size}")
        
        # Create embeddings
        embeddings = []
        for chunk in chunks:
            words = self.preprocess_text_for_embedding(chunk['text']).split()
            
            # Create binary vector
            embedding = np.zeros(vocab_size)
            for word in words:
                if word in word_to_idx:
                    embedding[word_to_idx[word]] = 1
            
            # Normalize
            if np.sum(embedding) > 0:
                embedding = embedding / np.sum(embedding)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def initialize_embeddings(self, chunking_method: str = "sentences", embedding_method: str = "tfidf") -> bool:
        """Initialize chunks and create embeddings."""
        try:
            print("Initializing chunk embeddings...")
            
            # Create chunks
            self.chunks = self.chunker.create_chunks_with_metadata(chunking_method)
            
            if not self.chunks:
                print("No chunks created. Check if extracted_text.txt exists.")
                return False
            
            print(f"Created {len(self.chunks)} chunks")
            
            # Create embeddings
            if embedding_method == "tfidf":
                self.embeddings, self.vectorizer = self.create_tfidf_embeddings(self.chunks)
            elif embedding_method == "simple":
                self.embeddings = self.create_simple_word_embeddings(self.chunks)
                self.vectorizer = None
            else:
                print(f"Unknown embedding method: {embedding_method}")
                return False
            
            if self.embeddings is None:
                print("Failed to create embeddings")
                return False
            
            # Add embedding info to chunks
            for i, chunk in enumerate(self.chunks):
                chunk['embedding_index'] = i
                chunk['embedding_method'] = embedding_method
            
            self.is_initialized = True
            print(f"Embeddings initialized successfully with shape: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            return False
    
    def get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query text."""
        if not self.is_initialized:
            print("Embeddings not initialized")
            return None
        
        preprocessed_query = self.preprocess_text_for_embedding(query)
        
        if self.vectorizer is not None:
            # Use TF-IDF vectorizer
            try:
                query_embedding = self.vectorizer.transform([preprocessed_query])
                return query_embedding.toarray()[0]
            except Exception as e:
                print(f"Error creating query embedding: {str(e)}")
                return None
        else:
            # Use simple word embedding (not implemented for queries in this version)
            print("Simple embedding query not implemented")
            return None
    
    def find_similar_chunks(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Find chunks most similar to the query using embeddings."""
        if not self.is_initialized:
            print("Embeddings not initialized")
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top similar chunks
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_chunks = []
        for idx in similar_indices:
            similarity_score = similarities[idx]
            
            if similarity_score >= similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarity_score)
                chunk['embedding_similarity_rank'] = len(similar_chunks) + 1
                similar_chunks.append(chunk)
        
        return similar_chunks
    
    def save_embeddings(self, embeddings_file: str = "chunk_embeddings.pkl", chunks_file: str = "embedded_chunks.json") -> Dict[str, str]:
        """Save embeddings and chunks to files."""
        if not self.is_initialized:
            print("Embeddings not initialized")
            return {}
        
        # Save embeddings and vectorizer
        embeddings_path = Path(embeddings_file)
        embedding_data = {
            'embeddings': self.embeddings,
            'vectorizer': self.vectorizer,
            'embedding_shape': self.embeddings.shape
        }
        
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Save chunks with embedding metadata
        chunks_path = Path(chunks_file)
        chunks_data = {
            'chunks': self.chunks,
            'total_chunks': len(self.chunks),
            'embedding_info': {
                'shape': self.embeddings.shape,
                'method': self.chunks[0].get('embedding_method', 'unknown') if self.chunks else 'unknown'
            }
        }
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"Embeddings saved to: {embeddings_path}")
        print(f"Chunks saved to: {chunks_path}")
        
        return {
            'embeddings_file': str(embeddings_path),
            'chunks_file': str(chunks_path)
        }
    
    def load_embeddings(self, embeddings_file: str = "chunk_embeddings.pkl", chunks_file: str = "embedded_chunks.json") -> bool:
        """Load embeddings and chunks from files."""
        try:
            # Load embeddings
            embeddings_path = Path(embeddings_file)
            if not embeddings_path.exists():
                print(f"Embeddings file not found: {embeddings_path}")
                return False
            
            with open(embeddings_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            self.embeddings = embedding_data['embeddings']
            self.vectorizer = embedding_data['vectorizer']
            
            # Load chunks
            chunks_path = Path(chunks_file)
            if not chunks_path.exists():
                print(f"Chunks file not found: {chunks_path}")
                return False
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.chunks = chunks_data['chunks']
            
            self.is_initialized = True
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")
            print(f"Loaded {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        if not self.is_initialized:
            return {"error": "Embeddings not initialized"}
        
        return {
            'total_chunks': len(self.chunks),
            'embedding_shape': self.embeddings.shape,
            'embedding_method': self.chunks[0].get('embedding_method', 'unknown') if self.chunks else 'unknown',
            'vectorizer_features': self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
            'avg_embedding_norm': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'embedding_sparsity': float(np.mean(self.embeddings == 0))
        }

if __name__ == "__main__":
    embedder = ChunkEmbedder(
        extracted_text_file="extracted_text.txt",
        chunk_size=400,
        overlap_size=50
    )
    
    print("=== Initializing Chunk Embeddings ===")
    
    if embedder.initialize_embeddings(chunking_method="sentences", embedding_method="tfidf"):
        # Save embeddings
        saved_files = embedder.save_embeddings()
        
        # Get statistics
        stats = embedder.get_embedding_stats()
        print(f"\nEmbedding Statistics: {stats}")
        
        # Test similarity search
        test_queries = [
            "রবীন্দ্রনাথ",
            "অপরিচিতা গল্প",
            "প্রশ্ন উত্তর",
            "চরিত্র বিশ্লেষণ"
        ]
        
        print("\n=== Testing Similarity Search ===")
        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            similar_chunks = embedder.find_similar_chunks(query, top_k=3)
            
            for i, chunk in enumerate(similar_chunks):
                print(f"Rank {i+1}: Similarity={chunk['similarity_score']:.3f}")
                print(f"Chunk: {chunk['text'][:100]}...")
        
        print("\n=== Embedding System Ready ===")
    else:
        print("Failed to initialize embedding system")