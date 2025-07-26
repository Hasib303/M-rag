#!/usr/bin/env python3
"""
Interactive Query Interface for Multilingual RAG System
Allows users to input queries and get semantic search results from the vector database.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_database import VectorRAGSystem
from chunk_embedder import ChunkEmbedder

class QueryInterface:
    def __init__(self, extracted_text_file: str = "extracted_text.txt", 
                 vector_db_path: str = "vector_database",
                 chunk_size: int = 400, overlap_size: int = 50):
        
        self.vector_rag = VectorRAGSystem(
            extracted_text_file=extracted_text_file,
            chunk_size=chunk_size, 
            overlap_size=overlap_size,
            vector_db_path=vector_db_path
        )
        self.is_ready = False
        self.query_history = []
    
    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """Initialize or load the vector RAG system."""
        print("🔧 Initializing Multilingual RAG System...")
        
        # Check if vector database already exists
        db_path = Path(self.vector_rag.vector_db.db_path)
        index_file = db_path / "faiss_index.bin"
        metadata_file = db_path / "metadata.db"
        
        if not force_rebuild and index_file.exists() and metadata_file.exists():
            print("📁 Found existing vector database, loading...")
            try:
                # Load existing vector database
                if self.vector_rag.vector_db.load_index():
                    # Load embedder components for query processing
                    embedder_files = ["chunk_embeddings.pkl", "embedded_chunks.json"]
                    if all(Path(f).exists() for f in embedder_files):
                        self.vector_rag.embedder.load_embeddings()
                        self.is_ready = True
                        print("✅ Successfully loaded existing system!")
                        return True
                    else:
                        print("⚠️  Embedder files missing, rebuilding...")
                        force_rebuild = True
                else:
                    print("⚠️  Failed to load index, rebuilding...")
                    force_rebuild = True
            except Exception as e:
                print(f"⚠️  Error loading existing system: {e}")
                force_rebuild = True
        
        if force_rebuild or not self.is_ready:
            print("🏗️  Building new vector database...")
            if self.vector_rag.initialize():
                self.is_ready = True
                print("✅ System initialized successfully!")
                return True
            else:
                print("❌ Failed to initialize system")
                return False
        
        return self.is_ready
    
    def process_query(self, query: str, max_results: int = 3, 
                     min_similarity: float = 0.1) -> Dict[str, Any]:
        """Process a user query and return results."""
        if not self.is_ready:
            return {"error": "System not initialized"}
        
        print(f"🔍 Processing query: '{query}'")
        
        # Get embedding-based search results
        search_results = self.vector_rag.search(query, top_k=max_results)
        
        # Filter by minimum similarity
        filtered_results = [
            result for result in search_results 
            if result.get('similarity_score', 0) >= min_similarity
        ]
        
        # Format response
        response = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_found": len(filtered_results),
            "search_results": filtered_results,
            "formatted_answer": self._format_answer(filtered_results),
            "sources": [
                {
                    "chunk_id": result["chunk_id"],
                    "similarity": f"{result['similarity_score']:.3f}",
                    "rank": result["rank"]
                } for result in filtered_results
            ]
        }
        
        # Add to query history
        self.query_history.append({
            "query": query,
            "timestamp": response["timestamp"],
            "results_count": len(filtered_results)
        })
        
        return response
    
    def _format_answer(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable answer."""
        if not results:
            return "দুঃখিত, আপনার প্রশ্নের জন্য কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"
        
        answer_parts = []
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity_score', 0)
            text = result.get('text', '')
            
            # Truncate very long texts
            if len(text) > 500:
                text = text[:500] + "..."
            
            answer_parts.append(
                f"📄 অংশ {i} (সাদৃশ্য: {similarity:.3f}):\n{text}"
            )
        
        return "\n\n" + "="*50 + "\n\n".join(answer_parts)
    
    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n" + "="*60)
        print("🌟 স্বাগতম! Multilingual RAG Query Interface")
        print("="*60)
        
        if not self.is_ready:
            print("❌ System not ready. Please initialize first.")
            return
        
        # Show system stats
        stats = self.vector_rag.vector_db.get_stats()
        print(f"📊 Vector Database: {stats['total_vectors']} vectors loaded")
        print(f"🔧 Index Type: {stats['index_type']}")
        print("\n💡 Tips:")
        print("  - আপনি বাংলা বা ইংরেজিতে প্রশ্ন করতে পারেন")
        print("  - 'quit', 'exit', বা 'q' টাইপ করে বের হন")
        print("  - 'history' টাইপ করে আগের প্রশ্নগুলো দেখুন")
        print("  - 'stats' টাইপ করে সিস্টেম তথ্য দেখুন")
        print("\n" + "-"*60)
        
        while True:
            try:
                # Get user input
                query = input("\n🤔 আপনার প্রশ্ন লিখুন: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q', 'বের', 'শেষ']:
                    print("👋 ধন্যবাদ! আবার আসবেন।")
                    break
                
                elif query.lower() in ['history', 'ইতিহাস']:
                    self._show_history()
                    continue
                
                elif query.lower() in ['stats', 'তথ্য']:
                    self._show_stats()
                    continue
                
                elif query.lower() in ['help', 'সাহায্য']:
                    self._show_help()
                    continue
                
                # Process the query
                print("\n⏳ অনুসন্ধান করা হচ্ছে...")
                result = self.process_query(query, max_results=3)
                
                # Display results
                if result.get("error"):
                    print(f"❌ ত্রুটি: {result['error']}")
                elif result["results_found"] == 0:
                    print("🤷 দুঃখিত, আপনার প্রশ্নের জন্য কোন উত্তর পাওয়া যায়নি।")
                    print("💡 অন্য শব্দ বা বাক্য দিয়ে চেষ্টা করুন।")
                else:
                    print(f"\n✅ {result['results_found']}টি প্রাসঙ্গিক উত্তর পাওয়া গেছে:")
                    print(result["formatted_answer"])
                    
                    # Show sources
                    print(f"\n📚 তথ্যসূত্র:")
                    for source in result["sources"]:
                        print(f"  • {source['chunk_id']} (সাদৃশ্য: {source['similarity']})")
                
            except KeyboardInterrupt:
                print("\n👋 প্রোগ্রাম বন্ধ করা হচ্ছে...")
                break
            except Exception as e:
                print(f"❌ ত্রুটি ঘটেছে: {str(e)}")
    
    def _show_history(self):
        """Show query history."""
        if not self.query_history:
            print("📜 কোন প্রশ্নের ইতিহাস নেই।")
            return
        
        print(f"\n📜 সাম্প্রতিক {len(self.query_history)}টি প্রশ্ন:")
        for i, entry in enumerate(self.query_history[-5:], 1):  # Show last 5
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            print(f"  {i}. [{timestamp}] {entry['query']} ({entry['results_count']}টি ফলাফল)")
    
    def _show_stats(self):
        """Show system statistics."""
        stats = self.vector_rag.vector_db.get_stats()
        print(f"\n📊 সিস্টেম তথ্য:")
        print(f"  🔢 Total Vectors: {stats['total_vectors']}")
        print(f"  📐 Vector Dimension: {stats['dimension']}")
        print(f"  🏗️  Index Type: {stats['index_type']}")
        print(f"  📄 Total Chunks: {stats['total_chunks']}")
        print(f"  💾 Database Path: {stats['database_path']}")
        print(f"  🕐 Queries Made: {len(self.query_history)}")
    
    def _show_help(self):
        """Show help information."""
        print(f"\n🆘 সাহায্য:")
        print(f"  📝 Commands:")
        print(f"    • 'history' - প্রশ্নের ইতিহাস দেখুন")
        print(f"    • 'stats' - সিস্টেম তথ্য দেখুন")
        print(f"    • 'help' - এই সাহায্য দেখুন")
        print(f"    • 'quit' - প্রোগ্রাম বন্ধ করুন")
        print(f"  🔍 Query Tips:")
        print(f"    • রবীন্দ্রনাথ সম্পর্কে বলুন")
        print(f"    • অপরিচিতা গল্পের চরিত্র কারা?")
        print(f"    • শব্দার্থ")
        print(f"    • প্রশ্ন ও উত্তর")
    
    def batch_query(self, queries: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """Process multiple queries at once."""
        if not self.is_ready:
            print("❌ System not ready")
            return []
        
        results = []
        print(f"🔄 Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"  Processing {i}/{len(queries)}: {query}")
            result = self.process_query(query)
            results.append(result)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        return results

def main():
    """Main function to run the query interface."""
    print("🚀 Starting Multilingual RAG Query Interface...")
    
    # Initialize the interface
    interface = QueryInterface(
        extracted_text_file="extracted_text.txt",
        vector_db_path="vector_database"
    )
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--rebuild":
            print("🔄 Force rebuilding vector database...")
            success = interface.initialize_system(force_rebuild=True)
        elif sys.argv[1] == "--batch":
            # Batch mode with predefined queries
            test_queries = [
                "রবীন্দ্রনাথ ঠাকুর",
                "অপরিচিতা গল্পের মূল চরিত্র কে?",
                "এই গল্পের প্রধান বিষয় কি?",
                "শব্দার্থ ও অর্থ",
                "HSC পরীক্ষার প্রশ্ন"
            ]
            
            if interface.initialize_system():
                results = interface.batch_query(test_queries, "batch_results.json")
                print(f"✅ Processed {len(results)} queries")
            return
        else:
            print(f"❓ Unknown argument: {sys.argv[1]}")
            print("Usage: python query_interface.py [--rebuild|--batch]")
            return
    else:
        success = interface.initialize_system()
    
    if success:
        # Start interactive mode
        interface.interactive_mode()
    else:
        print("❌ Failed to initialize system. Please check your files:")
        print("  • extracted_text.txt - should contain your document text")
        print("  • Make sure all required packages are installed")

if __name__ == "__main__":
    main()