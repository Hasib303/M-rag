import os
from app.config import settings
from app.rag_system import MultilingualRAGSystem

def main():
    """
    Main function to run the RAG system in command-line interface mode.
    """
    print("Initializing RAG system...")
    
    # The settings object automatically validates that the required keys are present.
    rag_system = MultilingualRAGSystem(
        openai_api_key=settings.OPENAI_API_KEY,
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index=settings.PINECONE_INDEX,
        retriever_k=settings.RETRIEVER_K
    )
    
    print("RAG system initialized successfully.")
    
    # Example PDF path, you can make this dynamic
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}...")
        try:
            rag_system.process_pdf(pdf_path)
            print("PDF processed and ingested successfully!")
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return # Exit if PDF processing fails
            
        # Interactive query loop
        while True:
            try:
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower().strip() == 'quit':
                    break
                if not question.strip():
                    continue
                
                result = rag_system.query_with_sources(question)
                
                print("\n--- Answer ---")
                print(result['answer'])
                    
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
    else:
        print(f"Error: PDF file not found at '{pdf_path}'")
        print("Please make sure the file exists and the path is correct.")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please place your PDF files there.")
        
    main()
