import os
from dotenv import load_dotenv
from app.rag_system import MultilingualRAGSystem

load_dotenv()

def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX", "multilingual-rag")
    
    if not openai_api_key or not pinecone_api_key:
        print("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        return
    
    rag_system = MultilingualRAGSystem(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index=pinecone_index
    )
    
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    
    if os.path.exists(pdf_path):
        print("Processing PDF...")
        rag_system.process_pdf(pdf_path)
        print("PDF processed successfully!")
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            result = rag_system.query_with_sources(question)
            print(f"Answer: {result['answer']}")
    else:
        print(f"PDF file not found: {pdf_path}")

if __name__ == "__main__":
    main()