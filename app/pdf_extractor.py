from langchain.schema import Document
from pdf2image import convert_from_path
import easyocr
import os
import tempfile
from typing import List

class PDFExtractor:
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        if use_ocr:
            # Initialize EasyOCR reader for Bengali and English
            self.ocr_reader = easyocr.Reader(['bn', 'en'])
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.use_ocr:
            return self._extract_with_ocr(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_ocr(self, pdf_path: str) -> List[Document]:
        documents = []
        
        try:
            print("Converting PDF to images...")
            # Convert PDF pages to images
            pages = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(pages)} pages to images")
            
            # Create temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num, page in enumerate(pages):
                    print(f"Processing page {page_num + 1}/{len(pages)} with OCR...")
                    
                    # Save page as temporary image
                    img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    page.save(img_path, 'PNG')
                    
                    # Extract text using OCR
                    results = self.ocr_reader.readtext(img_path, detail=0)
                    text = '\n'.join(results)
                    
                    if text.strip():  # Only add non-empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path, 
                                "page": page_num + 1,
                                "extraction_method": "ocr"
                            }
                        )
                        documents.append(doc)
            
            print(f"OCR extraction completed for {len(documents)} pages")
            return documents
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            print("Falling back to PyPDF extraction...")
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_pypdf(self, pdf_path: str) -> List[Document]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add extraction method to metadata
            for doc in documents:
                doc.metadata["extraction_method"] = "pypdf"
            
            return documents
        except Exception as e:
            print(f"PyPDF extraction also failed: {e}")
            return []
    
    def extract_text_as_string(self, pdf_path: str) -> str:
        documents = self.extract_text_from_pdf(pdf_path)
        return "\n\n".join([doc.page_content for doc in documents])