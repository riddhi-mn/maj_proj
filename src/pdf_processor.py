"""PDF processing and chunking."""
import os
import pdfplumber
from typing import List, Dict
from src.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Process PDF files and prepare them for vector storage."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extract text from PDF file and return as chunks with metadata."""
        chunks = []
        filename = os.path.basename(pdf_path)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n\n--- Page {page_num} ---\n\n{page_text}"
                        page_texts.append((page_num, page_text))
                
                # Split text into chunks
                text_chunks = self.text_splitter.split_text(full_text)
                
                # Map chunks back to page numbers
                for chunk_idx, chunk in enumerate(text_chunks):
                    # Find which page this chunk likely came from
                    page_num = self._find_page_for_chunk(chunk, page_texts)
                    
                    chunks.append({
                        "content": chunk,
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                    })
        
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
        
        return chunks
    
    def _find_page_for_chunk(self, chunk: str, page_texts: List[tuple]) -> int:
        """Find the page number for a given chunk."""
        # Simple approach: find the page with the most matching text
        chunk_snippet = chunk[:100].lower()
        best_match = 1
        max_matches = 0
        
        for page_num, page_text in page_texts:
            page_snippet = page_text[:200].lower()
            matches = sum(1 for word in chunk_snippet.split() if word in page_snippet)
            if matches > max_matches:
                max_matches = matches
                best_match = page_num
        
        return best_match
    
    def process_directory(self, directory: str = None) -> List[Dict[str, any]]:
        """Process all PDF files in a directory."""
        directory = directory or Config.PDF_DIR
        all_chunks = []
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            return all_chunks
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return all_chunks
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            print(f"Processing: {pdf_file}")
            chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"  Extracted {len(chunks)} chunks")
        
        return all_chunks

