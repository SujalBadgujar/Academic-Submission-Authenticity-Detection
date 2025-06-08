import PyPDF2
import os

class PDFHandler:
    def extract_text(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")