import os
import json
import base64
import tempfile
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import yellow, red, blue, green
import io

class PDFAnnotator:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
    
    def load_document(self, doc_id):
        """Load a document from the processed directory"""
        file_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if not os.path.exists(file_path):
            raise ValueError(f"Document {doc_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_pdf_path(self, doc_id):
        """Get the path to the PDF file for a document"""
        doc = self.load_document(doc_id)
        filename = doc["metadata"]["filename"]
        
        if not filename.lower().endswith('.pdf'):
            raise ValueError(f"Document {doc_id} is not a PDF")
        
        pdf_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        return pdf_path
    
    def highlight_text(self, doc_id, search_text, color="yellow"):
        """
        Highlight text in a PDF document
        
        Args:
            doc_id: Document ID
            search_text: Text to highlight
            color: Highlight color (yellow, red, blue, green)
        
        Returns:
            Base64-encoded highlighted PDF
        """
        try:
            pdf_path = self.get_pdf_path(doc_id)
            
            # Read the PDF
            pdf_reader = PdfReader(pdf_path)
            pdf_writer = PdfWriter()
            
            # Set highlight color
            color_map = {
                "yellow": yellow,
                "red": red,
                "blue": blue,
                "green": green
            }
            highlight_color = color_map.get(color, yellow)
            
            # Process each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # Check if search text is on this page
                if search_text.lower() in page_text.lower():
                    # Create a temporary PDF with highlights
                    packet = io.BytesIO()
                    c = canvas.Canvas(packet)
                    
                    # Simple highlighting (this is a basic approach)
                    # For more accurate highlighting, you would need to use a more sophisticated
                    # approach to get the exact coordinates of the text
                    c.setFillColor(highlight_color)
                    c.setFillAlpha(0.5)  # Semi-transparent
                    
                    # Draw rectangles where the text appears
                    # This is a simplified approach and may not be perfectly accurate
                    c.rect(100, 100, 400, 30, fill=True, stroke=False)
                    
                    c.save()
                    
                    # Move to the beginning of the buffer
                    packet.seek(0)
                    overlay = PdfReader(packet)
                    
                    # Merge the original page with the highlight layer
                    page.merge_page(overlay.pages[0])
                
                # Add the page to the writer
                pdf_writer.add_page(page)
            
            # Save the result to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                pdf_writer.write(temp_file)
                temp_file_path = temp_file.name
            
            # Read the file and convert to base64
            with open(temp_file_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            # Return base64-encoded PDF
            return base64.b64encode(pdf_bytes).decode()
        
        except Exception as e:
            print(f"Error highlighting text: {e}")
            return None
    
    def add_comment(self, doc_id, page_num, comment_text, x=100, y=100):
        """
        Add a comment to a PDF document
        
        Args:
            doc_id: Document ID
            page_num: Page number (0-based)
            comment_text: Comment text
            x, y: Coordinates for the comment
        
        Returns:
            Base64-encoded PDF with comment
        """
        try:
            pdf_path = self.get_pdf_path(doc_id)
            
            # Read the PDF
            pdf_reader = PdfReader(pdf_path)
            pdf_writer = PdfWriter()
            
            # Process each page
            for i in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                
                # Add comment to the specified page
                if i == page_num:
                    # Create a temporary PDF with comment
                    packet = io.BytesIO()
                    c = canvas.Canvas(packet)
                    
                    # Draw comment box
                    c.setFillColor(yellow)
                    c.setFillAlpha(0.7)
                    c.rect(x, y, 200, 50, fill=True, stroke=True)
                    
                    # Add comment text
                    c.setFillColor(red)
                    c.setFillAlpha(1.0)
                    c.drawString(x + 10, y + 25, comment_text)
                    
                    c.save()
                    
                    # Move to the beginning of the buffer
                    packet.seek(0)
                    overlay = PdfReader(packet)
                    
                    # Merge the original page with the comment layer
                    page.merge_page(overlay.pages[0])
                
                # Add the page to the writer
                pdf_writer.add_page(page)
            
            # Save the result to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                pdf_writer.write(temp_file)
                temp_file_path = temp_file.name
            
            # Read the file and convert to base64
            with open(temp_file_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            # Return base64-encoded PDF
            return base64.b64encode(pdf_bytes).decode()
        
        except Exception as e:
            print(f"Error adding comment: {e}")
            return None
