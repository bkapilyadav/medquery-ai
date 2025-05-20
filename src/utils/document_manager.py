import os
import json
import shutil
from datetime import datetime

class DocumentManager:
    def __init__(self, raw_dir, processed_dir, embeddings_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.embeddings_dir = embeddings_dir
        
        # Create directories if they don't exist
        for directory in [raw_dir, processed_dir, embeddings_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def list_processed_documents(self):
        """List all processed documents with metadata"""
        documents = []
        
        for filename in os.listdir(self.processed_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.processed_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        documents.append({
                            'id': data['metadata']['id'],
                            'type': data['metadata']['type'],
                            'filename': data['metadata']['filename'],
                            'date_processed': data['metadata']['date_processed'],
                            'chunk_count': data['metadata']['chunk_stats']['total_chunks'],
                            'token_count': data['metadata']['token_count']
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        # Sort by date processed (newest first)
        documents.sort(key=lambda x: x['date_processed'], reverse=True)
        return documents
    
    def get_document_details(self, doc_id):
        """Get detailed information about a document"""
        file_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    
    def delete_document(self, doc_id):
        """Delete a document and its associated files"""
        # Delete processed document
        processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        # Delete embeddings
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        if os.path.exists(embeddings_path):
            os.remove(embeddings_path)
        
        # Delete metadata
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return True
    
    def export_document(self, doc_id, export_dir):
        """Export a document and its associated files to a directory"""
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy processed document
        processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if os.path.exists(processed_path):
            shutil.copy(processed_path, os.path.join(export_dir, f"{doc_id}.json"))
        
        # Copy embeddings
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        if os.path.exists(embeddings_path):
            shutil.copy(embeddings_path, os.path.join(export_dir, f"{doc_id}_embeddings.pkl"))
        
        # Copy metadata
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        if os.path.exists(metadata_path):
            shutil.copy(metadata_path, os.path.join(export_dir, f"{doc_id}_metadata.json"))
        
        return export_dir
