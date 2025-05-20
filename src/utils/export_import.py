import os
import json
import shutil
import zipfile
import tempfile
import datetime

class ExportImport:
    def __init__(self, raw_dir, processed_dir, embeddings_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.embeddings_dir = embeddings_dir
    
    def export_document(self, doc_id, include_raw=True):
        """
        Export a document and its associated files to a zip file
        
        Args:
            doc_id: Document ID to export
            include_raw: Whether to include the raw document file
        
        Returns:
            Path to the created zip file
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create subdirectories
            os.makedirs(os.path.join(temp_dir, "processed"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "embeddings"), exist_ok=True)
            if include_raw:
                os.makedirs(os.path.join(temp_dir, "raw"), exist_ok=True)
            
            # Copy processed document
            processed_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            if os.path.exists(processed_path):
                shutil.copy(processed_path, os.path.join(temp_dir, "processed", f"{doc_id}.json"))
                
                # Get raw filename from processed document
                with open(processed_path, 'r') as f:
                    data = json.load(f)
                    raw_filename = data["metadata"].get("filename", "")
            else:
                raise ValueError(f"Processed document {doc_id} not found")
            
            # Copy embeddings
            embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
            if os.path.exists(embeddings_path):
                shutil.copy(embeddings_path, os.path.join(temp_dir, "embeddings", f"{doc_id}_embeddings.pkl"))
            
            # Copy metadata
            metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
            if os.path.exists(metadata_path):
                shutil.copy(metadata_path, os.path.join(temp_dir, "embeddings", f"{doc_id}_metadata.json"))
            
            # Copy raw document if requested
            if include_raw and raw_filename:
                raw_path = os.path.join(self.raw_dir, raw_filename)
                if os.path.exists(raw_path):
                    shutil.copy(raw_path, os.path.join(temp_dir, "raw", raw_filename))
            
            # Create a manifest file
            manifest = {
                "doc_id": doc_id,
                "export_date": datetime.datetime.now().isoformat(),
                "files": {
                    "processed": [f"{doc_id}.json"],
                    "embeddings": [f"{doc_id}_embeddings.pkl", f"{doc_id}_metadata.json"],
                    "raw": [raw_filename] if include_raw and raw_filename else []
                }
            }
            
            with open(os.path.join(temp_dir, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create zip file
            zip_path = f"{doc_id}_export.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            return zip_path
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def import_document(self, zip_path):
        """
        Import a document from a zip file
        
        Args:
            zip_path: Path to the zip file
        
        Returns:
            Imported document ID
        """
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Read manifest
            with open(os.path.join(temp_dir, "manifest.json"), 'r') as f:
                manifest = json.load(f)
            
            doc_id = manifest["doc_id"]
            
            # Copy processed document
            for filename in manifest["files"]["processed"]:
                src = os.path.join(temp_dir, "processed", filename)
                dst = os.path.join(self.processed_dir, filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            # Copy embeddings
            for filename in manifest["files"]["embeddings"]:
                src = os.path.join(temp_dir, "embeddings", filename)
                dst = os.path.join(self.embeddings_dir, filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            # Copy raw document
            for filename in manifest["files"].get("raw", []):
                src = os.path.join(temp_dir, "raw", filename)
                dst = os.path.join(self.raw_dir, filename)
                if os.path.exists(src):
                    shutil.copy(src, dst)
            
            return doc_id
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
