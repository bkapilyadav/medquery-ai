import os
import json
import re
from datetime import datetime
import pandas as pd

class AdvancedSearch:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
    
    def load_all_documents(self):
        """Load metadata for all documents"""
        documents = []
        
        for filename in os.listdir(self.processed_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.processed_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        documents.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def search_by_metadata(self, filters):
        """
        Search documents by metadata filters
        
        Args:
            filters: Dict of metadata filters
                - type: Document type
                - date_from: Process date from (YYYY-MM-DD)
                - date_to: Process date to (YYYY-MM-DD)
                - min_tokens: Minimum token count
                - max_tokens: Maximum token count
        """
        documents = self.load_all_documents()
        results = []
        
        for doc in documents:
            metadata = doc["metadata"]
            
            # Check document type
            if "type" in filters and filters["type"] and metadata["type"] != filters["type"]:
                continue
            
            # Check date range
            if "date_from" in filters and filters["date_from"]:
                date_from = datetime.strptime(filters["date_from"], "%Y-%m-%d")
                doc_date = datetime.fromisoformat(metadata["date_processed"].split("T")[0])
                if doc_date < date_from:
                    continue
            
            if "date_to" in filters and filters["date_to"]:
                date_to = datetime.strptime(filters["date_to"], "%Y-%m-%d")
                doc_date = datetime.fromisoformat(metadata["date_processed"].split("T")[0])
                if doc_date > date_to:
                    continue
            
            # Check token count
            if "min_tokens" in filters and filters["min_tokens"]:
                if metadata["token_count"] < filters["min_tokens"]:
                    continue
            
            if "max_tokens" in filters and filters["max_tokens"]:
                if metadata["token_count"] > filters["max_tokens"]:
                    continue
            
            # Document passed all filters
            results.append({
                "id": metadata["id"],
                "type": metadata["type"],
                "filename": metadata["filename"],
                "date_processed": metadata["date_processed"],
                "token_count": metadata["token_count"],
                "chunk_count": metadata["chunk_stats"]["total_chunks"]
            })
        
        return results
    
    def search_by_content(self, query, case_sensitive=False):
        """
        Search document content for a specific query
        
        Args:
            query: Search query string
            case_sensitive: Whether to perform case-sensitive search
        """
        documents = self.load_all_documents()
        results = []
        
        for doc in documents:
            metadata = doc["metadata"]
            
            # Search in document content
            matches = []
            
            for i, page in enumerate(doc["pages"]):
                content = page["content"]
                
                # Perform search
                if case_sensitive:
                    found = query in content
                    match_positions = [m.start() for m in re.finditer(re.escape(query), content)]
                else:
                    found = query.lower() in content.lower()
                    match_positions = [m.start() for m in re.finditer(re.escape(query.lower()), content.lower())]
                
                if found:
                    # Extract context around matches
                    for pos in match_positions:
                        # Get context (50 chars before and after)
                        start = max(0, pos - 50)
                        end = min(len(content), pos + len(query) + 50)
                        context = content[start:end]
                        
                        # Highlight the match
                        if case_sensitive:
                            highlighted = context.replace(query, f"**{query}**")
                        else:
                            # Case-insensitive replacement is more complex
                            pattern = re.compile(re.escape(query), re.IGNORECASE)
                            highlighted = pattern.sub(f"**\\g<0>**", context)
                        
                        matches.append({
                            "page": i,
                            "context": highlighted
                        })
            
            if matches:
                results.append({
                    "id": metadata["id"],
                    "type": metadata["type"],
                    "filename": metadata["filename"],
                    "matches": matches,
                    "match_count": len(matches)
                })
        
        # Sort by number of matches (descending)
        results.sort(key=lambda x: x["match_count"], reverse=True)
        
        return results
