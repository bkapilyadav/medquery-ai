import os
import json
import difflib
import pandas as pd
import numpy as np
from collections import Counter

class DocumentComparison:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
    
    def load_document(self, doc_id):
        """Load a document from the processed directory"""
        file_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if not os.path.exists(file_path):
            raise ValueError(f"Document {doc_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def compare_metadata(self, doc_id1, doc_id2):
        """Compare metadata between two documents"""
        doc1 = self.load_document(doc_id1)
        doc2 = self.load_document(doc_id2)
        
        metadata1 = doc1["metadata"]
        metadata2 = doc2["metadata"]
        
        # Create comparison dataframe
        comparison = []
        
        # Compare basic metadata
        for key in set(metadata1.keys()) | set(metadata2.keys()):
            if key in ["chunk_stats", "embedding_stats"]:
                continue
                
            val1 = metadata1.get(key, "N/A")
            val2 = metadata2.get(key, "N/A")
            
            comparison.append({
                "Field": key,
                doc_id1: val1,
                doc_id2: val2,
                "Different": val1 != val2
            })
        
        # Compare chunk stats
        for key in set(metadata1.get("chunk_stats", {}).keys()) | set(metadata2.get("chunk_stats", {}).keys()):
            val1 = metadata1.get("chunk_stats", {}).get(key, "N/A")
            val2 = metadata2.get("chunk_stats", {}).get(key, "N/A")
            
            comparison.append({
                "Field": f"chunk_stats.{key}",
                doc_id1: val1,
                doc_id2: val2,
                "Different": val1 != val2
            })
        
        return pd.DataFrame(comparison)
    
    def compare_content(self, doc_id1, doc_id2):
        """Compare content between two documents"""
        doc1 = self.load_document(doc_id1)
        doc2 = self.load_document(doc_id2)
        
        # Extract full text content
        text1 = "\n".join([page["content"] for page in doc1["pages"]])
        text2 = "\n".join([page["content"] for page in doc2["pages"]])
        
        # Compare using difflib
        diff = difflib.unified_diff(
            text1.splitlines(),
            text2.splitlines(),
            lineterm='',
            n=3
        )
        
        return "\n".join(diff)
    
    def compare_key_terms(self, doc_id1, doc_id2, top_n=20):
        """Compare key terms between two documents"""
        doc1 = self.load_document(doc_id1)
        doc2 = self.load_document(doc_id2)
        
        # Extract full text content
        text1 = "\n".join([page["content"] for page in doc1["pages"]])
        text2 = "\n".join([page["content"] for page in doc2["pages"]])
        
        # Simple tokenization and counting
        def get_top_terms(text, n):
            words = text.lower().split()
            # Filter out common words and short words
            stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
            words = [word for word in words if word not in stopwords and len(word) > 2]
            return Counter(words).most_common(n)
        
        terms1 = get_top_terms(text1, top_n)
        terms2 = get_top_terms(text2, top_n)
        
        # Convert to dictionaries for easier comparison
        terms1_dict = dict(terms1)
        terms2_dict = dict(terms2)
        
        # Create comparison dataframe
        comparison = []
        
        # Add all terms from both documents
        all_terms = set(terms1_dict.keys()) | set(terms2_dict.keys())
        for term in all_terms:
            count1 = terms1_dict.get(term, 0)
            count2 = terms2_dict.get(term, 0)
            
            comparison.append({
                "Term": term,
                f"{doc_id1} Count": count1,
                f"{doc_id2} Count": count2,
                "Difference": count1 - count2
            })
        
        # Sort by absolute difference
        comparison.sort(key=lambda x: abs(x["Difference"]), reverse=True)
        
        return pd.DataFrame(comparison)
