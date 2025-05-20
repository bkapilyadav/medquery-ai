import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import io
import base64

class DocumentVisualizer:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
    
    def load_document(self, doc_id):
        """Load a document from the processed directory"""
        file_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if not os.path.exists(file_path):
            raise ValueError(f"Document {doc_id} not found")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def plot_chunk_distribution(self, doc_id):
        """Plot the distribution of chunk sizes"""
        doc = self.load_document(doc_id)
        
        # Get token counts for each chunk
        token_counts = [chunk["tokens"] for chunk in doc["chunks"]]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        sns.histplot(token_counts, bins=20, kde=True)
        plt.title(f"Chunk Size Distribution - {doc_id}")
        plt.xlabel("Tokens per Chunk")
        plt.ylabel("Count")
        plt.axvline(x=np.mean(token_counts), color='r', linestyle='--', label=f"Mean: {np.mean(token_counts):.1f}")
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 string for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode()
    
    def plot_key_terms(self, doc_id, top_n=20):
        """Plot the most frequent terms in the document"""
        doc = self.load_document(doc_id)
        
        # Extract full text content
        text = "\n".join([page["content"] for page in doc["pages"]])
        
        # Simple tokenization and counting
        words = text.lower().split()
        # Filter out common words and short words
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'and', 'or', 'is', 'are', 'was', 'were'}
        words = [word for word in words if word not in stopwords and len(word) > 2]
        term_counts = Counter(words).most_common(top_n)
        
        # Create dataframe
        df = pd.DataFrame(term_counts, columns=['Term', 'Count'])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Term', data=df)
        plt.title(f"Top {top_n} Terms - {doc_id}")
        plt.xlabel("Count")
        plt.ylabel("Term")
        plt.tight_layout()
        
        # Convert plot to base64 string for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode()
    
    def plot_document_comparison(self, doc_ids, metric='chunk_count'):
        """Plot comparison of a metric across multiple documents"""
        data = []
        
        for doc_id in doc_ids:
            try:
                doc = self.load_document(doc_id)
                
                if metric == 'chunk_count':
                    value = doc["metadata"]["chunk_stats"]["total_chunks"]
                elif metric == 'token_count':
                    value = doc["metadata"]["token_count"]
                elif metric == 'avg_chunk_size':
                    value = doc["metadata"]["chunk_stats"]["avg_tokens_per_chunk"]
                else:
                    value = 0
                
                data.append({
                    'Document': doc_id,
                    'Value': value,
                    'Type': doc["metadata"]["type"]
                })
            except Exception as e:
                print(f"Error loading {doc_id}: {e}")
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Document', y='Value', hue='Type', data=df)
        plt.title(f"Comparison of {metric} Across Documents")
        plt.xlabel("Document")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode()
