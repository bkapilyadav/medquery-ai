import os
import json
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings

class SimpleRetriever:
    def __init__(self, embeddings_dir="data/embeddings", use_openai=True):
        self.embeddings_dir = embeddings_dir
        self.use_openai = use_openai
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.model_name = "text-embedding-3-small"
    
    def list_available_documents(self):
        """List all documents available for retrieval"""
        docs = []
        for filename in os.listdir(self.embeddings_dir):
            if filename.endswith("_metadata.json"):
                doc_id = filename.replace("_metadata.json", "")
                with open(os.path.join(self.embeddings_dir, filename), 'r') as f:
                    metadata = json.load(f)
                docs.append({
                    "doc_id": doc_id,
                    "chunk_count": metadata.get("chunk_count", 0),
                    "embedding_model": metadata.get("embedding_model", "unknown")
                })
        return docs
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query, doc_id, top_k=3):
        """Retrieve most relevant chunks for a query from a specific document"""
        # Load embeddings
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        if not os.path.exists(embeddings_path):
            raise ValueError(f"No embeddings found for document {doc_id}")
        
        with open(embeddings_path, 'rb') as f:
            chunk_embeddings = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities
        similarities = [self.cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get results
        results = []
        for idx in top_indices:
            if idx < len(metadata["chunks"]):
                chunk = metadata["chunks"][idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "score": float(similarities[idx]),
                    "metadata": chunk["metadata"]
                })
        
        return results
