import os
import json
import pickle
import numpy as np
from src.utils.enhanced_embeddings import EnhancedEmbeddingsManager

class MultiDocRetriever:
    def __init__(self, embeddings_dir="data/embeddings", provider="mock"):
        self.embeddings_dir = embeddings_dir
        self.embeddings_manager = EnhancedEmbeddingsManager(embeddings_dir, provider)
    
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
                    "embedding_model": metadata.get("embedding_model", "unknown"),
                    "embedding_provider": metadata.get("embedding_provider", "unknown"),
                    "type": doc_id.split("_")[0] if "_" in doc_id else "unknown"
                })
        return docs
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve_from_single_doc(self, query, doc_id, top_k=3):
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
        query_embedding = self.embeddings_manager.embed_query(query)
        
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
                    "metadata": chunk["metadata"],
                    "doc_id": doc_id
                })
        
        return results
    
    def retrieve_from_multiple_docs(self, query, doc_ids, top_k=3):
        """Retrieve most relevant chunks across multiple documents"""
        all_results = []
        
        for doc_id in doc_ids:
            try:
                results = self.retrieve_from_single_doc(query, doc_id, top_k=top_k)
                all_results.extend(results)
            except Exception as e:
                print(f"Error retrieving from {doc_id}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k overall
        return all_results[:top_k]
    
    def retrieve_by_type(self, query, doc_type, top_k=3):
        """Retrieve most relevant chunks from documents of a specific type"""
        # Get all documents of the specified type
        docs = self.list_available_documents()
        doc_ids = [doc["doc_id"] for doc in docs if doc["type"] == doc_type]
        
        return self.retrieve_from_multiple_docs(query, doc_ids, top_k=top_k)
