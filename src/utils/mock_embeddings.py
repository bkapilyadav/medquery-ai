import os
import json
import numpy as np
import pickle
import random
import time
from hashlib import md5

class MockEmbeddingsManager:
    def __init__(self, embeddings_dir="data/embeddings"):
        self.embeddings_dir = embeddings_dir
        os.makedirs(embeddings_dir, exist_ok=True)
        
    def _generate_mock_embedding(self, text, dim=384):
        """Generate a deterministic mock embedding based on text content"""
        # Use MD5 hash of text to seed random number generator for deterministic output
        seed = int(md5(text.encode()).hexdigest(), 16) % 10000000
        np.random.seed(seed)
        
        # Generate a random vector and normalize it
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_chunks(self, chunks, doc_id):
        """Create mock embeddings for document chunks"""
        start_time = time.time()
        
        # Get text from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate mock embeddings
        embeddings = [self._generate_mock_embedding(text) for text in texts]
        
        # Save embeddings to disk
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "embedding_model": "mock_embeddings",
            "chunks": [
                {
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for i, chunk in enumerate(chunks)
            ],
            "processing_time": time.time() - start_time
        }
        
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_stats": {
                "chunk_count": len(chunks),
                "embedding_model": "mock_embeddings",
                "processing_time": time.time() - start_time
            }
        }

class MockRetriever:
    def __init__(self, embeddings_dir="data/embeddings"):
        self.embeddings_dir = embeddings_dir
    
    def _generate_mock_embedding(self, text, dim=384):
        """Generate a deterministic mock embedding based on text content"""
        # Use MD5 hash of text to seed random number generator for deterministic output
        seed = int(md5(text.encode()).hexdigest(), 16) % 10000000
        np.random.seed(seed)
        
        # Generate a random vector and normalize it
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
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
        
        # Generate query embedding
        query_embedding = self._generate_mock_embedding(query)
        
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
