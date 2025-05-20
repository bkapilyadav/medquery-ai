import os
import json
import numpy as np
import pickle
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class EnhancedEmbeddingsManager:
    def __init__(self, embeddings_dir="data/embeddings", provider="mock"):
        """
        Initialize embeddings manager with different provider options
        
        Args:
            embeddings_dir: Directory to store embeddings
            provider: One of "mock", "openai", or "huggingface"
        """
        self.embeddings_dir = embeddings_dir
        self.provider = provider
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize embeddings model based on provider
        if provider == "openai":
            # Check if API key is available
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.model_name = "text-embedding-3-small"
            self.embedding_dim = 1536
            self.cost_per_1k_tokens = 0.00002  # $0.00002 per 1K tokens
        elif provider == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            self.model_name = "all-MiniLM-L6-v2"
            self.embedding_dim = 384
            self.cost_per_1k_tokens = 0  # Free
        else:  # mock
            self.embeddings = None
            self.model_name = "mock_embeddings"
            self.embedding_dim = 384
            self.cost_per_1k_tokens = 0
        
        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0
    
    def _generate_mock_embedding(self, text):
        """Generate a deterministic mock embedding based on text content"""
        # Use hash of text to seed random number generator for deterministic output
        seed = hash(text) % 10000000
        np.random.seed(seed)
        
        # Generate a random vector and normalize it
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_documents(self, texts):
        """Embed a list of texts"""
        if self.provider == "mock":
            return [self._generate_mock_embedding(text) for text in texts]
        else:
            return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text):
        """Embed a single query text"""
        if self.provider == "mock":
            return self._generate_mock_embedding(text)
        else:
            return self.embeddings.embed_query(text)
    
    def embed_chunks(self, chunks, doc_id):
        """Embed document chunks and save embeddings to disk"""
        start_time = time.time()
        
        # Get text from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Track token usage for OpenAI
        if self.provider == "openai":
            from src.utils.text_chunker import TextChunker
            chunker = TextChunker()
            token_count = sum(chunker.count_tokens(text) for text in texts)
            self.total_tokens += token_count
            self.total_cost += (token_count / 1000) * self.cost_per_1k_tokens
        else:
            token_count = 0
        
        # Generate embeddings
        embeddings = self.embed_documents(texts)
        
        # Save embeddings to disk
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "embedding_model": self.model_name,
            "embedding_provider": self.provider,
            "chunks": [
                {
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for i, chunk in enumerate(chunks)
            ],
            "processing_time": time.time() - start_time
        }
        
        if self.provider == "openai":
            metadata["token_count"] = token_count
            metadata["estimated_cost"] = (token_count / 1000) * self.cost_per_1k_tokens
        
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_stats": {
                "chunk_count": len(chunks),
                "embedding_model": self.model_name,
                "embedding_provider": self.provider,
                "processing_time": time.time() - start_time,
                "token_count": token_count if self.provider == "openai" else None,
                "estimated_cost": (token_count / 1000) * self.cost_per_1k_tokens if self.provider == "openai" else 0
            }
        }
    
    def get_cost_summary(self):
        """Get summary of embedding costs"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cost_per_1k_tokens": self.cost_per_1k_tokens
        }
