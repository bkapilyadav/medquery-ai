import os
import json
import numpy as np
import pickle
from langchain_openai import OpenAIEmbeddings
import time

class SimpleEmbeddingsManager:
    def __init__(self, embeddings_dir="data/embeddings", use_openai=True):
        self.embeddings_dir = embeddings_dir
        self.use_openai = use_openai
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize embeddings model
        if use_openai:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.model_name = "text-embedding-3-small"
        else:
            # Fallback to OpenAI but log a warning
            print("Warning: Only OpenAI embeddings are supported in this simplified version")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.model_name = "text-embedding-3-small"
        
        # Cost tracking for OpenAI
        self.total_tokens = 0
        self.total_cost = 0
        self.cost_per_1k_tokens = 0.00002  # $0.00002 per 1K tokens for text-embedding-3-small
    
    def embed_chunks(self, chunks, doc_id):
        """Embed document chunks and save embeddings to disk"""
        start_time = time.time()
        
        # Get text from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Track token usage for OpenAI
        from src.utils.text_chunker import TextChunker
        chunker = TextChunker()
        token_count = sum(chunker.count_tokens(text) for text in texts)
        self.total_tokens += token_count
        self.total_cost += (token_count / 1000) * self.cost_per_1k_tokens
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Save embeddings to disk
        embeddings_path = os.path.join(self.embeddings_dir, f"{doc_id}_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save metadata
        metadata = {
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "embedding_model": self.model_name,
            "chunks": [
                {
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for i, chunk in enumerate(chunks)
            ],
            "processing_time": time.time() - start_time,
            "token_count": token_count,
            "estimated_cost": (token_count / 1000) * self.cost_per_1k_tokens
        }
        
        metadata_path = os.path.join(self.embeddings_dir, f"{doc_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_stats": {
                "chunk_count": len(chunks),
                "embedding_model": self.model_name,
                "processing_time": time.time() - start_time,
                "token_count": token_count,
                "estimated_cost": (token_count / 1000) * self.cost_per_1k_tokens
            }
        }
    
    def get_cost_summary(self):
        """Get summary of embedding costs"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "cost_per_1k_tokens": self.cost_per_1k_tokens
        }
