import os
import json
from datetime import datetime
from src.utils.text_chunker import TextChunker
from src.utils.mock_embeddings import MockEmbeddingsManager

class DocumentProcessor:
    def __init__(self, raw_dir, processed_dir, embeddings_dir="data/embeddings"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embeddings_manager = MockEmbeddingsManager(embeddings_dir)
        os.makedirs(processed_dir, exist_ok=True)

    def process_document(self, document, doc_type):
        """Process a document and save metadata"""
        # Create a unique ID for the document
        doc_id = f"{doc_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Chunk the document
        chunks = self.chunker.chunk_document(document)
        
        # Get token statistics
        token_stats = self.chunker.get_chunk_stats(chunks)
        
        # Calculate total tokens in original document
        total_doc_tokens = sum(self.chunker.count_tokens(page.page_content) for page in document)
        
        # Generate embeddings and save to vector store
        embedding_results = self.embeddings_manager.embed_chunks(chunks, doc_id)
        
        # Extract basic metadata
        metadata = {
            "id": doc_id,
            "type": doc_type,
            "date_processed": datetime.now().isoformat(),
            "page_count": len(document),
            "filename": os.path.basename(document[0].metadata.get("source", "unknown")),
            "token_count": total_doc_tokens,
            "chunk_stats": token_stats,
            "embedding_stats": embedding_results["embedding_stats"]
        }

        # Save processed document with metadata
        output_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": metadata,
                "pages": [{"page": i, "content": page.page_content}
                          for i, page in enumerate(document)],
                "chunks": [{"chunk_id": i, 
                            "content": chunk.page_content,
                            "tokens": self.chunker.count_tokens(chunk.page_content),
                            "metadata": chunk.metadata}
                           for i, chunk in enumerate(chunks)],
                "vector_store": {
                    "embeddings_path": embedding_results["embeddings_path"],
                    "metadata_path": embedding_results["metadata_path"]
                }
            }, f, indent=2)

        return doc_id, metadata, chunks
