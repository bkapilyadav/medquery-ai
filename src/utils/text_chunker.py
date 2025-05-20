from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def _token_length(self, text):
        """Count tokens using tiktoken for accurate OpenAI token counting"""
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
        
    def chunk_document(self, document):
        """Split a document into chunks based on token size"""
        return self.text_splitter.split_documents(document)
        
    def count_tokens(self, text):
        """Count tokens in a text string"""
        return self._token_length(text)
        
    def get_chunk_stats(self, chunks):
        """Get statistics about the chunks"""
        token_counts = [self.count_tokens(chunk.page_content) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks) if chunks else 0,
            "max_tokens": max(token_counts) if chunks else 0,
            "min_tokens": min(token_counts) if chunks else 0
        }
