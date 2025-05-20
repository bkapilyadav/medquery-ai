import re
from datetime import datetime

class SimpleQAChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.conversation_history = []
    
    def _extract_keywords(self, query):
        """Extract important keywords from the query"""
        # Remove common words and keep important ones
        common_words = {'what', 'when', 'where', 'who', 'how', 'why', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near'}
        
        # Split query into words and filter out common words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords
    
    def _format_answer(self, query, results, doc_id):
        """Format the answer based on retrieved chunks"""
        if not results:
            return "I couldn't find any relevant information in the document."
        
        # Extract content from results
        contents = [result['content'] for result in results]
        
        # Simple answer formatting
        answer = f"Based on the document, here's what I found:\n\n"
        
        # Add relevant information from each chunk
        for i, content in enumerate(contents):
            answer += f"Source {i+1}:\n{content}\n\n"
        
        # Add a disclaimer
        answer += "Note: This information is extracted directly from the document and may require medical interpretation."
        
        return answer
    
    def add_to_history(self, query, answer):
        """Add the Q&A pair to conversation history"""
        self.conversation_history.append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_answer(self, query, doc_id, top_k=3):
        """Get answer for a query using retrieval-based QA"""
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, doc_id, top_k=top_k)
        
        # Format the answer
        answer = self._format_answer(query, results, doc_id)
        
        # Add to history
        self.add_to_history(query, answer)
        
        return {
            "answer": answer,
            "sources": results,
            "query": query,
            "doc_id": doc_id
        }
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
