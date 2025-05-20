import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class DocumentLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_document(self, file_path):
        """Load a document based on its file extension"""
        _, ext = os.path.splitext(file_path)

        if ext.lower() == '.pdf':
            return self._load_pdf(file_path)
        elif ext.lower() == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_text(self, file_path):
        return TextLoader(file_path).load()
