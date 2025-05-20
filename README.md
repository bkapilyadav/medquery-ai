# MedQuery AI

An intelligent healthcare document Q&A assistant that allows you to process, query, and analyze medical documents.

## Features

- Document processing and chunking
- Natural language querying
- Multi-document search and comparison
- Document visualization and annotation
- Advanced search capabilities
- User authentication
- Export/import functionality

## Demo

[View the live demo on Streamlit Cloud](https://your-streamlit-cloud-url)

## Local Setup

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/medquery-ai.git
   cd medquery-ai
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   - Create a `.streamlit/secrets.toml` file with your API keys
   - Or set environment variables directly

5. Run the application
   ```bash
   streamlit run app.py
   ```

## Usage

1. Register for an account or log in
2. Upload medical documents
3. Process documents to make them searchable
4. Use the chat interface to ask questions about your documents

## License

MIT
