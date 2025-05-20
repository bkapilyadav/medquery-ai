import os
import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from src.utils.document_loader import DocumentLoader
from src.pipelines.document_processor import DocumentProcessor
from src.utils.text_chunker import TextChunker
from src.utils.mock_embeddings import MockRetriever
from src.chains.qa_chain import SimpleQAChain
from src.utils.document_manager import DocumentManager
from src.utils.multi_doc_retriever import MultiDocRetriever
from src.utils.document_comparison import DocumentComparison
from src.utils.visualizations import DocumentVisualizer
from src.utils.advanced_search import AdvancedSearch
from src.utils.export_import import ExportImport
from src.utils.pdf_annotation import PDFAnnotator
from src.utils.user_auth import UserAuth

# Initialize components
loader = DocumentLoader("data/raw")
processor = DocumentProcessor("data/raw", "data/processed")
retriever = MockRetriever()
qa_chain = SimpleQAChain(retriever)
doc_manager = DocumentManager("data/raw", "data/processed", "data/embeddings")
multi_retriever = MultiDocRetriever()
doc_comparison = DocumentComparison()
visualizer = DocumentVisualizer()
advanced_search = AdvancedSearch()
export_import = ExportImport("data/raw", "data/processed", "data/embeddings")
pdf_annotator = PDFAnnotator()
user_auth = UserAuth()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.session_token = None

# Authentication UI
if not st.session_state.authenticated:
    st.title("MedQuery AI - Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            success, result = user_auth.authenticate(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.session_token = result
                st.session_state.role = user_auth.get_user_role(result)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = user_auth.register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
else:
    # Main application UI
   st.title("🏥 MedQuery AI: Intelligent Medical Document Analysis")

    
    # Sidebar with user info and logout
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        st.write(f"Role: **{st.session_state.role}**")

        if st.button("Logout"):
            user_auth.logout(st.session_state.session_token)
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.session_token = None
            st.rerun()

    # Create tabs
    tabs = ["Process Documents", "Query Documents", "Chat", "Multi-Document Chat",
            "Document Comparison", "Visualizations", "Advanced Search",
            "Export/Import", "PDF Annotation", "Document Management"]

    selected_tab = st.selectbox("Select Function", tabs)

    if selected_tab == "Process Documents":
        st.header("Upload & Process Documents")

        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

        if uploaded_file:
            # Save the uploaded file
            file_path = os.path.join("data/raw", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File uploaded: {uploaded_file.name}")

            # Process the document
            doc_type = st.selectbox("Document Type",
                                   ["discharge_summary", "prescription", "lab_report", "clinical_note"])

            chunk_size = st.slider("Chunk Size (tokens)", min_value=100, max_value=2000, value=1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap (tokens)", min_value=0, max_value=500, value=200, step=50)

            # Update chunker settings by creating a new chunker
            if chunk_size != processor.chunker.chunk_size or chunk_overlap != processor.chunker.chunk_overlap:
                processor.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Load the document
                    document = loader.load_document(file_path)

                    # Process and save
                    doc_id, metadata, chunks = processor.process_document(document, doc_type)

                    st.success(f"Document processed successfully! ID: {doc_id}")

                    # Display token information
                    st.subheader("Document Statistics")
                    st.write(f"Total tokens: {metadata['token_count']}")
                    st.write(f"Total chunks: {metadata['chunk_stats']['total_chunks']}")
                    st.write(f"Average tokens per chunk: {metadata['chunk_stats']['avg_tokens_per_chunk']:.1f}")

                    # Display embedding information
                    st.subheader("Embedding Statistics")
                    st.write(f"Embedding model: {metadata['embedding_stats']['embedding_model']}")
                    st.write(f"Processing time: {metadata['embedding_stats']['processing_time']:.2f} seconds")

                    # Display chunks
                    st.subheader("Document Chunks")
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1} ({processor.chunker.count_tokens(chunk.page_content)} tokens)"):
                            st.write(chunk.page_content)

    elif selected_tab == "Query Documents":
        st.header("Query Documents")

        # List available documents
        available_docs = retriever.list_available_documents()

        if not available_docs:
            st.warning("No documents available. Please process a document first.")
        else:
            # Select document
            doc_options = {f"{doc['doc_id']} ({doc['chunk_count']} chunks)": doc['doc_id'] for doc in available_docs}
            selected_doc = st.selectbox("Select a document to query", options=list(doc_options.keys()))
            doc_id = doc_options[selected_doc]

            # Query input
            query = st.text_input("Enter your query")
            top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

            if query and st.button("Search"):
                with st.spinner("Searching..."):
                    results = retriever.retrieve(query, doc_id, top_k=top_k)

                    st.subheader("Search Results")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result['score']:.4f})"):
                            st.write(result['content'])

    elif selected_tab == "Chat":
        st.header("Chat with your Documents")

        # List available documents
        available_docs = retriever.list_available_documents()

        if not available_docs:
            st.warning("No documents available. Please process a document first.")
        else:
            # Select document
            doc_options = {f"{doc['doc_id']} ({doc['chunk_count']} chunks)": doc['doc_id'] for doc in available_docs}
            selected_doc_chat = st.selectbox("Select a document to chat with", options=list(doc_options.keys()), key="chat_doc_select")
            doc_id_chat = doc_options[selected_doc_chat]

            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your document"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate a response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = qa_chain.get_answer(prompt, doc_id_chat, top_k=3)
                        st.markdown(response["answer"])

                        # Show sources if available
                        if response["sources"]:
                            with st.expander("View Sources"):
                                for i, source in enumerate(response["sources"]):
                                    st.markdown(f"**Source {i+1}** (Score: {source['score']:.4f})")
                                    st.markdown(source["content"])

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    elif selected_tab == "Multi-Document Chat":
        st.header("Chat with Multiple Documents")

        # List available documents
        available_docs = multi_retriever.list_available_documents()

        if not available_docs:
            st.warning("No documents available. Please process a document first.")
        else:
            # Select multiple documents
            doc_options = {f"{doc['doc_id']} ({doc['chunk_count']} chunks)": doc['doc_id'] for doc in available_docs}
            selected_docs = st.multiselect("Select documents to chat with", options=list(doc_options.keys()))

            if not selected_docs:
                st.warning("Please select at least one document.")
            else:
                doc_ids = [doc_options[doc] for doc in selected_docs]

                # Query input
                query = st.text_input("Enter your query")
                top_k = st.slider("Number of results per document", min_value=1, max_value=5, value=2)

                if query and st.button("Search"):
                    with st.spinner("Searching across multiple documents..."):
                        results = multi_retriever.retrieve_from_multiple_docs(query, doc_ids, top_k=top_k)

                        st.subheader("Search Results")
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} (Score: {result['score']:.4f}, Document: {result['doc_id']})"):
                                st.write(result['content'])

    elif selected_tab == "Document Comparison":
        st.header("Document Comparison")

        # List available documents
        documents = doc_manager.list_processed_documents()

        if len(documents) < 2:
            st.warning("You need at least two documents to compare. Please process more documents.")
        else:
            # Select documents to compare
            doc1 = st.selectbox("Select first document", options=[f"{doc['id']} - {doc['filename']}" for doc in documents], key="doc1")
            doc2 = st.selectbox("Select second document", options=[f"{doc['id']} - {doc['filename']}" for doc in documents], key="doc2")

            doc1_id = doc1.split(" - ")[0]
            doc2_id = doc2.split(" - ")[0]

            comparison_type = st.selectbox("Comparison Type", ["Metadata", "Content", "Key Terms"])

            if st.button("Compare Documents"):
                with st.spinner("Comparing documents..."):
                    if comparison_type == "Metadata":
                        comparison_df = doc_comparison.compare_metadata(doc1_id, doc2_id)
                        st.dataframe(comparison_df)

                    elif comparison_type == "Content":
                        diff = doc_comparison.compare_content(doc1_id, doc2_id)
                        st.text(diff)

                    elif comparison_type == "Key Terms":
                        terms_df = doc_comparison.compare_key_terms(doc1_id, doc2_id)
                        st.dataframe(terms_df)

    elif selected_tab == "Visualizations":
        st.header("Document Visualizations")

        # List available documents
        documents = doc_manager.list_processed_documents()

        if not documents:
            st.warning("No documents available. Please process a document first.")
        else:
            viz_type = st.selectbox("Visualization Type", ["Chunk Distribution", "Key Terms", "Document Comparison"])

            if viz_type == "Chunk Distribution":
                # Select document
                doc = st.selectbox("Select document", options=[f"{doc['id']} - {doc['filename']}" for doc in documents])
                doc_id = doc.split(" - ")[0]

                if st.button("Generate Visualization"):
                    with st.spinner("Generating visualization..."):
                        img_b64 = visualizer.plot_chunk_distribution(doc_id)
                        st.image(f"data:image/png;base64,{img_b64}")

            elif viz_type == "Key Terms":
                # Select document
                doc = st.selectbox("Select document", options=[f"{doc['id']} - {doc['filename']}" for doc in documents])
                doc_id = doc.split(" - ")[0]

                top_n = st.slider("Number of terms", min_value=5, max_value=50, value=20)

                if st.button("Generate Visualization"):
                    with st.spinner("Generating visualization..."):
                        img_b64 = visualizer.plot_key_terms(doc_id, top_n=top_n)
                        st.image(f"data:image/png;base64,{img_b64}")

            elif viz_type == "Document Comparison":
                # Select documents
                selected_docs = st.multiselect("Select documents to compare", options=[f"{doc['id']} - {doc['filename']}" for doc in documents])

                if len(selected_docs) < 2:
                    st.warning("Please select at least two documents.")
                else:
                    doc_ids = [doc.split(" - ")[0] for doc in selected_docs]
                    metric = st.selectbox("Comparison Metric", ["chunk_count", "token_count", "avg_chunk_size"])

                    if st.button("Generate Visualization"):
                        with st.spinner("Generating visualization..."):
                            img_b64 = visualizer.plot_document_comparison(doc_ids, metric=metric)
                            st.image(f"data:image/png;base64,{img_b64}")

    elif selected_tab == "Advanced Search":
        st.header("Advanced Search")

        search_type = st.selectbox("Search Type", ["Metadata Search", "Content Search"])

        if search_type == "Metadata Search":
            st.subheader("Search by Metadata")

            # Metadata filters
            col1, col2 = st.columns(2)

            with col1:
                doc_type = st.selectbox("Document Type", ["", "discharge_summary", "prescription", "lab_report", "clinical_note"])
                date_from = st.date_input("Date From", value=None)
                date_to = st.date_input("Date To", value=None)

            with col2:
                min_tokens = st.number_input("Minimum Tokens", min_value=0, value=0)
                max_tokens = st.number_input("Maximum Tokens", min_value=0, value=0)

            if st.button("Search"):
                with st.spinner("Searching..."):
                    # Convert dates to string format
                    date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
                    date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None

                    filters = {
                        "type": doc_type,
                        "date_from": date_from_str,
                        "date_to": date_to_str,
                        "min_tokens": min_tokens if min_tokens > 0 else None,
                        "max_tokens": max_tokens if max_tokens > 0 else None
                    }

                    results = advanced_search.search_by_metadata(filters)

                    if results:
                        st.success(f"Found {len(results)} documents")
                        df = pd.DataFrame(results)
                        st.dataframe(df)
                    else:
                        st.warning("No documents found matching the criteria")

        elif search_type == "Content Search":
            st.subheader("Search Document Content")

            query = st.text_input("Search Query")
            case_sensitive = st.checkbox("Case Sensitive")

            if query and st.button("Search"):
                with st.spinner("Searching document content..."):
                    results = advanced_search.search_by_content(query, case_sensitive=case_sensitive)

                    if results:
                        st.success(f"Found {len(results)} documents with matches")

                        for doc in results:
                            with st.expander(f"{doc['filename']} ({doc['match_count']} matches)"):
                                for i, match in enumerate(doc['matches']):
                                    st.markdown(f"**Match {i+1} (Page {match['page']+1}):**")
                                    st.markdown(match['context'])
                    else:
                        st.warning("No matches found")

    elif selected_tab == "Export/Import":
        st.header("Export/Import Documents")

        tab1, tab2 = st.tabs(["Export", "Import"])

        with tab1:
            st.subheader("Export Document")

            # List available documents
            documents = doc_manager.list_processed_documents()

            if not documents:
                st.warning("No documents available. Please process a document first.")
            else:
                # Select document to export
                doc = st.selectbox("Select document to export", options=[f"{doc['id']} - {doc['filename']}" for doc in documents])
                doc_id = doc.split(" - ")[0]

                include_raw = st.checkbox("Include raw document", value=True)

                if st.button("Export Document"):
                    with st.spinner("Exporting document..."):
                        try:
                            zip_path = export_import.export_document(doc_id, include_raw=include_raw)

                            # Provide download link
                            with open(zip_path, "rb") as f:
                                zip_bytes = f.read()

                            b64 = base64.b64encode(zip_bytes).decode()
                            href = f'<a href="data:application/zip;base64,{b64}" download="{zip_path}">Download Export</a>'
                            st.markdown(href, unsafe_allow_html=True)

                            # Clean up
                            os.remove(zip_path)
                        except Exception as e:
                            st.error(f"Error exporting document: {e}")

        with tab2:
            st.subheader("Import Document")

            uploaded_file = st.file_uploader("Upload export file", type=["zip"])

            if uploaded_file:
                # Save the uploaded file
                zip_path = os.path.join("data/temp", uploaded_file.name)
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)

                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.button("Import Document"):
                    with st.spinner("Importing document..."):
                        try:
                            doc_id = export_import.import_document(zip_path)
                            st.success(f"Document imported successfully! ID: {doc_id}")

                            # Clean up
                            os.remove(zip_path)
                        except Exception as e:
                            st.error(f"Error importing document: {e}")

    elif selected_tab == "PDF Annotation":
        st.header("PDF Annotation")

        # List available PDF documents
        documents = doc_manager.list_processed_documents()
        pdf_docs = [doc for doc in documents if doc["filename"].lower().endswith(".pdf")]

        if not pdf_docs:
            st.warning("No PDF documents available. Please process a PDF document first.")
        else:
            # Select document
            doc = st.selectbox("Select PDF document", options=[f"{doc['id']} - {doc['filename']}" for doc in pdf_docs])
            doc_id = doc.split(" - ")[0]

            annotation_type = st.selectbox("Annotation Type", ["Highlight Text", "Add Comment"])

            if annotation_type == "Highlight Text":
                search_text = st.text_input("Text to highlight")
                color = st.selectbox("Highlight Color", ["yellow", "red", "blue", "green"])

                if search_text and st.button("Highlight Text"):
                    with st.spinner("Highlighting text..."):
                        try:
                            pdf_b64 = pdf_annotator.highlight_text(doc_id, search_text, color=color)

                            if pdf_b64:
                                st.success("Text highlighted successfully!")

                                # Display PDF
                                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="700" height="1000" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)

                                # Provide download link
                                href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="highlighted_{doc_id}.pdf">Download Highlighted PDF</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Failed to highlight text. Text may not be found in the document.")
                        except Exception as e:
                            st.error(f"Error highlighting text: {e}")

            elif annotation_type == "Add Comment":
                page_num = st.number_input("Page Number", min_value=1, value=1) - 1  # Convert to 0-based
                comment_text = st.text_area("Comment Text")

                col1, col2 = st.columns(2)
                with col1:
                    x_pos = st.slider("X Position", min_value=50, max_value=500, value=100)
                with col2:
                    y_pos = st.slider("Y Position", min_value=50, max_value=700, value=100)

                if comment_text and st.button("Add Comment"):
                    with st.spinner("Adding comment..."):
                        try:
                            pdf_b64 = pdf_annotator.add_comment(doc_id, page_num, comment_text, x=x_pos, y=y_pos)

                            if pdf_b64:
                                st.success("Comment added successfully!")

                                # Display PDF
                                pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="700" height="1000" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)

                                # Provide download link
                                href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="annotated_{doc_id}.pdf">Download Annotated PDF</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Failed to add comment.")
                        except Exception as e:
                            st.error(f"Error adding comment: {e}")

    elif selected_tab == "Document Management":
        st.header("Document Management")

        # List all documents
        documents = doc_manager.list_processed_documents()

        if not documents:
            st.warning("No documents available. Please process a document first.")
        else:
            st.subheader("Document List")

            # Create a dataframe for display
            df = pd.DataFrame([{
                "ID": doc["id"],
                "Filename": doc["filename"],
                "Type": doc["type"],
                "Date Processed": doc["date_processed"],
                "Tokens": doc["token_count"],
                "Chunks": doc["chunk_count"]
            } for doc in documents])

            st.dataframe(df)

            # Document actions
            st.subheader("Document Actions")

            # Select document
            doc = st.selectbox("Select document", options=[f"{doc['id']} - {doc['filename']}" for doc in documents])
            doc_id = doc.split(" - ")[0]

            action = st.selectbox("Action", ["View Details", "Delete Document"])

            if action == "View Details":
                if st.button("View Document Details"):
                    with st.spinner("Loading document details..."):
                        doc_details = doc_manager.get_document_details(doc_id)

                        if doc_details:
                            st.json(doc_details)
                        else:
                            st.error("Failed to load document details.")

            elif action == "Delete Document":
                if st.button("Delete Document"):
                    confirm = st.checkbox("I understand this action cannot be undone")

                    if confirm:
                        with st.spinner("Deleting document..."):
                            success = doc_manager.delete_document(doc_id)

                            if success:
                                st.success(f"Document {doc_id} deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")
