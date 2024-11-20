import os
import streamlit as st
from dotenv import load_dotenv

from google.generativeai import configure, GenerativeModel
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from utils import make_prompt,load_json_file
from database import DocumentDatabase
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Configure Google Gemini API
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use @st.cache_resource for loading resources like models
@st.cache_resource
def load_model():
    return CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

# Use @st.cache_data for data processes like loading documents
@st.cache_resource
def load_database():
    database = DocumentDatabase(persist_directory="./chroma_db_claude_NBC_2020")
    return database.return_db()

db = load_database()
model = load_model()

# bm25_retriever = BM25Retriever.from_documents(pages)
retriever_Chroma = db.as_retriever(search_kwargs={"k": 100})
file_path = r'data/extracted_clauses_OBC_2020.json'

data_dict = load_json_file(file_path)
from langchain_core.documents.base import Document
document_objects = []
for entry in data_dict:
    for clause in entry['clauses']:
        metadata = {
            'section': entry['section'],
            'title': entry['title'].strip(),
            'page_num': entry['page_num']
        }
        clause = [item for item in clause if item is not None]
        document = Document(page_content=entry['title'] +' '+ ' '.join(clause), metadata=metadata)
        document_objects.append(document)
        
bm25_retriever = BM25Retriever.from_documents(document_objects)        
# Streamlit Interface Setup
st.title("Question and Answer about BUILDING CODE")
query = st.text_input("Enter your question:")
use_reranking = st.sidebar.checkbox("Enable Re-Ranking", value=True)

# Sidebar for dynamic weights adjustment
st.sidebar.header("Adjust Retriever Weights")
weight_bm25 = st.sidebar.slider("Weight for BM25 Retriever", 0.0, 1.0, 0.4, 0.1)
weights = [weight_bm25, 1.0 - weight_bm25]
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_Chroma], weights=weights)

if query:
    # Document retrieval
    docs = ensemble_retriever.get_relevant_documents(query)

    if docs:
        if use_reranking:
            # Rerank the retrieved documents
            documents = [doc.page_content for doc in docs]
            ranked_indices = model.rank(query, documents, return_documents=False, top_k=5)

            
            indices =[ranked_indice['corpus_id'] for  ranked_indice in ranked_indices]
            ranked_docs = [docs[idx] for idx in indices]
            
        else:
            # Use the top 3 documents without reranking
            ranked_docs = docs[:5]

        combined_text =  " ".join([doc.page_content for doc in ranked_docs])
        print("combined_text",combined_text)
        prompt = make_prompt(query, combined_text)
        gen_model = GenerativeModel('gemini-1.0-pro-latest')
        answer = gen_model.generate_content(prompt)

        # Display the result
        st.markdown("### Answer")
        st.markdown(answer.candidates[0].content.parts[0].text)

        # Show the relevant pages in the sidebar
        with st.sidebar:
            st.header("Relevant Pages")
            for index, doc in enumerate(ranked_docs):
                doc_id = doc.metadata['section']  # Change made here
                doc_content = doc.page_content
                st.write(f"Document ID: {doc_id}")
                # Pass a unique key for each text_area
                st.text_area("Content Preview", doc_content, height=100, key=f"doc_{index}")
    else:
        st.markdown("No relevant documents found.")