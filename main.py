import os
import streamlit as st
from dotenv import load_dotenv

from google.generativeai import configure, GenerativeModel
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from util import make_prompt
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
    database = DocumentDatabase(persist_directory="./chroma_db_claude")
    return database.return_db()

@st.cache_data
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

db = load_database()
model = load_model()

import json

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
# bm25_retriever = BM25Retriever.from_documents(pages)
retriever_Chroma = db.as_retriever(search_kwargs={"k": 100})
file_path = r'/mnt/sda1/projects/Nam_exp/RAG_for_book/extracted_clauses.json'

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
        document = Document(page_content=entry['title'] +' '+ clause, metadata=metadata)
        document_objects.append(document)
        
bm25_retriever = BM25Retriever.from_documents(document_objects)        
# Streamlit Interface Setup
st.title("Question and Answer with Gemini AI")
query = st.text_input("Enter your question:")
use_reranking = st.sidebar.checkbox("Enable Re-Ranking", value=True)

# Sidebar for dynamic weights adjustment
st.sidebar.header("Adjust Retriever Weights")
weight_bm25 = st.sidebar.slider("Weight for BM25 Retriever", 0.0, 1.0, 0.2, 0.1)
weights = [weight_bm25, 1.0 - weight_bm25]
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_Chroma], weights=weights)

if query:
    # Document retrieval
    docs = ensemble_retriever.get_relevant_documents(query)

    if docs:
        if use_reranking:
            # Rerank the retrieved documents
            # print(docs)
            documents = [doc.page_content for doc in docs]
            ranked_indices = model.rank(query, documents, return_documents=False, top_k=3)
            
            # metadata = [doc.metadata for doc in docs]
            
            print("ranked_indices",ranked_indices)
            indices =[ranked_indice['corpus_id'] for  ranked_indice in ranked_indices]
            ranked_docs = [docs[idx] for idx in indices]
            
            print(ranked_docs)
        else:
            # Use the top 3 documents without reranking
            ranked_docs = docs[:3]

        combined_text = " ".join([doc.page_content for doc in ranked_docs])

        prompt = make_prompt(query, combined_text)
        gen_model = GenerativeModel('gemini-1.0-pro-latest')
        answer = gen_model.generate_content(prompt)

        # Display the result
        st.markdown("### Answer")
        st.markdown(answer.candidates[0].content.parts[0])

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