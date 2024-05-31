import streamlit as st
import os
from dotenv import load_dotenv

import google.generativeai as genai
from util import make_prompt
from database import DocumentDatabase
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the database
db = DocumentDatabase(persist_directory="./chroma_db")
db.load_database()

# Streamlit app
st.title("Question and Answer with Gemini AI")

# Input for user query
query = st.text_input("Enter your question:")

if query:
    # Perform similarity search in the database
    similar_docs = db.similarity_search(query)
    if similar_docs:
        # Sort documents by relevance (assuming they are already sorted by the db.similarity_search function)
        sorted_docs = sorted(similar_docs, key=lambda doc: doc.metadata.get('relevance_score', 0), reverse=True)
        
        # Display relevant pages in the left panel
        with st.sidebar:
            st.header("Relevant Pages")
            for doc in sorted_docs:
                page_number = doc.metadata.get('page', 'Unknown')
                st.write(f"Page {page_number}: {doc.metadata.get('title', 'No Title')}")

        # Use the most relevant document to generate the prompt for Gemini AI
        context = sorted_docs[0].page_content
        
        # Generate the prompt for Gemini AI
        prompt = make_prompt(query, context)
        
        # Call Gemini API to get the answer
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        answer = model.generate_content(prompt)
        print(answer.candidates[0].content.parts)
        # Display the result
        st.markdown("### Answer")
        st.markdown(answer.text)
    else:
        st.markdown("No relevant documents found.")