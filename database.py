from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_chroma import Chroma
import os

class DocumentDatabase:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_path = os.path.join(self.persist_directory, "chroma.db")  # Ensure correct path to your database file
        self.pages = []
        # Delete the database file if it exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)

    def add_document_from_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        self.pages = loader.load_and_split()
        self.db.add_documents(self.pages)
        
    def save_database(self):
        self.db.save()

    def load_database(self):
        self.db = Chroma(persist_directory=self.persist_directory , embedding_function=self.embedding_function)

    def similarity_search(self, query, k=10):
        return self.db.similarity_search(query,k)
    
    def return_db(self):
        return self.db

if __name__ == "__main__":
    db = DocumentDatabase(persist_directory="./chroma_db")

    # Add documents from PDF
    pdf_path = "book/ullman_the_complete_book.pdf"
    db.add_document_from_pdf(pdf_path)

    # Save the database
    # db.save_database()

    # Load the saved database
    db.load_database()

    # Perform similarity search
    query = "Modifying Relation Schemas"
    similar_docs = db.similarity_search(query)
    for doc in similar_docs:
        print(doc.page_content)