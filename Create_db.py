import json
from langchain_core.documents.base import Document
from database import DocumentDatabase

def load_json_file(file_path):
    """Load and return JSON data from a given file path."""
    with open(file_path, 'r') as file:
        return json.load(file)

def find_depth(data):
    """Recursively find the maximum depth of the data structure."""
    if isinstance(data, dict):
        return 1 + max((find_depth(value) for value in data.values()), default=0)
    elif isinstance(data, list):
        return max((find_depth(item) for item in data), default=0)
    return 0

def process_and_add_to_db(file_path=r'D:\Project\RAG-Core-Service\data\extracted_clauses_OBC_2020.json', 
                          directory="./chroma_db_claude_NBC_2020"):
    """Process JSON data from file and add documents to the database."""
    data_dict = load_json_file(file_path)

    document_objects = []
    for entry in data_dict:
        for clause in entry['clauses']:
            metadata = {
                'section': entry['section'],
                'title': entry['title'].strip(),
                'page_num': entry['page_num']
            }
            document = Document(page_content=f"{entry['section']} {entry['title']} {clause}", metadata=metadata)
            document_objects.append(document)

    db = DocumentDatabase(persist_directory=directory)
    db.add_document(document_objects)

# You can call the function with custom file_path and directory if needed.
process_and_add_to_db()
