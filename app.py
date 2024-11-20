# import os
# import json
# from dotenv import load_dotenv

# from google.generativeai import configure, GenerativeModel
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from sentence_transformers import CrossEncoder
# from util import make_prompt
# from database import DocumentDatabase

# from typing import List
# from pathlib import Path
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from langchain.schema import Document
# from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
# from langchain.callbacks.base import BaseCallbackHandler

# import chainlit as cl

# # Load environment variables
# load_dotenv()

# # Configure Google Gemini API
# configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def load_model():
#     return CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")

# def load_database():
#     database = DocumentDatabase(persist_directory="./chroma_db_claude")
#     return database.return_db()

# def load_documents(pdf_path):
#     loader = PyPDFLoader(pdf_path)
#     return loader.load_and_split()

# db = load_database()
# model = load_model()

# def load_json_file(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# file_path = r'D:\Project\RAG-Core-Service\data\extracted_clauses.json'
# data_dict = load_json_file(file_path)

# document_objects = []
# for entry in data_dict:
#     for clause in entry['clauses']:
#         metadata = {
#             'section': entry['section'],
#             'title': entry['title'].strip(),
#             'page_num': entry['page_num']
#         }
#         document = Document(page_content=entry['title'] + ' ' + clause, metadata=metadata)
#         document_objects.append(document)

# bm25_retriever = BM25Retriever.from_documents(document_objects)
# retriever_Chroma = db.as_retriever(search_kwargs={"k": 100})

# # Define prompt template
# template = """When you respond to a question, please use the language from the reference passage provided below.
#                            Responses should be in complete sentences and detail. As the audience has technical expertise, keep the answers concise and clear; 
#                            If the passage is irrelevant to the question, PASSAGE:

# {context}

# Question: {question}
# """
# # """When you respond to a question, please use the language from the reference passage provided below.
# #                            Responses should be in complete sentences and detail. As the audience has technical expertise, keep the answers concise and clear; 
# #                            If the passage is irrelevant to the question, you must be respond "I'm sorry.I dont know".
# #   QUESTION: '{query}'
# #   PASSAGE: '{relevant_passage}'
# #     ANSWER:
# #   """
# prompt = ChatPromptTemplate.from_template(template)

# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

# # Ensemble retriever combining BM25 and Chroma
# ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_Chroma], weights=[0.5, 0.5])

# # Define runnable
# runnable = (
#     {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | StrOutputParser()
# )

# @cl.on_chat_start
# async def on_chat_start():
#     cl.user_session.set("runnable", runnable)
#     await cl.Message(content="Question and Answer about BUILDING CODE").send()

# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("runnable")  # type: Runnable
#     msg = cl.Message(content="")

#     class PostMessageHandler(BaseCallbackHandler):
#         """
#         Callback handler for handling the retriever and LLM processes.
#         Used to post the sources of the retrieved documents as a Chainlit element.
#         """

#         def __init__(self, msg: cl.Message):
#             super().__init__()
#             self.msg = msg
#             self.sources = set()  # To store unique pairs

#         def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
#             for d in documents:
#                 source_page_pair = (d.metadata['section'], d.metadata['page_num'])
#                 self.sources.add(source_page_pair)  # Add unique pairs to the set

#         def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
#             if len(self.sources):
#                 sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
#                 self.msg.elements.append(
#                     cl.Text(name="Sources", content=sources_text, display="inline")
#                 )

#     async for chunk in runnable.astream(
#         message.content,
#         config=RunnableConfig(callbacks=[
#             cl.LangchainCallbackHandler(),
#             PostMessageHandler(msg)
#         ]),
#     ):
#         if isinstance(chunk, dict):
#             chunk = json.dumps(chunk)  # Ensure the chunk is a string
#         elif not isinstance(chunk, str):
#             chunk = str(chunk)  # Convert the chunk to a string
#         await msg.stream_token(chunk)

#     await msg.send()




from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

import chainlit as cl
os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest")


@cl.on_chat_start
async def on_chat_start():
    model = llm #hatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = LLMChain(llm=model, prompt=prompt, output_parser=StrOutputParser())

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: LLMChain

    res = await chain.arun(
        question=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=res).send()