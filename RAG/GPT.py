import pinecone
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader

load_dotenv(find_dotenv(), override=True)


def chunk_data(data, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks


def load_document(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    return data


def embeddings(index_name, chunks):
    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    vector_store = LangChainPinecone.from_documents(chunks, embeddings, index_name=index_name)

    return vector_store


def create_vector_db():

    path = "D:/2. GP/FinalApp/Data/Data/"
    for count, file in enumerate(os.listdir(path)):
        data = load_document(path + file)
        chunks = chunk_data(data, 256)
        if count == 0:
            print("testing .... ")
            vector_db = embeddings("gp-iti", chunks)
        else:
            vector_db.add_documents(chunks)

create_vector_db()