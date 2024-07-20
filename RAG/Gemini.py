import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.client_options import ClientOptions
from PyPDF2 import PdfReader
import os

os.environ['GOOGLE_API_KEY']="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"
genai.configure(api_key="AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM")

def extract_data(file):
    data = ""
    reader = PdfReader(file)
    for page in reader.pages:
        data += page.extract_text() or ""
    return data


def text_spliter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    return text_splitter.split_text(data)


def embedding_google():
    return GoogleGenerativeAIEmbeddings(model='models/embedding-001', task_type="SEMANTIC_SIMILARITY")


def vector_db_faiss(chunks):
    vector_db = FAISS.from_texts(chunks, embedding_google())
    return vector_db


def save_vector_db():
    path = "D:/2. GP/FinalApp/Data/Data/"
    for count, file in enumerate(os.listdir(path)):
        data = extract_data(path + file)
        chunks = text_spliter(data)
        if count == 0:
            vector_db = vector_db_faiss(chunks)
        else:
            vector_db.merge_from(vector_db_faiss(chunks))

    vector_db.save_local("faiss_index")

def load_vector_db():
    return FAISS.load_local("faiss_index", embedding_google(), allow_dangerous_deserialization=True)