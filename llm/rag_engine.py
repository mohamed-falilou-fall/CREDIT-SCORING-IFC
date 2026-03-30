# -*- coding: utf-8 -*-

# PDF loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + vector DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM (CORRIGÉ)
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# ================================
# BUILD VECTORSTORE
# ================================
def build_vectorstore(path="report/"):

    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("vectorstore")

    return db


# ================================
# LOAD VECTORSTORE
# ================================
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local("vectorstore", embeddings)


# ================================
# LOAD LLM
# ================================
def load_llm():

    pipe = pipeline(
        "text2text-generation",  # 🔥 IMPORTANT pour flan-t5
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2
    )

    return HuggingFacePipeline(pipeline=pipe)


# ================================
# QUERY
# ================================
def ask_ifc_llm(question, db, llm):

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an IFC expert.

    Context:
    {context}

    Question:
    {question}

    Answer clearly and professionally.
    """

    return llm(prompt)