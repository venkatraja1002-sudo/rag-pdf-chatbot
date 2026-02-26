import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_documents(pdf_path: str = "data/sample.pdf"):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


def create_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )


def answer_question(vectorstore, question: str, k: int = 4):
    # 1) Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # 2) Build prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the document."

Context:
{context}

Question: {question}
Answer:"""

    # 3) Call LLM
    llm = create_llm()
    result = llm.invoke(prompt)
    return result.content, docs