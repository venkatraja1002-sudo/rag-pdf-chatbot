import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

# ----------------------------
# Env / Secrets handling
# ----------------------------
# Load local .env for LOCAL development (Streamlit Cloud will use st.secrets)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


def get_groq_api_key() -> str:
    """
    Streamlit Cloud: put GROQ_API_KEY into Secrets.
    Local: .env or environment variables.
    Priority:
      1) st.secrets["GROQ_API_KEY"]
      2) os.getenv("GROQ_API_KEY")
    """
    # Streamlit Cloud secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        # If secrets not available (some local runs), ignore
        pass

    # Local env / .env
    return os.getenv("GROQ_API_KEY", "")


# ----------------------------
# PDF loading + chunking
# ----------------------------
def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


# ----------------------------
# Embeddings + Vector store
# ----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


def create_vector_store(docs):
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


# ----------------------------
# LLM
# ----------------------------
def create_llm():
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "On Streamlit Cloud: set it in App → Settings → Secrets. "
            "Locally: set it in .env or environment variables."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )


# ----------------------------
# RAG QA
# ----------------------------
def answer_question(vectorstore, question: str, k: int = 4):
    # 1) Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # 2) Prompt
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