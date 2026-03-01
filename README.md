📚 RAG-Based PDF Question Answering Chatbot

A live, deployed Retrieval-Augmented Generation (RAG) application that allows users to ask questions from a PDF document and receive context-aware answers using LLMs.

🚀 Live Demo

🔗 Live App:
https://rag-pdf-chatbot-wdxgjqd6imdwgreusibgdj.streamlit.app/

💻 GitHub Repository:
https://github.com/venkatraja1002-sudo/rag-pdf-chatbot

🧠 Project Overview

This project implements a complete RAG pipeline:
Load PDF document
Split into text chunks
Generate embeddings
Store embeddings in FAISS vector database
Retrieve relevant chunks
Send context + question to Groq LLaMA 3.1 model
Display answer via Streamlit UI

🏗 Architecture
User Question
      ↓
Streamlit UI
      ↓
Retriever (FAISS)
      ↓
Relevant Chunks
      ↓
Groq LLM (LLaMA 3.1)
      ↓
Final Answer


🛠 Tech Stack
Python
Streamlit
LangChain
FAISS (Vector Database)
Sentence Transformers
Groq LLaMA 3.1
Git & GitHub
Streamlit Community Cloud (Deployment)

📂 Project Structure
rag-pdf-chatbot/
│
├── app.py                 # Streamlit frontend
├── rag_pipeline.py        # RAG backend logic
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore local files
│
└── data/
    └── sample.pdf         # Document used for Q&A

⚙️ Installation (Run Locally)
1️⃣ Clone Repository
git clone https://github.com/venkatraja1002-sudo/rag-pdf-chatbot.git
cd rag-pdf-chatbot
2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate
3️⃣ Install Requirements
pip install -r requirements.txt
4️⃣ Add GROQ API Key
Create a .env file:
GROQ_API_KEY=your_api_key_here
5️⃣ Run App
streamlit run app.py

🌍 Deployment

This app is deployed using Streamlit Community Cloud.

