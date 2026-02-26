import streamlit as st
from rag_pipeline import load_documents, split_text, create_vector_store, answer_question

st.title("📚 PDF RAG Chatbot (Groq + FAISS)")

@st.cache_resource
def setup_rag():
    docs = load_documents("data/sample.pdf")
    chunks = split_text(docs)
    vs = create_vector_store(chunks)
    return vs

vectorstore = setup_rag()

question = st.text_input("Ask a question from your PDF:")

if question:
    answer, sources = answer_question(vectorstore, question, k=4)
    st.subheader("🤖 Answer")
    st.write(answer)

    with st.expander("📌 Sources used"):
        for i, d in enumerate(sources, start=1):
            st.markdown(f"**Chunk {i}:**")
            st.write(d.page_content[:800] + "...")