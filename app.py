import os
import tempfile
import hashlib
import streamlit as st
from rag_pipeline import load_documents, split_text, create_vector_store, answer_question

st.set_page_config(page_title="PDF Chatbot", page_icon="📒", layout="wide")
st.title("Welcome to PDF chatbot")

# ----------------------------
# Helpers
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_resource(show_spinner=False)
def build_vectorstore_from_pdf_bytes(pdf_bytes: bytes) :
    """
    Build vectorstore from PDF bytes.
    Cached by Streamlit so same PDF won't re-index repeatedly.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    try:
        docs = load_documents(pdf_path)
        chunks = split_text(docs)
        vs = create_vector_store(chunks)
        return vs
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass

def reset_chat():
    st.session_state.messages = []

# ----------------------------
# Session State init
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "sources": optional}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ----------------------------
# Sidebar: Upload + Controls
# ----------------------------
with st.sidebar:
    st.header("📂 Upload PDF")

    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Clear chat button (requested)
    if st.button("🔄 Clear chat", use_container_width=True):
        reset_chat()

    st.divider()

    # Optional (Professional Touch): status + file name
    if uploaded_pdf is None and st.session_state.vectorstore is None:
        st.info("Please upload a PDF to start chatting.")
    else:
        # Show file name (requested)
        current_name = uploaded_pdf.name if uploaded_pdf is not None else st.session_state.pdf_name
        if current_name:
            st.caption("Current file:")
            st.write(f"📄 **{current_name}**")

# ----------------------------
# Index PDF if uploaded (and new)
# ----------------------------
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    new_hash = sha256_bytes(pdf_bytes)

    # Re-index only if PDF changed
    if st.session_state.pdf_hash != new_hash:
        st.session_state.pdf_hash = new_hash
        st.session_state.pdf_name = uploaded_pdf.name
        reset_chat()  # optional: reset chat when new PDF uploaded

        with st.spinner("Indexing your PDF..."):
            st.session_state.vectorstore = build_vectorstore_from_pdf_bytes(pdf_bytes)

        st.success("✅ PDF indexed! You can start asking questions.")
else:
    # If no upload yet, keep whatever was previously indexed in the same session.
    # If you want to force upload every time, uncomment the next two lines:
    # st.session_state.vectorstore = None
    # st.session_state.pdf_hash = None
    pass

vectorstore = st.session_state.vectorstore

# ----------------------------
# ChatGPT-style Chat UI
# ----------------------------
# Show a nice hint in main area
if vectorstore is None:
    st.warning("⚠️ Kindly upload a PDF file from the left sidebar to start.")
else:
    st.caption(f"Chatting with: 📄 {st.session_state.pdf_name}")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Sources"):
                for i, d in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(d.page_content[:900] + ("..." if len(d.page_content) > 900 else ""))

# Input box (ChatGPT style)
user_text = st.chat_input("Ask a question about your PDF...")

if user_text:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    # If no PDF uploaded/indexed yet -> assistant tells user to upload
    if vectorstore is None:
        assistant_text = "⚠️ Kindly upload a PDF file first (use the left sidebar)."
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.rerun()

    # Otherwise answer using RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Keep your existing function signature.
            # If your answer_question supports k, you can pass k=4.
            try:
                answer, sources = answer_question(vectorstore, user_text, k=4)
            except TypeError:
                answer, sources = answer_question(vectorstore, user_text)

        st.write(answer)

        with st.expander("📌 Sources"):
            for i, d in enumerate(sources, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(d.page_content[:900] + ("..." if len(d.page_content) > 900 else ""))

    # Store assistant message + sources in history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    st.rerun()