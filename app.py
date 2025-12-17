import streamlit as st

from retriever import RAGRetriever
from llm import RAGAnswerGenerator

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-PDF RAG Demo",
    page_icon="📄",
    layout="centered",
)

# ---------------- APP TITLE ----------------
st.title("📄 Multi-Document Q&A (RAG Demo)")
st.write(
    "Ask questions over multiple PDF documents using "
    "Retrieval-Augmented Generation (RAG)."
)

st.caption("Demo project for LLMs, embeddings, and vector search.")

# ---------------- INIT RAG COMPONENTS ----------------
@st.cache_resource
def load_rag_components():
    retriever = RAGRetriever()
    llm = RAGAnswerGenerator()
    return retriever, llm

retriever, llm = load_rag_components()

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- USER INPUT ----------------
question = st.text_input(
    "Ask a question about the documents:",
    placeholder="e.g. Can international students apply for RPL?"
)

# ---------------- PROCESS QUESTION ----------------
if st.button("Ask") and question:
    with st.spinner("Searching documents and generating answer..."):
        docs = retriever.retrieve(question)
        answer = llm.answer(question, docs)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer,
            "sources": docs,
        }
    )

# ---------------- DISPLAY CHAT HISTORY ----------------
for item in reversed(st.session_state.chat_history):
    st.markdown("### ❓ Question")
    st.write(item["question"])

    st.markdown("### ✅ Answer")
    st.write(item["answer"])

    with st.expander("📚 Sources used"):
        for i, d in enumerate(item["sources"], 1):
            meta = d.metadata
            st.markdown(
                f"""
**Source {i}**
- Institution: `{meta.get('institution')}`
- Type: `{meta.get('source_type')}`
- File: `{meta.get('source_file')}`
"""
            )
            st.write(d.page_content[:500] + "...")
