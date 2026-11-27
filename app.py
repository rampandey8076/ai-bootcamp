import hashlib
from pathlib import Path

import streamlit as st

from rag_core import (
    create_client,
    load_sample_documents,
    load_uploaded_documents,
    build_index,
    ask_question,
    FALLBACK
)


st.set_page_config(
    page_title="Vector & RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Vector Databases, Embeddings & RAG Assistant")

st.caption(
    "Ask questions about text embeddings, vector databases "
    "and Retrieval-Augmented Generation."
)

st.info(
    "This assistant answers using the selected documents. "
    "It does not intentionally use outside knowledge."
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Your Gemini API key",
        type="password",
        help="Your key is used for this session and is not saved by this app."
    )

    st.caption("Get a key from Google AI Studio.")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    st.divider()

    st.subheader("Suggested questions")

    suggestions = [
        "What are text embeddings?",
        "How does a vector database work?",
        "What is the purpose of chunking in RAG?",
        "Explain the complete RAG pipeline."
    ]

    for question in suggestions:
        if st.button(question, use_container_width=True):
            st.session_state.pending_question = question

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "index_key" not in st.session_state:
    st.session_state.index_key = None


# Select uploaded documents, or use sample documents.
try:
    if uploaded_files:
        documents = load_uploaded_documents(uploaded_files)
        document_label = "Your uploaded documents"
    else:
        documents = load_sample_documents("knowledge_base")
        document_label = "Sample knowledge base"

    if not documents:
        st.warning(
            "No readable documents found. Add sample documents "
            "to knowledge_base or upload your own."
        )
        st.stop()

except Exception as error:
    st.error(f"Document loading failed: {error}")
    st.stop()


# Build a fingerprint so the index is rebuilt only when
# the document selection or API key changes.
fingerprint_source = api_key + "|" + "|".join(
    f"{doc['source']}:{doc['page']}:{doc['text']}"
    for doc in documents
)

fingerprint = hashlib.sha256(
    fingerprint_source.encode("utf-8")
).hexdigest()


st.caption(
    f"Using: {document_label} · "
    f"{len(documents)} document page(s) loaded"
)


if not api_key:
    st.warning("Enter your Gemini API key in the sidebar to start chatting.")
    st.stop()


if st.session_state.index_key != fingerprint:
    with st.spinner("Creating document embeddings and building the index..."):
        try:
            client = create_client(api_key)
            st.session_state.index = build_index(client, documents)
            st.session_state.index_key = fingerprint
            st.session_state.messages = []

        except Exception as error:
            st.session_state.index = None
            st.session_state.index_key = None
            st.error(f"Could not index the documents: {error}")
            st.stop()


# Display previous conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"**{source['source']}** — "
                        f"Page {source['page']} "
                        f"(similarity: {source['score']:.3f})"
                    )


# Handle a suggested question or normal chat input.
pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a question about your documents...")

if pending:
    question = pending

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            try:
                client = create_client(api_key)

                answer, sources = ask_question(
                    client,
                    question,
                    st.session_state.index
                )

                st.markdown(answer)

                if sources:
                    with st.expander("Sources", expanded=True):
                        for source in sources:
                            st.write(
                                f"**{source['source']}** — "
                                f"Page {source['page']} "
                                f"(similarity: {source['score']:.3f})"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")
