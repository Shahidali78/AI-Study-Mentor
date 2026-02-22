import os
import shutil
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Prevent transformers from importing TensorFlow/Keras in mixed ML environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_rag"


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Tuple[object, str]:
    """Return (embedding_instance, backend_name)."""
    try:
        return (
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            "local",
        )
    except Exception:
        if os.getenv("OPENAI_API_KEY"):
            return (OpenAIEmbeddings(model="text-embedding-3-small"), "openai")
        raise RuntimeError(
            "Failed to initialize embeddings. "
            "Your environment likely has a TensorFlow/Keras compatibility conflict. "
            "Fix option: `pip install tf-keras`."
        )


def get_embedding_instance():
    embeddings, _ = get_embeddings()
    return embeddings


def get_embedding_backend() -> str:
    _, backend = get_embeddings()
    return backend


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = os.path.isdir(DB_DIR)
    if "embed_notice_shown" not in st.session_state:
        st.session_state.embed_notice_shown = False


def render_embedding_notice():
    backend = get_embedding_backend()
    if backend == "openai" and not st.session_state.embed_notice_shown:
        st.info(
            "Local embedding model was unavailable; using OpenAI embeddings (`text-embedding-3-small`)."
        )
        st.session_state.embed_notice_shown = True


def load_documents(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    os.makedirs("tmp_uploads", exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join("tmp_uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path)
        file_docs = loader.load()

        for d in file_docs:
            d.metadata["source_file"] = file.name
            if "page" in d.metadata:
                d.metadata["page_number"] = int(d.metadata["page"]) + 1

        docs.extend(file_docs)

    return docs


def build_vector_store(docs: List[Document]) -> Tuple[Chroma, int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_instance()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb, len(chunks)


def get_retriever() -> Chroma:
    embeddings = get_embedding_instance()
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb


def format_citations(docs: List[Document]) -> str:
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown.pdf")
        pg = d.metadata.get("page_number", "?")
        key = (src, pg)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{src}` (page {pg})")
    return "\n".join(lines) if lines else "- No citations"


def extractive_answer(docs: List[Document]) -> str:
    snippets = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "unknown.pdf")
        pg = d.metadata.get("page_number", "?")
        text = d.page_content.strip().replace("\n", " ")[:600]
        snippets.append(f"[{i}] {text}\n(source: {src}, page {pg})")

    joined = "\n\n".join(snippets)
    return (
        "OpenAI key not found, so this is extractive mode. "
        "Top relevant chunks are shown below:\n\n" + joined
    )


def llm_answer(question: str, docs: List[Document], model_name: str) -> str:
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "unknown.pdf")
        pg = d.metadata.get("page_number", "?")
        context_blocks.append(
            f"[{i}] Source: {src}, Page: {pg}\nContent: {d.page_content.strip()}"
        )

    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_template(
        """
You are a precise assistant answering questions from PDF context only.
Rules:
1) Use only provided context.
2) If the answer is not in context, say: "I do not know from the provided PDFs."
3) Keep answer concise.
4) End every factual sentence with citation(s) in this format: [filename p.X]
5) Never invent page numbers or sources.

Question:
{question}

Context:
{context}
"""
    )

    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = prompt | llm
    result = chain.invoke({"question": question, "context": context})
    return result.content


def query_documents(question: str, k: int, model_name: str):
    with st.spinner("Retrieving relevant chunks..."):
        vectordb = get_retriever()
        docs = vectordb.similarity_search(question, k=k)

    if not docs:
        return None, None

    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    with st.spinner("Generating answer..."):
        if has_openai_key:
            answer = llm_answer(question, docs, model_name)
        else:
            answer = extractive_answer(docs)

    return answer, docs


def main():
    load_dotenv()
    ensure_session_state()

    st.set_page_config(page_title="PDF RAG Assistant", page_icon="PDF", layout="wide")
    st.title("PDF RAG Assistant")
    st.caption("Upload PDFs, build index, and ask grounded questions with citations.")

    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=4)
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.success("Chat history cleared.")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Build / Rebuild Index", type="primary"):
            if not uploaded_files:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Reading PDFs and building vector index..."):
                    docs = load_documents(uploaded_files)
                    _, chunk_count = build_vector_store(docs)
                    st.session_state.index_ready = True
                st.success(f"Index built successfully. Total chunks: {chunk_count}")

    with col2:
        if st.button("Clear Index"):
            if os.path.isdir(DB_DIR):
                shutil.rmtree(DB_DIR)
                st.session_state.index_ready = False
                st.success("Index cleared.")
            else:
                st.info("No index found.")

    st.divider()

    if st.session_state.messages:
        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.caption("Citations")
                    st.markdown(msg["citations"])

    question = st.chat_input("Ask a question about your PDFs")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        render_embedding_notice()

        if not st.session_state.index_ready or not os.path.isdir(DB_DIR):
            st.error("No index found. Build the index first.")
            return

        answer, docs = query_documents(question, k, model_name)
        if not docs:
            st.warning("No relevant content found.")
            return

        citations_text = format_citations(docs)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations_text}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption("Citations")
            st.markdown(citations_text)

        with st.expander("Retrieved Chunks"):
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source_file", "unknown.pdf")
                pg = d.metadata.get("page_number", "?")
                st.markdown(f"**Chunk {i}** - `{src}` page `{pg}`")
                st.write(d.page_content[:1200])


if __name__ == "__main__":
    main()
