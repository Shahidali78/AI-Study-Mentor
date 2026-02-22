# PDF RAG Assistant

A simple Retrieval-Augmented Generation (RAG) app that lets you:
- Upload one or more PDF files
- Build a local vector index
- Ask questions about your PDFs
- See source citations (file + page)

## Features
- Chat-style Q&A over your uploaded PDFs
- Local vector store using Chroma (`./chroma_db`)
- Citation-aware answers with file + page references
- Embedding fallback support for environments where local model loading fails

## 1) Setup

```bash
cd pdf_rag_assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional (for LLM-generated answers):
1. Copy `.env.example` to `.env`
2. Add your OpenAI key in `.env`

```bash
copy .env.example .env
```

## 2) Run

```bash
streamlit run app.py
```

## Notes
- Embeddings run locally using `sentence-transformers`.
- Vector DB is persisted in `./chroma_db`.
- If `OPENAI_API_KEY` is missing, the app still works in extractive mode (returns top relevant chunks with citations).

## GitHub Safety Checklist (Before Push)
1. Rotate API keys if they were ever exposed in chat/screenshots/logs.
2. Confirm `.env` is ignored and not staged.
3. Commit `.env.example` only (never real secrets).
4. Ensure runtime folders are not staged: `chroma_db/`, `tmp_uploads/`, `.venv/`.
5. Run a final check:

```bash
git status
git add .
git status
```

You should not see `.env`, `chroma_db/`, `tmp_uploads/`, or `.venv/` in staged files.
