# Vector Database and RAG Knowledge Assistant

## Project Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using a knowledge base about:

- Text embeddings
- Vector databases
- Semantic search
- Retrieval-Augmented Generation

The application retrieves relevant document chunks using Gemini embeddings and cosine similarity, then generates grounded answers using Gemini.

## Features

- Gemini-powered text embeddings
- Document chunking with overlap
- Vector similarity search
- Cosine similarity retrieval
- Grounded question answering
- Source file and page references
- Fallback for questions outside the knowledge base
- PDF, TXT and Markdown uploads
- Visitor-provided Gemini API key
- Suggested questions
- Conversation history

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- NumPy
- PyPDF

## Project Structure

```text
ai-bootcamp/
├── app.py
├── rag_core.py
├── requirements.txt
├── test_rag_core.py
├── tests.md
├── README.md
├── .gitignore
└── knowledge_base/
    ├── 01_text_embeddings.md
    ├── 02_vector_databases.md
    └── 03_rag_pipeline.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rampandey8076/ai-bootcamp.git
cd ai-bootcamp
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

Enter your Gemini API key in the sidebar to start asking questions.

Get an API key from:
https://aistudio.google.com/apikey

## How RAG Works

1. Load documents from the knowledge base or uploaded files.
2. Split documents into overlapping chunks.
3. Generate vector embeddings using Gemini.
4. Normalize vectors for cosine similarity.
5. Embed the user's question.
6. Retrieve the most relevant document chunks.
7. Generate an answer using the retrieved context.
8. Display source file and page references.

## Testing

Run automated tests:

```bash
python -m unittest -v test_rag_core
```

See `tests.md` for manual chatbot test cases.

## Extra Feature

Suggested questions are available in the sidebar to help visitors explore the knowledge base.

## Privacy

The application processes uploaded documents in memory and does not intentionally save them to disk. The Gemini API key is supplied by the visitor and used to communicate with Google's Gemini API.

## Author

Ram Pandey

GitHub: https://github.com/rampandey8076
