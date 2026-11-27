import io
import re
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader


EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 35
TOP_K = 5
MIN_SIMILARITY_SCORE = 0.30
BATCH_SIZE = 50

FALLBACK = (
    "I couldn't find this information in the provided documents. "
    "Please ask a question related to the knowledge base."
)




def call_with_retry(operation, request, max_attempts=3):
    """Retry temporary Gemini API failures with exponential backoff."""
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
        raise ValueError("max_attempts must be a positive integer.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")

    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_attempts):
        try:
            return request()

        except Exception as error:
            # Prefer structured HTTP status information.
            status_code = None

            for attribute in ("status_code", "code", "status"):
                value = getattr(error, attribute, None)

                if value is not None:
                    try:
                        status_code = int(value)
                        break
                    except (TypeError, ValueError):
                        continue

            # Retain support for exceptions exposing only an error message.
            if status_code is None:
                match = re.search(
                    r"\b(?:HTTP\s*)?(429|500|502|503|504)\b",
                    str(error),
                    re.IGNORECASE
                )
                if match:
                    status_code = int(match.group(1))

            retryable = status_code in retryable_statuses

            if not retryable:
                raise

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Gemini {operation} failed after "
                    f"{max_attempts} attempts. "
                    f"Please try again later. Details: {error}"
                ) from error

            delay = 2 ** attempt

            print(
                f"[RAG] Gemini {operation} temporarily unavailable. "
                f"Retrying in {delay}s..."
            )

            time.sleep(delay)


def create_client(api_key):
    """Create a Gemini client using the visitor's API key."""
    if not api_key or not api_key.strip():
        raise ValueError("Please enter your Gemini API key.")

    return genai.Client(api_key=api_key.strip())


def extract_pdf_pages(file_bytes):
    """Extract selectable text from every PDF page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def extract_text_pages(file_bytes):
    """Extract text from TXT or Markdown documents."""
    text = file_bytes.decode("utf-8", errors="ignore")

    # A form-feed character can represent a page break.
    sections = text.split("\f")

    return [
        {"page": i + 1, "text": section}
        for i, section in enumerate(sections)
        if section.strip()
    ]


def load_sample_documents(folder="knowledge_base"):
    """Load the repository's sample Markdown and text documents."""
    documents = []

    for path in sorted(Path(folder).glob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            pages = extract_pdf_pages(path.read_bytes())
        elif suffix in [".txt", ".md"]:
            pages = extract_text_pages(path.read_bytes())
        else:
            continue

        for item in pages:
            documents.append({
                "source": path.name,
                "page": item["page"],
                "text": item["text"]
            })

    return documents


def load_uploaded_documents(uploaded_files):
    """Read uploaded files in memory. Nothing is written to disk."""
    documents = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            pages = extract_pdf_pages(file_bytes)
        elif suffix in [".txt", ".md"]:
            pages = extract_text_pages(file_bytes)
        else:
            continue

        for item in pages:
            documents.append({
                "source": filename,
                "page": item["page"],
                "text": item["text"]
            })

    return documents


def split_into_chunks(documents):
    """Split pages into overlapping word-based chunks."""
    chunks = []

    for document in documents:
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", document["text"])
            if p.strip()
        ]

        words = " ".join(paragraphs).split()

        if not words:
            continue

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"]
            })

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def embed_texts(client, texts, task_type):
    """Create Gemini embeddings in batches."""
    all_vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        response = call_with_retry(
            "document embedding",
            lambda: client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768
                )
            )
        )

        all_vectors.extend(
            np.array(item.values, dtype=np.float32)
            for item in response.embeddings
        )

    return np.vstack(all_vectors)


def build_index(client, documents):
    """Chunk documents and create their vector index."""
    chunks = split_into_chunks(documents)

    if not chunks:
        raise ValueError(
            "No readable text found. Use text-based PDFs, TXT or Markdown."
        )

    texts = [chunk["text"] for chunk in chunks]

    vectors = embed_texts(
        client,
        texts,
        task_type="RETRIEVAL_DOCUMENT"
    )

    # Normalize vectors for cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    return {
        "chunks": chunks,
        "vectors": vectors
    }


def retrieve(client, question, index):
    """Find relevant document chunks using cosine similarity."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not index or not index.get("chunks"):
        return []

    vectors = index["vectors"]

    if len(vectors) != len(index["chunks"]):
        raise ValueError("Index chunks and vectors do not match.")

    response = call_with_retry(
        "query embedding",
        lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
    )

    if not response.embeddings:
        return []

    query_vector = np.array(
        response.embeddings[0].values,
        dtype=np.float32
    )

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains invalid values.")

    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError("Query and document embedding dimensions differ.")

    norm = np.linalg.norm(query_vector)

    if norm < 1e-12:
        return []

    query_vector = query_vector / norm

    similarities = vectors @ query_vector

    best_indices = np.argsort(similarities)[::-1][:TOP_K]

    results = []

    for idx in best_indices:
        score = float(similarities[idx])

        # Ignore chunks that are not sufficiently relevant.
        if score < MIN_SIMILARITY_SCORE:
            continue

        results.append({
            **index["chunks"][int(idx)],
            "score": score
        })

    return results


def ask_question(client, question, index):
    """Retrieve evidence and generate an answer grounded in it."""
    retrieval_started = time.perf_counter()
    results = retrieve(client, question, index)
    print(
        f"[RAG] Retrieval + query embedding: "
        f"{time.perf_counter() - retrieval_started:.2f}s"
    )

    if not results:
        return FALLBACK, []

    context_parts = []

    for number, result in enumerate(results, start=1):
        context_parts.append(
            f"[{number}] File: {result['source']}, "
            f"Page: {result['page']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-grounded educational assistant.

Answer the user's question using ONLY the CONTEXT below.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain the answer, respond exactly:
   {FALLBACK}
4. Cite every factual claim using the evidence number, such as [1].
5. Do not create fake citations.
6. Explain clearly and simply.
7. If only part of the question is answered by the context,
   explain what is supported and what is missing.

CONTEXT:
{context}

USER QUESTION:
{question}

GROUNDED ANSWER:
"""

    generation_started = time.perf_counter()

    response = call_with_retry(
        "answer generation",
        lambda: client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000
            )
        )
    )

    print(
        f"[RAG] Gemini answer generation: "
        f"{time.perf_counter() - generation_started:.2f}s"
    )

    answer = response.text or FALLBACK

    # Keep only one displayed source per file and page.
    unique_sources = []
    seen_sources = set()

    for result in results:
        source_key = (result["source"], result["page"])

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(result)

    return answer, unique_sources
