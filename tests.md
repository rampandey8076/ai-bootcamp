# RAG Chatbot Test Results

**Project:** Vector Database and RAG Knowledge Assistant  
**Test date:** 2026-11-27

## 1. Automated test results

Test framework: pytest 9.2.0  
Python version: 3.14.10

Latest recorded result: **20 passed, 0 failed**

Coverage includes:

- PDF and text extraction
- Page-break handling
- Document chunking and overlap
- Empty document handling
- Sample document loading
- Gemini API key validation
- Empty index validation
- Retrieval and similarity filtering
- Retry after temporary API failures
- Maximum retry attempts
- Permanent error handling
- Retry configuration validation
- Unrelated-number false-positive prevention
- Unanswerable question fallback
- Source page preservation
- Duplicate source page removal

Run the automated tests:

```bash
python -m pytest -v
```

## 2. Manual chatbot testing

These scenarios must be tested against the deployed application.
Do not mark them as passed until actual chatbot responses are verified.

### Test 1 — Text embeddings

Question: What is a text embedding?

Expected: A numerical representation of text that allows software to compare meaning.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

### Test 2 — Cosine similarity

Question: What does cosine similarity measure?

Expected: It measures the angular similarity between two vectors.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

### Test 3 — Vector databases

Question: What is the purpose of a vector database?

Expected: It stores vector representations and retrieves similar vectors.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

### Test 4 — RAG workflow

Question: What are the main phases of a RAG workflow?

Expected: The indexing phase and the question-answering phase.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

### Test 5 — Unanswerable question

Question: Who won the FIFA World Cup in 2022?

Expected: The chatbot should explain that this information is not present in the uploaded knowledge base.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

### Test 6 — Semantic retrieval

Question: How can a chatbot find a paragraph about semantic retrieval if the user searches using different words?

Expected: Text embeddings represent meaning as vectors. The question and document chunks can be embedded and compared using similarity search.

Actual answer: Pending live test.

Retrieved source: Pending live test.

Result: PENDING

## 3. Improvements implemented

- Added retry handling for temporary Gemini API failures.
- Added exponential backoff.
- Added retry-attempt validation.
- Improved HTTP status detection.
- Added tests for permanent errors and unrelated numbers.
- Added duplicate source-page removal validation.
- Preserved document page references in generated answers.
