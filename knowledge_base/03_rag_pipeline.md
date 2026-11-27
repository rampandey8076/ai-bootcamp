# Document 3: Retrieval-Augmented Generation (RAG)

## 1. Introduction

A large language model can generate fluent answers, but it may not know the contents of a private document or a recently updated knowledge base.

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. Before generating an answer, the application searches a knowledge base and supplies relevant text to the language model.

The goal is to make answers more useful and grounded in supplied evidence.

## 2. What is RAG?

RAG is a workflow with two broad phases:

**Indexing phase:** Documents are prepared, divided into chunks, converted into embeddings, and stored.

**Question-answering phase:** A user's question is embedded, relevant chunks are retrieved, and a language model generates an answer using the retrieved context.

The model receives both the question and the selected evidence.

## 3. The indexing pipeline

A typical indexing pipeline includes the following steps.

### Step 1: Document collection

Collect source documents such as PDFs, text files, Markdown files, manuals, or notes.

The documents should be relevant to one topic and legally usable by the application.

### Step 2: Text extraction

Extract readable text from the files.

PDF files may contain selectable text or only scanned images. A basic text extractor cannot reliably read a scanned page without OCR.

### Step 3: Chunking

Long documents are divided into smaller pieces called chunks.

Chunking helps the retrieval system find specific information without sending the entire knowledge base to the language model.

### Step 4: Embedding

Each chunk is sent to an embedding model. The model returns a vector representing the chunk's semantic content.

### Step 5: Storage

Store the vector along with the original text and metadata.

Useful metadata includes the source file and page number. This allows the application to show where retrieved information came from.

## 4. Chunking strategies

Chunking is an important design decision.

### Fixed-size chunking

Text is divided according to a fixed number of characters or tokens.

It is simple to implement, but may split a sentence or concept at an awkward point.

### Overlapping chunking

Adjacent chunks share a small amount of text.

For example, a chunk may contain 180 words and overlap the next chunk by 35 words.

Overlap can preserve context near chunk boundaries. Too much overlap can create duplicate content and increase embedding cost.

### Structure-aware chunking

The application uses document structure such as headings, paragraphs, sections, or pages to create more meaningful chunks.

This can preserve relationships between headings and the content that follows them.

The best strategy depends on the documents and the questions users are expected to ask.

## 5. The retrieval pipeline

When a user asks a question:

1. The question is converted into an embedding.
2. The vector database searches for similar document vectors.
3. The application selects the top relevant chunks.
4. The application retrieves the original text and metadata.
5. The selected text is placed into the language model prompt.

The number of retrieved chunks is often called top-k.

A very small top-k may omit necessary evidence. A very large top-k may introduce irrelevant context and increase prompt size.

## 6. Grounded generation

The language model receives the question and retrieved evidence.

A grounded prompt should tell the model to:

- Use only the supplied context.
- Avoid inventing facts.
- Admit when evidence is missing.
- Cite the retrieved source identifiers.
- Separate supported facts from unsupported assumptions.

The application should not treat a fluent answer as proof that the answer is correct.

## 7. Citations

A useful RAG application preserves source metadata during indexing.

A retrieved chunk might contain:

- Source: database_guide.pdf
- Page: 8
- Text: a paragraph about vector indexing

The model can refer to this evidence using a label such as [1]. The application then displays the corresponding file and page under the answer.

Citations should point to the actual retrieved evidence. The application should never invent a page number or source name.

## 8. Handling questions without evidence

A reliable knowledge-base chatbot should not answer every question.

For example, if a knowledge base discusses vector databases but the user asks for tomorrow's weather, the documents do not contain the answer.

The application should return a clear fallback message rather than guessing.

A retrieval similarity threshold can help identify weak matches, but thresholds must be evaluated with real questions. A high score does not guarantee that a chunk answers the question.

## 9. RAG compared with ordinary generation

Ordinary generation sends a question to a language model without retrieving application-specific evidence.

RAG first searches a knowledge base and then includes relevant passages in the generation prompt.

RAG can make answers more closely connected to supplied documents, but it does not guarantee correctness. Retrieval can fail, documents can be incomplete, and the model can misinterpret context.

## 10. Evaluation

RAG systems should be tested using realistic questions.

Useful test categories include:

- Fact questions answered by one passage
- Questions requiring multiple passages
- Questions using different wording from the source
- Questions whose answers are absent from the knowledge base

For each test, record:
- The question
- The expected answer
- The actual answer
- The retrieved sources
- Pass or fail
- A brief explanation

If the wrong passage is retrieved, investigate chunking and retrieval. If the right passage is retrieved but the answer is wrong, investigate the prompt and generation step.

## 11. Common RAG failure points

**Poor text extraction:** The application cannot retrieve information that was never extracted.

**Poor chunk boundaries:** Important definitions or conditions may be separated.

**Weak retrieval:** The relevant chunk may not appear in the top-k results.

**Missing metadata:** The application cannot display reliable source citations.

**Weak grounding instructions:** The model may use unsupported information.

**Incomplete knowledge base:** A missing answer cannot be retrieved.

## 12. Privacy and API-key handling

A public RAG application may ask each visitor to provide their own API key.

The application should:
- Accept the key through a password-style input.
- Use it only to create the API client.
- Avoid printing it in logs.
- Avoid saving it in files or source control.
- Avoid storing uploaded visitor documents.
- Process uploaded content in memory where possible.

A public repository must never contain a real API key.

## 13. Summary

RAG connects document retrieval with language generation. Its quality depends on the complete pipeline: extraction, chunking, embeddings, storage, retrieval, grounded prompting, citations, and evaluation.

A trustworthy RAG assistant should answer from evidence and clearly admit when the knowledge base does not contain the answer.
