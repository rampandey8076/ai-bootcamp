# Document 1: Text Embeddings

## 1. Introduction

Computers traditionally process text as characters, words, or tokens. These representations are useful for exact matching, but they do not naturally represent the meaning of a sentence.

Text embeddings solve this problem by converting text into numerical vectors. A vector is an ordered collection of numbers. An embedding model maps a piece of text to a point in a multidimensional vector space.

Texts with related meanings can have vectors that are close to each other. This makes it possible to search by meaning rather than only by exact words.

## 2. What is an embedding?

An embedding is a numerical representation of an input such as a word, sentence, paragraph, or document.

For example, consider these sentences:

- A car is a vehicle used for transportation.
- An automobile can carry people from one place to another.
- I enjoy eating mangoes.

The first two sentences discuss similar concepts even though they use different words. An embedding model can represent their semantic similarity in vector form.

The third sentence is about a different subject and would generally be less similar to the first two.

## 3. Dimensions

A vector contains multiple numerical dimensions. An embedding model determines the number of dimensions in its output.

For example, a simplified vector might look like:

[0.12, -0.34, 0.81, 0.09]

Real embedding models commonly produce much larger vectors. The dimensions are learned numerical features; developers should not assume that one particular dimension represents a human-readable concept.

## 4. Semantic similarity

Semantic similarity measures how closely two pieces of text relate in meaning.

A common measure is cosine similarity. It compares the angle between two vectors.

Cosine similarity is calculated as:

cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Here:
- A and B are vectors.
- A · B is their dot product.
- ||A|| and ||B|| are their magnitudes.

A higher cosine similarity usually indicates that two vectors point in a more similar direction.

The exact interpretation of a score depends on the embedding model and dataset. A similarity score should not automatically be treated as a probability.

## 5. Why embeddings are useful

Embeddings are useful when users do not know the exact words used in a document.

For example, a user may search for "How can I find related information?" while a document uses the phrase "semantic retrieval."

A keyword search might miss the relationship. An embedding-based search can retrieve text based on meaning.

Common applications include:
- Semantic search
- Recommendation systems
- Document clustering
- Duplicate detection
- Retrieval-Augmented Generation
- Similarity-based classification

## 6. The embedding workflow

A typical embedding workflow contains these steps:

1. Collect the text.
2. Clean unnecessary formatting.
3. Divide long documents into manageable chunks.
4. Send each chunk to an embedding model.
5. Receive a numerical vector for each chunk.
6. Store the vectors with their source information.
7. Embed a user's search query.
8. Compare the query vector with stored vectors.
9. Return the most relevant text.

The original text must be stored alongside the vector. A vector alone is not a readable replacement for the source document.

## 7. Document embeddings and query embeddings

Retrieval systems often distinguish between document and query embeddings.

A document embedding represents content that can be retrieved. A query embedding represents the user's search request.

Some embedding APIs support task types that tell the model whether the input is intended for retrieval documents or retrieval queries.

Using the appropriate task type can improve retrieval behavior.

## 8. Limitations

Embeddings are not perfect representations of meaning.

They can:
- Retrieve text that is related but does not answer the question.
- Miss rare terms or exact identifiers.
- Be affected by ambiguous language.
- Reflect limitations in the training data.
- Produce similarity scores that require evaluation.

For these reasons, a reliable application should inspect retrieved text and use it as evidence rather than assuming the closest vector is always correct.

## 9. Key terms

**Embedding:** A numerical representation of text.

**Vector:** An ordered collection of numerical values.

**Semantic search:** Searching using meaning rather than exact word matching.

**Cosine similarity:** A measure of the angular similarity between two vectors.

**Query embedding:** A vector representing a user's search question.

**Document embedding:** A vector representing a document or document chunk.

## 10. Summary

Text embeddings allow software to compare text by meaning. They are an important foundation for semantic search and RAG systems. Their quality depends on the embedding model, text preparation, chunking, and retrieval design.
