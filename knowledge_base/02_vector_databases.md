# Document 2: Vector Databases

## 1. Introduction

Applications that use embeddings need a way to store and search numerical vectors. A vector database is designed to manage vector representations and retrieve records that are similar to a query vector.

A traditional relational database stores structured information in tables. A vector database focuses on similarity search over high-dimensional vectors, often while keeping metadata and original text associated with each vector.

## 2. What is a vector database?

A vector database stores records that commonly include:

- A unique identifier
- An embedding vector
- Original text or a reference to it
- Metadata such as document name, page, category, or timestamp

For example, a record might contain a vector for a paragraph from a programming guide, along with the paragraph text and its page number.

When a user submits a question, the application converts the question into a vector and searches for nearby stored vectors.

## 3. Similarity search

Similarity search identifies vectors that are close to a query vector according to a selected distance or similarity measure.

Common measures include:

### Cosine similarity

Cosine similarity compares the angle between vectors. It is widely used for text embeddings.

### Euclidean distance

Euclidean distance measures the straight-line distance between two points in vector space.

### Dot product

The dot product combines vector direction and magnitude. Its behavior depends on whether vectors have been normalized.

The correct measure depends on the embedding model and the way its vectors are produced.

## 4. Exact search and approximate search

An exact search compares the query against every stored vector. It can be accurate but becomes expensive as the collection grows.

Approximate nearest neighbor (ANN) search uses indexing techniques to find likely nearest vectors without exhaustively comparing every record.

ANN search is often faster for large collections, but it can trade some retrieval accuracy for speed.

## 5. Vector indexing

An index organizes vectors to make similarity searches more efficient.

One commonly used family of indexing approaches is graph-based approximate nearest neighbor search. Hierarchical Navigable Small World (HNSW) is an example.

HNSW builds a navigable graph of vectors. Search moves through the graph to find vectors that are likely to be close to the query.

Other indexing approaches may use inverted files or quantization.

The choice of index affects:
- Search speed
- Recall
- Memory use
- Index construction time
- Storage requirements

## 6. Metadata filtering

Vector databases often support filtering by metadata.

Imagine a knowledge base containing documents from several subjects. A user may want results only from the "database" category.

The application can combine vector similarity with a metadata condition.

Examples of metadata include:
- Document title
- Page number
- Topic
- Author
- Date
- Access category

Metadata is also important for showing citations in a RAG chatbot.

## 7. Vector databases and traditional databases

Traditional databases are useful for structured queries such as:

- Find a user with a particular ID.
- Retrieve all orders from a date.
- Find products with a price below a value.

Vector databases are useful for questions such as:

- Find paragraphs similar in meaning to this question.
- Retrieve documents about a concept even if they use different words.
- Find content related to a description.

Some modern database systems support both structured data and vector search. The best choice depends on application requirements.

## 8. ChromaDB

ChromaDB is a vector database that can be used to store embeddings and perform similarity searches.

A basic workflow is:

1. Create a collection.
2. Prepare text documents.
3. Generate embeddings.
4. Add documents, IDs, and metadata.
5. Query the collection.
6. Inspect the returned documents and metadata.

A collection groups related records. IDs help identify individual records, while metadata preserves information such as source files and page numbers.

ChromaDB can be used for local experiments and application prototypes. Production deployment requires considering persistence, scaling, security, and operational requirements.

## 9. PostgreSQL and pgvector

PostgreSQL is a relational database system. The pgvector extension adds vector storage and similarity-search capabilities.

With pgvector, an application can store regular relational fields alongside embedding vectors.

This can be useful when an application already uses PostgreSQL and wants to combine normal database queries with semantic retrieval.

The developer must still choose an appropriate distance operator and index strategy.

## 10. Choosing a vector database

Consider these factors:

- Collection size
- Query volume
- Latency requirements
- Filtering requirements
- Persistence
- Deployment environment
- Operational complexity
- Cost
- Integration with the existing application

A small learning project may work with an in-memory vector index. A larger application may need a persistent vector database.

## 11. Common mistakes

**Storing vectors without their source text:** This makes it difficult to present useful retrieved evidence.

**Losing metadata:** The application may retrieve content but fail to show the document and page.

**Using inconsistent embeddings:** Query and document vectors should use compatible embedding models and settings.

**Ignoring index configuration:** Search performance and recall depend on index choices.

**Assuming the nearest result is always correct:** Similarity is a ranking signal, not proof that the result answers the question.

## 12. Summary

A vector database stores and retrieves numerical embeddings. It supports semantic search, metadata filtering, and nearest-neighbor retrieval. The vector index, distance measure, metadata design, and source-text handling all influence the quality of a retrieval application.
