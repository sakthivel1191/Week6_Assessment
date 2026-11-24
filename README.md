
# Week 6 - Embeddings and Semantic Search

## Project Overview

This project implements a semantic search application using Python, embeddings and FastAPI.

## Technologies Used

- Python
- Sentence Transformers
- all-MiniLM-L6-v2
- OpenAI Embeddings API
- NumPy
- Scikit-learn
- PyMuPDF
- python-docx
- LangChain Text Splitter
- FastAPI
- Pytest

## Task 1 - Embeddings

- Generated embeddings using local and API models.
- Implemented cosine similarity.
- Compared similarity with Scikit-learn.
- Implemented semantic search.
- Demonstrated duplicate detection.
- Implemented K-Means clustering.

## Task 2 - Chunking

- Extracted PDF, Word and TXT documents.
- Implemented fixed-size chunking.
- Implemented sentence-based chunking.
- Implemented recursive character splitting.
- Added metadata and stable chunk IDs.

## Task 3 - Vector Database

Skipped as per project scope.

## Task 4 - Search API

- Created FastAPI endpoints.
- Implemented document upload.
- Implemented semantic search.
- Added API key protection.
- Implemented retrieval evaluation.
- Created six pytest test cases.

## How to Run

Install packages:

pip install sentence-transformers scikit-learn numpy openai python-dotenv pymupdf python-docx langchain-text-splitters fastapi uvicorn python-multipart pytest httpx

Run Task 1:

python embeddings.py

Run Task 2:

python chunking.py

Run API:

uvicorn api:app --reload

Run evaluation:

python evaluation.py

Run tests:

pytest -v

## Conclusion

This project demonstrates document extraction, embeddings, text chunking, cosine similarity and semantic search using FastAPI.
