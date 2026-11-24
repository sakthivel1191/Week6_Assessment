import os
import uuid
import tempfile
import pymupdf
import docx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = {}
API_KEY = os.getenv("API_KEY", "")
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
def check_api_key(api_key):
    if not API_KEY or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
def extract_pages(filename, content):
    extension = filename.lower().split(".")[-1]
    if extension == "txt":
        return [{
            "page": 1,
            "text": content.decode("utf-8")
        }]
    if extension == "pdf":
        with pymupdf.open(stream=content,filetype="pdf") as pdf:
            return [
                {
                    "page": number + 1,
                    "text": page.get_text()
                }
                for number, page in enumerate(pdf)
            ]
    if extension == "docx":
        with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            word_document = docx.Document(temp_file.name)
            text = "\n".join(
                paragraph.text
                for paragraph in word_document.paragraphs
            )
            return [{
                "page": 1,
                "text": text
            }]
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type"
    )
def create_chunk_id(source, page, index, text):
    value = f"{source}_{page}_{index}_{text}"
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            value
        )
    )
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_chunks": len(documents)
    }
@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Header(...)
):
    check_api_key(api_key)
    filename = file.filename
    content = await file.read()
    pages = extract_pages(filename, content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    new_chunks = []
    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]
        chunks = splitter.split_text(text)
        for index, chunk_text in enumerate(chunks):
            chunk_id = create_chunk_id(
                filename,
                page_number,
                index,
                chunk_text
            )
            new_chunks.append({
                "id": chunk_id,
                "source": filename,
                "page": page_number,
                "section": "General",
                "doc_type": filename.split(".")[-1].upper(),
                "text": chunk_text
            })
    if new_chunks:
        embeddings = model.encode(
            [chunk["text"] for chunk in new_chunks]
        )
        for chunk, embedding in zip(new_chunks, embeddings):
            chunk["embedding"] = embedding
    for chunk in new_chunks:
        documents[chunk["id"]] = chunk
    return {
        "message": "Document uploaded successfully",
        "filename": filename,
        "total_chunks": len(new_chunks),
        "total_stored_chunks": len(documents)
    }
@app.post("/search")
def search(
    request: SearchRequest,
    api_key: str = Header(...)
):
    check_api_key(api_key)
    if not documents:
        return {
            "query": request.query,
            "results": []
        }
    query_embedding = model.encode([request.query])
    results = []
    for document in documents.values():
        if request.doc_type:
            if document["doc_type"] != request.doc_type.upper():
                continue
        score = cosine_similarity(
            query_embedding,
            [document["embedding"]]
        )[0][0]
        if score >= request.min_score:
            results.append({
                "source": document["source"],
                "page": document["page"],
                "section": document["section"],
                "score": float(score),
                "text": document["text"]
            })
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )
    return {
        "query": request.query,
        "results": results[:request.top_k]
    }
