import os
import re
import uuid
import fitz
import docx
from datetime import date
from langchain_text_splitters import RecursiveCharacterTextSplitter
folder = "documents"
documents = []
def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return [{
            "page": 1,
            "text": file.read()
        }]
def read_pdf(path):
    pdf = fitz.open(path)
    pages = []
    for page_number, page in enumerate(pdf):
        pages.append({
            "page": page_number + 1,
            "text": page.get_text()
        })
    return pages
def read_docx(path):
    document = docx.Document(path)
    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )
    return [{
        "page": 1,
        "text": text
    }]
def load_documents():
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".txt"):
            pages = read_txt(path)
        elif filename.endswith(".pdf"):
            pages = read_pdf(path)
        elif filename.endswith(".docx"):
            pages = read_docx(path)
        else:
            continue
        for item in pages:
            documents.append({
                "source": filename,
                "page": item["page"],
                "text": item["text"],
                "doc_type": filename.split(".")[-1].upper()
            })
def fixed_chunking(text, size=512, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks
def sentence_chunking(text):
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )
    return [
        sentence
        for sentence in sentences
        if sentence
    ]
def recursive_chunking(text, size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
def get_section(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:100]
    return "General"
def create_chunk_id(source, page, index, text):
    value = f"{source}_{page}_{index}_{text}"
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            value
        )
    )
def describe_chunk(chunk):
    return (
        f"Source: {chunk['source']}, "
        f"Page: {chunk['page']}, "
        f"Section: {chunk['section']}, "
        f"Chunk: {chunk['chunk_index']}"
    )
load_documents()
all_chunks = []
for document in documents:
    chunks = recursive_chunking(document["text"])
    for index, chunk_text in enumerate(chunks):
        chunk = {
            "id": create_chunk_id(
                document["source"],
                document["page"],
                index,
                chunk_text
            ),
            "source": document["source"],
            "page": document["page"],
            "section": get_section(chunk_text),
            "created_date": str(date.today()),
            "chunk_index": index,
            "doc_type": document["doc_type"],
            "text": chunk_text
        }
        all_chunks.append(chunk)
print("Total documents:", len(documents))
print("Total chunks:", len(all_chunks))
print("\nCHUNK CITATION")
if all_chunks:
    print(describe_chunk(all_chunks[0]))
print("\nCHUNKING COMPARISON")
for document in documents:
    text = document["text"]
    fixed = fixed_chunking(text)
    sentence = sentence_chunking(text)
    recursive = recursive_chunking(text)
    print("\nDocument:", document["source"])
    print("Fixed chunks:", len(fixed))
    print("Sentence chunks:", len(sentence))
    print("Recursive chunks:", len(recursive))
print("\nCHUNK SIZE EXPERIMENT")
sizes = [128, 256, 512, 1024]
for size in sizes:
    total = 0
    for document in documents:
        chunks = recursive_chunking(
            document["text"],
            size=size,
            overlap=50
        )
        total += len(chunks)
    print("Chunk size:", size)
    print("Total chunks:", total)
print("\nSENTENCE CUT EXAMPLE")
sample = "Employees must follow company rules. Safety is important."
chunks = fixed_chunking(
    sample,
    size=5,
    overlap=1
)
for chunk in chunks:
    print(chunk)
print("\nSTABLE ID TEST")
first_run = {
    chunk["id"]: chunk
    for chunk in all_chunks
}
second_run = {
    chunk["id"]: chunk
    for chunk in all_chunks
}
print("Run 1:", len(first_run))
print("Run 2:", len(second_run))
print("New IDs:", len(set(second_run) - set(first_run)))
print("Removed IDs:", len(set(first_run) - set(second_run)))
print("\nDOCUMENT EDIT TEST")
if documents:
    edited_documents = [
        document.copy()
        for document in documents
    ]
    edited_documents[0]["text"] += (
        "\nNew company policy paragraph."
    )
    edited_ids = set()
    for document in edited_documents:
        chunks = recursive_chunking(document["text"])
        for index, text in enumerate(chunks):
            chunk_id = create_chunk_id(
                document["source"],
                document["page"],
                index,
                text
            )
            edited_ids.add(chunk_id)
    old_ids = set(first_run.keys())
    print("New chunk IDs:",len(edited_ids - old_ids))
    print("Removed chunk IDs:",len(old_ids - edited_ids))
