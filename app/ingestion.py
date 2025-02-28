from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import chromadb

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("documents")

def extract_text(file_path, file_type):
    if file_type == "pdf":
        reader = PdfReader(file_path)
        text = "".join(page.extract_text() for page in reader.pages)
    elif file_type == "html":
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()
    return text

def ingest_document(file_path, file_type, doc_id):
    text = extract_text(file_path, file_type)
    chunk_size, overlap = 500, 50
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
    )
    return len(chunks), chunks  # Return chunk count and chunks