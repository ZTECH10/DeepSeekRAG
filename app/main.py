from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from app.ingestion import ingest_document
from app.retrieval import retrieve, index_chunks
from app.generation import generate_response
import os
from cachetools import TTLCache

app = FastAPI()

# Initialize cache with TTL of 1 hour and max size of 100 entries
cache = TTLCache(maxsize=100, ttl=3600)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"data/raw/{file.filename}"
    os.makedirs("data/raw", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    file_type = "pdf" if file.filename.endswith(".pdf") else "html"
    chunk_count, chunks = ingest_document(file_path, file_type, file.filename)
    index_chunks(chunks, file.filename)
    return {"message": f"Uploaded and processed {file.filename} with {chunk_count} chunks"}

@app.get("/query")
async def query_system(query: str):
    async def stream_response():
        # Check cache first
        if query in cache:
            print(f"Cache hit for query: {query}")
            yield cache[query]
            return  # Exit generator

        # Retrieve relevant chunks
        chunks = await retrieve(query)

        # If no chunks are found, yield a simple message
        if not chunks:
            response = "No relevant content found."
            cache[query] = response
            yield response
            return  # Exit generator

        # Stream the response with error handling
        response = ""
        try:
            for chunk in generate_response(query, chunks):
                response += chunk
                yield chunk
            cache[query] = response
        except Exception as e:
            print(f"Streaming failed: {e}")
            yield response if response else "Error: Failed to generate a response."

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)