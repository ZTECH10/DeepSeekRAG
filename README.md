# DeepSeekRAG


### Intro:
--------------------------------
DeepSeekRAG is a retrieval-augmented generation (RAG) application that integrates hybrid retrieval(semantic and keyword search) with the HuggingFace transformers for accurate document question-answering. The application will be tested on the popular DeepSeek-V3 technical report.




###  Application Workflow Overview :
--------------------------------
- Provide a high-level overview of how the system processes uploads and queries.
- Focuses on key stages: file ingestion, storage, retrieval, generation, and caching.
<img width="320" alt="Screenshot 2025-02-27 at 7 28 15 PM" src="https://github.com/user-attachments/assets/12089782-db1f-4486-9798-37bf858b2f83" />



### Application Workflow Details:
--------------------------------
- Breaks down components and technologies used at each stage.
- Shows specific libraries (PyPDF2, BeautifulSoup, Whoosh, ChromaDB, Hugging Face).
- Includes error handling, ranking, and streaming logic.
<img width="1490" alt="Screenshot 2025-02-27 at 7 54 47 PM" src="https://github.com/user-attachments/assets/00a6d65c-52fb-4286-84a8-ca50fd099d90" />


### Python-based Tech Stack Choices and Why?
--------------------------------
- Web Framework: FastAPI
  - Why? it's lightweight, asynchronous (i.e. streaming), and has excellent support for RESTful APIs.

- Vector Database: Chroma
  - Why? It's simple, open-source, and optimized for storing embeddings with semantic search. 

- Embedding Model: Sentence Transformers (e.g., all-MiniLM-L6-v2)
  - Why? It’s fast, produces high-quality embeddings for semantic search, and balances accuracy with resource usage.

- Keyword Search: Whoosh
  - Why? it's lightweight and effective for keyword-based retrieval.

- Language Model: Hugging Face Transformers (e.g.mistralai/Mixtral-8x7B-Instruct-v0.1 via Hugging Face API) . **HF Token Needed!**
  - Why? it's open-source with high-quality generation.

- PDF/HTML Parsing: PyPDF2 (for PDFs) and BeautifulSoup (for HTML)
  - Why?  it's reliable for text extraction.

- Streaming: FastAPI’s StreamingResponse
  - Why? It has built-in support for asynchronous streaming.
