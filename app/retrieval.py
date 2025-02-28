from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os
import asyncio
from app.ingestion import embedding_model, collection
from sentence_transformers import CrossEncoder

# Initialize the cross-encoder model (unchanged)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Set up Whoosh index (unchanged)
schema = Schema(content=TEXT(stored=True), chunk_id=TEXT(stored=True))
if not os.path.exists("whoosh_index"):
    os.mkdir("whoosh_index")
ix = create_in("whoosh_index", schema)

def index_chunks(chunks, doc_id):
    writer = ix.writer()
    for i, chunk in enumerate(chunks):
        writer.add_document(content=chunk, chunk_id=f"{doc_id}_{i}")
    writer.commit()

async def semantic_search(query, top_k):
    """Perform semantic search asynchronously."""
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        print(f"Query embedding shape: {len(query_embedding)}")
        semantic_results = collection.query(query_embeddings=[query_embedding], n_results=top_k * 2)
        semantic_chunks = semantic_results["documents"][0]
        print(f"Semantic chunks: {semantic_chunks}")
        return semantic_chunks
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []

async def keyword_search(query, top_k):
    """Perform keyword search asynchronously."""
    try:
        with ix.searcher() as searcher:
            whoosh_query = QueryParser("content", ix.schema).parse(query)
            keyword_results = searcher.search(whoosh_query, limit=top_k * 2)
            keyword_chunks = [hit["content"] for hit in keyword_results]
            print(f"Keyword chunks: {keyword_chunks}")
            return keyword_chunks
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []

async def retrieve(query, top_k=3):
    """
    Retrieve relevant chunks using parallel semantic and keyword search, then rerank with cross-encoder.
    
    Args:
        query (str): The user's question.
        top_k (int): Number of top results to return.
    
    Returns:
        list: Ranked list of top_k relevant chunks.
    """
    print(f"Processing query: {query}")
    
    # Run semantic and keyword searches in parallel
    semantic_task = semantic_search(query, top_k)
    keyword_task = keyword_search(query, top_k)
    semantic_chunks, keyword_chunks = await asyncio.gather(semantic_task, keyword_task)
    
    # Combine and deduplicate chunks
    combined_chunks = list(dict.fromkeys(semantic_chunks + keyword_chunks))
    print(f"Combined chunks before reranking: {combined_chunks}")
    
    # Rerank with cross-encoder
    if combined_chunks:
        query_chunk_pairs = [(query, chunk) for chunk in combined_chunks]
        scores = cross_encoder.predict(query_chunk_pairs)
        ranked_chunks = [chunk for _, chunk in sorted(zip(scores, combined_chunks), reverse=True)][:top_k]
        print(f"Ranked chunks: {ranked_chunks}")
    else:
        ranked_chunks = []
    
    return ranked_chunks