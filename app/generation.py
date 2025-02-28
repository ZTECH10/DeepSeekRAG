from huggingface_hub import InferenceClient
import os

# Get HF_TOKEN from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it before running the app.")

# Initialize Hugging Face Inference API client
generator = InferenceClient(token=HF_TOKEN, model="mistralai/Mixtral-8x7B-Instruct-v0.1")

def generate_response(query, chunks):
    """
    Generate a streamed response based on the query and retrieved chunks.
    
    Args:
        query (str): The user's question.
        chunks (list): List of retrieved text chunks from the document.
    
    Yields:
        str: Chunks of the generated response from the API.
    """
    # Combine chunks into context, or use a fallback if empty
    context = "\n\n".join(chunks) if chunks else "No relevant context found."
    # Create a detailed prompt for the API
    prompt = (
        "You are an assistant that answers questions based strictly on the provided context. "
        "Do not add external information or speculate beyond what is given. "
        "If the context does not contain enough information to answer the question, "
        "say 'I don’t have enough information to answer this question.'\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    try:
        # Call the Hugging Face API with streaming enabled
        response = generator.text_generation(
            prompt,
            max_new_tokens=200,    # Limit response length
            do_sample=True,        # Enable sampling for varied outputs
            temperature=0.7,       
            stream=True            # Enable streaming
        )
        # Yield each chunk directly as it comes from the API
        for chunk in response:
            yield chunk  # No slicing, just raw output
    except Exception as e:
        # Handle API errors (e.g., rate limits, network issues)
        print(f"API Error: {e}")
        yield "Error: Failed to generate a response. Please try again."