import os
import torch
from pinecone_utils import initialize_pinecone, load_model

def get_pinecone_vector(word, tokenizer, index):
    """Get the Pinecone vector for a word"""
    token_id = tokenizer.encode(word, add_special_tokens=False)[0]
    print(f"\nFetching '{word}' (token ID: {token_id}) from Pinecone")
    
    pinecone_vector = index.fetch(ids=[str(token_id)])
    if not pinecone_vector['vectors']:
        print(f"'{word}' not found in Pinecone index!")
        return None
    
    return pinecone_vector['vectors'][str(token_id)]['values']

def analogy_test():
    # Initialize model and tokenizer (only needed for token IDs)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model, tokenizer = load_model(model_name)
    
    # Initialize Pinecone
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX_NAME')
    index = initialize_pinecone(pinecone_api_key, pinecone_environment, index_name)
    
    # Get Pinecone vectors for each word
    words = ["king", "man", "woman", "queen"]
    pinecone_vectors = {}
    
    for word in words:
        vector = get_pinecone_vector(word, tokenizer, index)
        if vector is None:
            print(f"Cannot continue - missing vector for '{word}'")
            return
        pinecone_vectors[word] = vector
    
    # Perform the vector arithmetic using Pinecone vectors
    result_vector = []
    for i in range(len(pinecone_vectors["king"])):
        result = (pinecone_vectors["king"][i] - 
                 pinecone_vectors["man"][i] + 
                 pinecone_vectors["woman"][i])
        result_vector.append(result)
    
    # Calculate cosine similarity with queen using Pinecone vectors
    queen_similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(result_vector), 
        torch.tensor(pinecone_vectors["queen"]), 
        dim=0
    )
    
    print("\n=== Analogy Results (Using Pinecone Vectors) ===")
    print("\nTesting: king - man + woman = ?")
    print(f"Cosine similarity with 'queen': {queen_similarity:.4f}")
    
    # Query Pinecone for nearest neighbors to result
    neighbors = index.query(vector=result_vector, top_k=5)
    print("\nNearest neighbors to the result vector:")
    for match in neighbors['matches']:
        token_id = int(match['id'])
        token = tokenizer.decode([token_id])
        print(f"Token: {token:10} Similarity: {match['score']:.4f}")

if __name__ == "__main__":
    analogy_test()