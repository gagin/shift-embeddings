import os, torch
from pinecone_utils import initialize_pinecone, load_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Debug print to verify the API key is actually set
print(f"API Key present: {'PINECONE_API_KEY' in os.environ}")
print(f"API Key value: {os.getenv('PINECONE_API_KEY')}")

# Get Pinecone credentials from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
index_name = os.getenv('PINECONE_INDEX_NAME')

# Initialize Pinecone
index = initialize_pinecone(pinecone_api_key, pinecone_environment, index_name)

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_with_pinecone(prompt: str, max_length: int = 20, temperature: float = 0.7, top_k_neighbors: int = 5):
    """Generate text using hidden states + nearest neighbors instead of LM head"""
    print("Generating text...")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
    generated = input_ids[0].tolist()  # Keep track of generated tokens
    
    for _ in range(max_length):
        # Get hidden states for current sequence
        outputs = model(torch.tensor([generated]).to("cpu"), output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1]  # Get last token's hidden state
        
        # Query Pinecone for nearest neighbors
        query_vector = last_hidden_state.detach().numpy().tolist()[0]
        results = index.query(vector=query_vector, top_k=top_k_neighbors)
        
        # Get logits from the model's LM head
        logits = model.lm_head(last_hidden_state)
        
        # Apply temperature and sample
        probs = (logits / temperature).softmax(dim=-1)
        next_token = torch.multinomial(probs[0], 1).item()
        
        # Debug print for token generation
        print(f"Selected token {next_token}: '{tokenizer.decode([next_token])}'")
        
        generated.append(next_token)
    
    return {
        'generated_text': tokenizer.decode(generated)
    }

def main():
    prompt = "Because I'm a carpenter, I do"
    print(f"\nGenerating text for prompt: '{prompt}'")
    outputs = generate_with_pinecone(prompt)
    print("\nFinal outputs:")
    print(f"Generated text: {outputs['generated_text']}")

if __name__ == "__main__":
    main()
