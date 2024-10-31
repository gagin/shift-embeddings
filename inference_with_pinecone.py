import os, torch
from pinecone_utils import initialize_pinecone, load_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from typing import List, Dict
import time
import nltk

nltk.download('punkt')  # Required for tokenization in BLEU score calculation

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
        
        # Average the nearest neighbor vectors to get a better representation
        neighbor_vectors = [torch.tensor(match.values) for match in results.matches if match.values]
        if neighbor_vectors:
            averaged_vector = torch.stack(neighbor_vectors).mean(dim=0)
            # Use the averaged vector instead of the original hidden state
            logits = model.lm_head(averaged_vector.unsqueeze(0))
        else:
            # Fallback to original hidden state if no neighbors found
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

def generate_classic(prompt: str, model, tokenizer, max_length: int = 20, temperature: float = 0.7):
    """Generate text using classic LM head approach"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
    generated = input_ids[0].tolist()
    
    for _ in range(max_length):
        outputs = model(torch.tensor([generated]).to("cpu"))
        logits = outputs.logits[:, -1, :]
        probs = (logits / temperature).softmax(dim=-1)
        next_token = torch.multinomial(probs[0], 1).item()
        generated.append(next_token)
    
    return tokenizer.decode(generated)

def evaluate_methods(test_prompts: List[str], reference_texts: List[str] = None):
    """Compare both generation methods across multiple metrics"""
    results = {
        'classic': {'texts': [], 'times': [], 'log_probs': []},
        'pinecone': {'texts': [], 'times': [], 'log_probs': []}
    }
    
    for prompt in test_prompts:
        # Test classic approach
        start_time = time.time()
        classic_text = generate_classic(prompt, model, tokenizer)
        results['classic']['times'].append(time.time() - start_time)
        results['classic']['texts'].append(classic_text)
        
        # Test Pinecone approach
        start_time = time.time()
        pinecone_text = generate_with_pinecone(prompt)['generated_text']
        results['pinecone']['times'].append(time.time() - start_time)
        results['pinecone']['texts'].append(pinecone_text)
        
        # Calculate log probabilities for both methods
        with torch.no_grad():
            # Classic
            classic_ids = tokenizer.encode(classic_text, return_tensors="pt")
            classic_outputs = model(classic_ids)
            classic_log_probs = torch.log_softmax(classic_outputs.logits, dim=-1)
            results['classic']['log_probs'].append(classic_log_probs.mean().item())
            
            # Pinecone
            pinecone_ids = tokenizer.encode(pinecone_text, return_tensors="pt")
            pinecone_outputs = model(pinecone_ids)
            pinecone_log_probs = torch.log_softmax(pinecone_outputs.logits, dim=-1)
            results['pinecone']['log_probs'].append(pinecone_log_probs.mean().item())
    
    # Calculate BLEU scores if reference texts are provided
    if reference_texts:
        results['classic']['bleu'] = [
            sentence_bleu([ref.split()], gen.split())
            for ref, gen in zip(reference_texts, results['classic']['texts'])
        ]
        results['pinecone']['bleu'] = [
            sentence_bleu([ref.split()], gen.split())
            for ref, gen in zip(reference_texts, results['pinecone']['texts'])
        ]
    
    return results

def main():
    # Test prompts (ideally with known good continuations)
    test_prompts = [
        "Because I'm a carpenter, I do",
        "The weather today is",
        "In computer science, a binary tree is",
        "The best way to learn programming is"
    ]
    
    # Optional: corresponding reference texts for these prompts
    reference_texts = [
        "Because I'm a carpenter, I do a lot of woodworking and furniture making",
        "The weather today is sunny with clear skies",
        "In computer science, a binary tree is a tree data structure where each node has at most two children",
        "The best way to learn programming is through hands-on practice and building projects"
    ]
    
    results = evaluate_methods(test_prompts, reference_texts)
    
    # Print results
    print("\nResults Summary:")
    print("\nAverage Generation Time:")
    print(f"Classic: {np.mean(results['classic']['times']):.3f}s")
    print(f"Pinecone: {np.mean(results['pinecone']['times']):.3f}s")
    
    print("\nAverage Log Probability:")
    print(f"Classic: {np.mean(results['classic']['log_probs']):.3f}")
    print(f"Pinecone: {np.mean(results['pinecone']['log_probs']):.3f}")
    
    if 'bleu' in results['classic']:
        print("\nAverage BLEU Score:")
        print(f"Classic: {np.mean(results['classic']['bleu']):.3f}")
        print(f"Pinecone: {np.mean(results['pinecone']['bleu']):.3f}")
    
    print("\nGenerated Text Samples:")
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt: {prompt}")
        print(f"Classic: {results['classic']['texts'][i]}")
        print(f"Pinecone: {results['pinecone']['texts'][i]}")

if __name__ == "__main__":
    main()
