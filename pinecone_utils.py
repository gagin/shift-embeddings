# pinecone_utils.py

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import time
import csv
from datetime import datetime

def initialize_pinecone(api_key, environment, index_name):
    """Initialize Pinecone and create the index if it doesn't exist."""
    # Initialize Pinecone with the API key
    pc = Pinecone(api_key=api_key)

    # Check if index exists and create if it doesn't
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=2048,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=environment
            )
        )

    # Get and return the index
    return pc.Index(index_name)

def load_model(model_name):
    """Load the specified model and tokenizer."""
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def populate_pinecone(index, model, tokenizer, batch_size=32):
    """Populate the Pinecone index with model embeddings."""
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pinecone_upload_log_{timestamp}.csv"
    
    print("Getting embeddings...")
    vocab_size = tokenizer.vocab_size
    ids = [str(i) for i in range(vocab_size)]
    
    with open(log_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['token_id', 'token_string', 'vector', 'status', 'timestamp'])
        
        # Process in batches
        for i in tqdm(range(0, vocab_size, batch_size)):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    batch_ids = ids[i:i + batch_size]
                    
                    # Check which IDs already exist
                    existing_vectors = index.fetch(ids=batch_ids)
                    existing_ids = set(existing_vectors.vectors.keys())
                    new_ids = [id for id in batch_ids if id not in existing_ids]
                    
                    if not new_ids:
                        # Log skipped tokens
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for id in batch_ids:
                            csvwriter.writerow([
                                id,
                                tokenizer.decode([int(id)]),
                                "exists",
                                "skipped",
                                current_time
                            ])
                        break  # Move to next batch
                    
                    # Only process new tokens
                    batch_texts = [tokenizer.decode([int(id)]) for id in new_ids]
                    
                    # Generate embeddings for new tokens
                    with torch.no_grad():
                        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                        outputs = model(**inputs)
                        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                    
                    # Prepare vectors for batch upsert
                    vectors = [(id, embedding.tolist()) for id, embedding in zip(new_ids, embeddings)]
                    
                    # Batch upsert to Pinecone
                    index.upsert(vectors=vectors)
                    
                    # Log successful uploads
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for id, text, embedding in zip(new_ids, batch_texts, embeddings):
                        csvwriter.writerow([
                            id,
                            text,
                            str(embedding.tolist())[:100] + "...",
                            "success",
                            current_time
                        ])
                    
                    # Log skipped tokens
                    for id in existing_ids:
                        csvwriter.writerow([
                            id,
                            tokenizer.decode([int(id)]),
                            "exists",
                            "skipped",
                            current_time
                        ])
                        
                    csvfile.flush()
                    break
                    
                except Exception as e:
                    retry_count += 1
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Log failed attempts
                    for id in batch_ids:
                        csvwriter.writerow([
                            id,
                            tokenizer.decode([int(id)]),
                            "",
                            f"failed (attempt {retry_count}/{max_retries}): {str(e)}",
                            current_time
                        ])
                    csvfile.flush()
                    
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts for batch starting at {i}. Error: {str(e)}")
                    else:
                        print(f"Retry {retry_count}/{max_retries} for batch starting at {i}")
                        time.sleep(2)

    print(f"Process completed. Log file saved as: {log_filename}")
