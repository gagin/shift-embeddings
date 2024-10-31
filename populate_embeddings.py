import torch
from common import MemoryMonitor, init_pinecone, load_model
import pinecone

def populate_pinecone_embeddings(chunk_size: int = 500):
    memory_monitor = MemoryMonitor()
    index_name = init_pinecone()

    # Load model
    model, _, _ = load_model()
    embedding_dim = model.config.hidden_size

    # Check if the index exists
    if index_name not in pinecone.list_indexes():
        print(f"Index '{index_name}' does not exist. Creating it now.")
        pinecone.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine"  # Ensure this matches your requirements
        )
    
    # Get index
    index = pinecone.Index(index_name)

    # Get embeddings
    embeddings = model.get_input_embeddings().weight
    total_vocab = embeddings.shape[0]

    # Upload embeddings in chunks
    for i in range(0, total_vocab, chunk_size):
        end_idx = min(i + chunk_size, total_vocab)
        chunk = embeddings[i:end_idx].detach().cpu().numpy()

        # Prepare vectors for Pinecone
        vectors = [(str(idx), vec.tolist(), {"token_id": idx}) 
                  for idx, vec in enumerate(chunk, start=i)]

        # Upsert to Pinecone
        index.upsert(vectors=vectors)

        print(f"Processed {end_idx}/{total_vocab} embeddings")
        memory_monitor.print_memory_usage(f"After chunk {i}")

        del chunk, vectors
        gc.collect()

if __name__ == "__main__":
    populate_pinecone_embeddings()