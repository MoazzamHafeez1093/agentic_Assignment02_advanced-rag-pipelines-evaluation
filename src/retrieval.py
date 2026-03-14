import numpy as np

def embed_query(query: str, model) -> np.ndarray:
    """
    Embed a query string using the provided sentence-transformer model.
    Normalizes the vector for cosine similarity (inner product).
    """
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return np.array(embedding, dtype=np.float32)

def retrieve(index, corpus_texts: list[dict], query: str, model, top_k: int) -> list[tuple[dict, float]]:
    """
    Retrieves the top_k most similar chunks for a given query.
    Returns a list of tuples: (chunk_metadata_dict, similarity_score).
    Metadata dict contains at least 'text', 'page_name', 'page_url'.
    """
    query_emb = embed_query(query, model)
    # query_emb shape is (d,), FAISS expects (1, d)
    query_emb_batch = np.expand_dims(query_emb, axis=0)
    
    # search index
    scores, indices = index.search(query_emb_batch, top_k)
    
    results = []
    # indices[0] will have length top_k
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(corpus_texts):
            results.append((corpus_texts[idx], float(score)))
            
    return results
