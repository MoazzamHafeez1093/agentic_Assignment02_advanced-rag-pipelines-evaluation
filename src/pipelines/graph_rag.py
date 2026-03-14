import networkx as nx
import numpy as np
from src import generation, retrieval

# Module-level graph cache
_graph = None
_graph_corpus_size = 0

def _build_similarity_graph(corpus_texts, index, embedder, threshold=0.65):
    """
    Build an offline chunk similarity graph over the entire corpus.
    Nodes = corpus chunks; edges = cosine similarity > threshold.
    This is built once and cached.
    """
    global _graph, _graph_corpus_size
    
    # Return cached graph if already built for this corpus
    if _graph is not None and _graph_corpus_size == len(corpus_texts):
        return _graph
    
    print(f"Building similarity graph over {len(corpus_texts)} chunks...")
    G = nx.Graph()
    
    # Add all chunks as nodes
    for i, chunk in enumerate(corpus_texts):
        G.add_node(i, data=chunk)
    
    # For efficiency, we use FAISS to find neighbors for each chunk
    # We search for the top-10 nearest neighbors per chunk and add edges where sim > threshold
    # To avoid O(n^2), we sample a subset for edge building
    sample_size = min(len(corpus_texts), 2000)
    sample_indices = np.random.choice(len(corpus_texts), size=sample_size, replace=False)
    
    for idx in sample_indices:
        chunk_text = corpus_texts[idx]['text']
        query_emb = retrieval.embed_query(chunk_text, embedder)
        query_emb_batch = np.expand_dims(query_emb, axis=0)
        scores, neighbors = index.search(query_emb_batch, 10)
        
        for score, neighbor_idx in zip(scores[0], neighbors[0]):
            if neighbor_idx != idx and neighbor_idx != -1 and float(score) > threshold:
                G.add_edge(int(idx), int(neighbor_idx), weight=float(score))
    
    _graph = G
    _graph_corpus_size = len(corpus_texts)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def run(query: str, index, corpus_texts: list[dict], embedder, top_k: int, generator_model: str = None) -> dict:
    """
    Graph RAG: graph-augmented retrieval over corpus.
    1. Build/load similarity graph over corpus chunks
    2. Retrieve seed chunks via standard vector search
    3. Expand to 1-hop graph neighbors
    4. Re-rank expanded set by similarity to query
    5. Generate answer from top-k expanded chunks
    """
    # 1. Build or load the similarity graph
    graph = _build_similarity_graph(corpus_texts, index, embedder)
    
    # 2. Base retrieval (seed nodes)
    ranked = retrieval.retrieve(index, corpus_texts, query, embedder, top_k)
    seed_indices = []
    seed_passages = []
    for chunk, score in ranked:
        # Find the index of this chunk in corpus_texts
        try:
            idx = corpus_texts.index(chunk)
            seed_indices.append(idx)
        except ValueError:
            pass
        seed_passages.append(chunk)
    
    # 3. Graph expansion: collect 1-hop neighbors of seed nodes
    expanded_indices = set(seed_indices)
    for seed_idx in seed_indices:
        if graph.has_node(seed_idx):
            neighbors = list(graph.neighbors(seed_idx))
            expanded_indices.update(neighbors)
    
    # 4. Re-rank the expanded set by similarity to query
    expanded_chunks = [corpus_texts[i] for i in expanded_indices if i < len(corpus_texts)]
    
    # Score each expanded chunk against the query
    query_emb = retrieval.embed_query(query, embedder)
    query_emb_batch = np.expand_dims(query_emb, axis=0)
    
    scored_chunks = []
    for chunk in expanded_chunks:
        chunk_emb = retrieval.embed_query(chunk['text'], embedder)
        sim = float(np.dot(query_emb, chunk_emb))
        scored_chunks.append((chunk, sim))
    
    # Sort by similarity and take top-k
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = scored_chunks[:top_k]
    
    final_passages = [c for c, s in top_chunks]
    final_scores = [s for c, s in top_chunks]
    
    # 5. Generate answer
    answer = generation.generate(query, final_passages)
    
    return {
        "passages": final_passages,
        "scores": final_scores,
        "graph_node_count": graph.number_of_nodes(),
        "graph_edge_count": graph.number_of_edges(),
        "expanded_from_seeds": len(seed_indices),
        "expanded_total": len(expanded_indices),
        "answer": answer
    }
