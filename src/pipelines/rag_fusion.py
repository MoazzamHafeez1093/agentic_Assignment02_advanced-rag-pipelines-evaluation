import collections
from src import generation, retrieval

def run(query: str, index, corpus_texts: list[dict], embedder, top_k: int, generator_model: str = None) -> dict:
    # 1. Generate multi-queries
    queries = generation.generate_queries(query, 3)
    queries.insert(0, query) # include original
    
    # 2. Retrieve for all queries
    list_of_ranked_lists = []
    for q in queries:
        ranked = retrieval.retrieve(index, corpus_texts, q, embedder, top_k)
        list_of_ranked_lists.append(ranked)
        
    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    fused_scores = collections.defaultdict(float)
    # dict format mapping text to metadata + fused score
    metadata_map = {}
    
    for ranked_list in list_of_ranked_lists:
        for rank, (chunk, score) in enumerate(ranked_list):
            text = chunk['text']
            metadata_map[text] = chunk
            fused_scores[text] += 1.0 / (rrf_k + rank + 1)
            
    # sort by fused score
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    top_fused = sorted_fused[:top_k]
    
    fused_passages = [metadata_map[text] for text, score in top_fused]
    fused_passage_scores = [score for text, score in top_fused]
    
    # 4. Generate answer
    answer = generation.generate(query, fused_passages)
    
    return {
        "passages": fused_passages,
        "scores": fused_passage_scores,
        "queries": queries,
        "answer": answer
    }
