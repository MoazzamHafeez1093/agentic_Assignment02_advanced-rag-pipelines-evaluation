from src import generation, retrieval

def run(query: str, index, corpus_texts: list[dict], embedder, top_k: int, generator_model: str = None) -> dict:
    # 1. Generate hypothetical doc
    hypothetical_doc = generation.generate_hypothetical_doc(query)
    
    # 2. Retrieve top-k chunks similar to hypothetical doc
    ranked = retrieval.retrieve(index, corpus_texts, hypothetical_doc, embedder, top_k)
    
    retrieved_passages = [chunk for chunk, score in ranked]
    scores = [score for chunk, score in ranked]
    
    # 3. Generate answer using these chunks
    answer = generation.generate(query, retrieved_passages)
    
    return {
        "hypothetical_doc": hypothetical_doc,
        "passages": retrieved_passages,
        "scores": scores,
        "answer": answer
    }
