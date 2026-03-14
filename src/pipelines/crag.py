from src import generation, retrieval

def run(query: str, index, corpus_texts: list[dict], embedder, top_k: int, generator_model: str = None) -> dict:
    # 1. Retrieve initial chunks
    ranked = retrieval.retrieve(index, corpus_texts, query, embedder, top_k)
    retrieved_passages = [chunk for chunk, score in ranked]
    scores = [score for chunk, score in ranked]
    
    # 2. Assess confidence
    confidence = generation.assess_confidence(query, retrieved_passages)
    
    # 3. Decision
    if confidence == "high":
        context_to_use = retrieved_passages
        action = "used_retrieval"
    else:
        # Fallback: skip retrieval context for generation (or use just top 1)
        # We will use top 1 just in case
        context_to_use = retrieved_passages[:1]
        action = "web_fallback"
        
    # 4. Generate answer
    ans_text = generation.generate(query, context_to_use)
    
    # 5. Add citations in IEEE style if chunks were used
    if context_to_use and action == "used_retrieval":
        citations = "\n\nReferences:\n"
        for i, c in enumerate(context_to_use):
            url = c.get('page_url', 'Unknown URL')
            name = c.get('page_name', 'Unknown Source')
            citations += f"[{i+1}] {name}, Available: {url}\n"
        ans_text += citations
        
    return {
        "passages": retrieved_passages,
        "scores": scores,
        "confidence": confidence,
        "action": action,
        "answer": ans_text
    }
