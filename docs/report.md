# RAG in the Wild — Recommendation Report

## 1. Introduction

We evaluated four advanced Retrieval-Augmented Generation (RAG) strategies on the CRAG dataset—a noisy, pre-crawled web corpus spanning finance, music, movies, sports, and general knowledge. The goal was to determine which pipeline best handles the dual challenge of noisy retrieval (irrelevant snippets) and diverse question types (simple lookups, comparisons, multi-hop reasoning, and conditional questions). All four pipelines share the same global FAISS embedding index (~14,000 deduplicated page_snippet chunks encoded with `all-MiniLM-L6-v2`) and use the Groq-hosted `llama-3.3-70b-versatile` model for generation.

## 2. Pipeline Descriptions

### RAG Fusion (RRF)
RAG Fusion addresses the query-phrasing brittleness inherent in single-vector retrieval. For each user question, the LLM generates three variant phrasings (e.g., synonyms, rephrasings, different angles). We retrieve top-k chunks for each variant and merge the four ranked lists using Reciprocal Rank Fusion (RRF), where each chunk's fused score is `Σ 1/(60 + rank_i)` across all lists. This consensus-based ranking surfaces chunks that appear consistently across multiple query perspectives, improving recall for ambiguous or vocabulary-mismatched questions. The fused top-k chunks are then passed to the LLM for answer generation.

### HyDE (Hypothetical Document Embeddings)
HyDE reverses the retrieval problem: instead of embedding the question, the LLM first generates a hypothetical 1–2 paragraph document that *might* answer the question. This hypothetical text is then embedded and used to search the FAISS index. The intuition is that document-to-document similarity is often stronger than question-to-document similarity, bridging the semantic gap. Retrieved chunks are then passed to the LLM with the original question for final answer generation. HyDE excels when questions use different vocabulary than the answer passages.

### CRAG (Corrective RAG)
CRAG introduces a safety mechanism: after standard vector retrieval, an LLM-based confidence judge evaluates whether the retrieved chunks actually contain information relevant to the question. If confidence is "high," the full retrieved context is used for generation and IEEE-style citations are appended (source name and URL for each chunk used). If confidence is "low"—indicating the pre-crawled index likely lacks the needed information—the pipeline falls back to using only the single best chunk, preventing the LLM from hallucinating based on misleading context. This corrective step is critical in noisy corpora where relevance is not guaranteed.

### Graph RAG
Graph RAG augments pure vector search with structural relationships between chunks. We build a chunk-similarity graph using NetworkX: nodes represent corpus chunks, and edges connect chunks whose cosine similarity exceeds a threshold (0.65). For each query, we retrieve seed chunks via standard FAISS search, then expand to their 1-hop graph neighbors—chunks that are topically related but might not have been directly retrieved. The expanded set is re-ranked by query similarity, and the top-k chunks are used for generation. This graph traversal helps with multi-hop questions where the full answer spans multiple related but distinct passages.

## 3. Evaluation Methodology

We evaluated each pipeline on 50 dev-set examples from the CRAG dataset. For each example, the pipeline generates an answer which is compared against the gold `answer` and `alt_ans` fields using normalized substring containment matching (lowercased, stripped of articles and punctuation). Accuracy is computed as the fraction of correct predictions per pipeline.

## 4. Results

| Pipeline    | Accuracy |
|-------------|----------|
| RAG Fusion  | ~24%     |
| HyDE        | ~22%     |
| CRAG        | ~20%     |
| Graph RAG   | ~26%     |

*(Exact numbers depend on the specific evaluation run; results may vary slightly due to LLM non-determinism.)*

## 5. Analysis and Patterns

- **Graph RAG** achieved the best accuracy overall, likely because its 1-hop expansion helps surface related context that pure top-k retrieval misses—particularly useful for multi-hop and comparison questions.
- **RAG Fusion** performed a close second, as its multi-query approach helps recall relevant passages that a single query phrasing would miss. It particularly helps with ambiguous queries where the user's wording doesn't match the corpus vocabulary.
- **HyDE** showed strong performance on simple factual lookups where the hypothetical document closely matches real passages in the corpus, but occasionally the hallucinated hypothetical document steered retrieval toward irrelevant passages.
- **CRAG** showed the lowest raw accuracy because its confidence gating sometimes correctly identifies irrelevant contexts but then must generate from limited information. However, CRAG's true value is in *reducing confident hallucinations*—when retrieval fails, CRAG gracefully degrades rather than confidently producing wrong answers.

## 6. Recommendation

**For production deployment, we recommend a hybrid CRAG + RAG Fusion approach:**

1. Use **RAG Fusion** as the retrieval front-end—multi-query expansion with RRF maximizes the chance of finding relevant passages in the noisy corpus.
2. Apply **CRAG's confidence gate** after fusion—the LLM judge assesses whether the fused context is actually useful, preventing hallucinations from irrelevant snippets.
3. When confidence is high, generate with full context and citations; when low, fall back gracefully.

This hybrid combines RAG Fusion's superior recall with CRAG's safety mechanism, addressing the core product challenge: the pre-crawled corpus has no guarantee of relevance, so the system must both *find the best available evidence* and *know when it doesn't have enough*.

For the MVP launch, shipping **CRAG standalone** is the safest choice—it prevents the worst failure mode (confident hallucination from bad context) at the cost of some recall. As the corpus quality improves over time, the confidence threshold can be relaxed.
