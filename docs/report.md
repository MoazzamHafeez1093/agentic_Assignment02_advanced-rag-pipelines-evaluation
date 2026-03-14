# RAG in the Wild — Recommendation Report

## 1. Introduction

We evaluated four advanced Retrieval-Augmented Generation (RAG) strategies on the CRAG dataset—a noisy, pre-crawled web corpus spanning finance, music, movies, sports, and general knowledge. The goal was to determine which pipeline best handles the dual challenge of noisy retrieval (irrelevant snippets) and diverse question types (simple lookups, comparisons, multi-hop reasoning, and conditional questions). All four pipelines share the same global FAISS embedding index (3,546 deduplicated page_snippet chunks encoded with `all-MiniLM-L6-v2`) and use Groq's `llama-3.1-8b-instant` model for generation. Retrieval scores (cosine similarity) are displayed alongside each retrieved chunk in both the evaluation output and the frontend interface.

## 2. Pipeline Descriptions

### RAG Fusion (RRF)
RAG Fusion addresses the query-phrasing brittleness inherent in single-vector retrieval. For each user question, the LLM generates three variant phrasings (e.g., synonyms, rephrasings, different angles). We retrieve top-k chunks for each variant and merge the ranked lists using Reciprocal Rank Fusion (RRF), where each chunk's fused score is `Σ 1/(60 + rank_i)` across all lists. This consensus-based ranking surfaces chunks that appear consistently across multiple query perspectives, improving recall for ambiguous or vocabulary-mismatched questions. The fused top-k chunks are then passed to the LLM for answer generation.

### HyDE (Hypothetical Document Embeddings)
HyDE reverses the retrieval problem: instead of embedding the question, the LLM first generates a hypothetical 1–2 paragraph document that *might* answer the question. This hypothetical text is then embedded and used to search the FAISS index. The intuition is that document-to-document similarity is often stronger than question-to-document similarity, bridging the semantic gap. Retrieved chunks are then passed to the LLM with the original question for final answer generation. HyDE excels when questions use different vocabulary than the answer passages.

### CRAG (Corrective RAG)
CRAG introduces a safety mechanism: after standard vector retrieval, an LLM-based confidence judge evaluates whether the retrieved chunks actually contain information relevant to the question. If confidence is "high," the full retrieved context is used for generation and IEEE-style citations are appended (source name and URL for each chunk used). If confidence is "low"—indicating the pre-crawled index likely lacks the needed information—the pipeline falls back to using only the single best chunk, preventing the LLM from hallucinating based on misleading context. This corrective step is critical in noisy corpora where relevance is not guaranteed.

### Graph RAG
Graph RAG augments pure vector search with structural relationships between chunks. We build a chunk-similarity graph using NetworkX: nodes represent corpus chunks (3,546 nodes), and edges connect chunks whose cosine similarity exceeds a threshold (0.65), resulting in 3,693 edges. For each query, we retrieve seed chunks via standard FAISS search, then expand to their 1-hop graph neighbors—chunks that are topically related but might not have been directly retrieved. The expanded set is re-ranked by query similarity, and the top-k chunks are used for generation. This graph traversal helps with multi-hop questions where the full answer spans multiple related but distinct passages.

## 3. Evaluation Methodology

We evaluated each pipeline on 50 dev-set examples from the CRAG dataset. For each example, the pipeline generates an answer which is compared against the gold `answer` and `alt_ans` fields using normalized substring containment matching (lowercased, stripped of articles and punctuation). Accuracy is computed as the fraction of correct predictions per pipeline. Each pipeline's retrieval scores (cosine similarity) are recorded for every query.

## 4. Results

| Pipeline    | Accuracy | Correct | Total |
|-------------|----------|---------|-------|
| HyDE        | 42.00%   | 21      | 50    |
| CRAG        | 40.00%   | 20      | 50    |
| Graph RAG   | 38.00%   | 19      | 50    |
| RAG Fusion  | 36.00%   | 18      | 50    |

## 5. Analysis and Patterns

- **HyDE** achieved the highest accuracy (42%) overall. By generating a hypothetical answer document before retrieval, HyDE bridges the vocabulary gap between how questions are phrased and how answers appear in the corpus. This is particularly effective for straightforward factual lookups (e.g., "who directed X?", "what is the longest river in Y?") where the hypothetical document closely resembles real passages and pulls in highly relevant chunks.

- **CRAG** came in second at 40%, demonstrating the value of confidence-based gating in a noisy corpus. When retrieval clearly matched the question, CRAG used the full context and produced correct, well-cited answers. When the retrieval was poor, CRAG's confidence judge correctly identified the mismatch and fell back gracefully. CRAG's true strength is qualitative: it reduces *confident hallucinations*—the worst failure mode for a user-facing product—by refusing to generate from misleading context. The IEEE-style citations also add transparency and verifiability.

- **Graph RAG** scored 38%, benefiting from 1-hop neighbor expansion on questions requiring multi-hop reasoning or context spanning multiple related passages. The similarity graph (3,546 nodes, 3,693 edges) successfully connected topically related chunks. However, the graph expansion sometimes introduced noise by pulling in tangentially related chunks that diluted the core context, slightly hurting accuracy on simpler questions.

- **RAG Fusion** scored 36%, the lowest among the four. While multi-query RRF helps with recall on ambiguous or vocabulary-mismatched queries, the three generated query variants sometimes drifted from the original intent, bringing in irrelevant chunks. On questions with precise factual answers, the additional variant queries added noise rather than value. RAG Fusion's strength is best realized on vague or broadly-worded questions where rephrasing helps.

## 6. Recommendation

**For production deployment, we recommend a hybrid HyDE + CRAG approach:**

1. Use **HyDE** as the retrieval front-end—hypothetical document generation bridges the semantic gap between user questions and corpus passages, achieving the highest raw accuracy (42%).
2. Apply **CRAG's confidence gate** after HyDE retrieval—the LLM judge assesses whether the retrieved context actually answers the question, preventing hallucinations from irrelevant snippets.
3. When confidence is high, generate with full context and IEEE-style citations for transparency; when low, fall back gracefully to a minimal-context answer.

This hybrid combines HyDE's superior retrieval accuracy with CRAG's safety mechanism, addressing the core product challenge: the pre-crawled corpus has no guarantee of relevance, so the system must both *find the best available evidence* and *know when it doesn't have enough*.

For the MVP launch, shipping **CRAG standalone** is the safest choice—it prevents the worst failure mode (confident hallucination from bad context) and provides source citations out of the box. As the team tunes the confidence threshold and adds more corpus data, layering HyDE retrieval underneath CRAG will yield both higher accuracy and maintained safety.
