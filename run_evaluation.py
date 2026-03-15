import os
import sys
import json
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data_loader, corpus, evaluation
from src.pipelines import rag_fusion, hyde, crag, graph_rag
from sentence_transformers import SentenceTransformer

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    dataset_path = config.get("dataset_path")
    embedding_model_name = config.get("embedding_model")
    top_k = config.get("top_k", 5)

    # 1. Load Data
    print("Loading dev examples for evaluation...")
    examples = list(data_loader.load_examples(dataset_path, limit=50))
    print(f"Loaded {len(examples)} evaluation examples.")

    # 2. Build or Load Index
    index_path = config.get("index_path")
    if os.path.exists(f"{index_path}.faiss"):
        print("Loading existing index...")
        index, corpus_texts = corpus.load_index(index_path)
        embedder = SentenceTransformer(embedding_model_name)
    else:
        print("Building corpus and index (this may take a while)...")
        all_examples = data_loader.load_examples(dataset_path, limit=1000)
        corpus_texts_list = corpus.build_corpus(all_examples)
        index, corpus_texts_list, _ = corpus.build_index(corpus_texts_list, embedding_model_name)
        corpus.save_index(index, corpus_texts_list, index_path)
        corpus_texts = corpus_texts_list
        embedder = SentenceTransformer(embedding_model_name)

    pipelines = {
        "RAG Fusion": rag_fusion.run,
        "HyDE": hyde.run,
        "CRAG": crag.run,
        "Graph RAG": graph_rag.run
    }

    results = {}
    all_details = {}

    for name, pipe_func in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Running Pipeline: {name}")
        print(f"{'='*60}")
        preds = []
        details = []
        for i, ex in enumerate(examples):
            query = ex["query"]
            print(f"  [{i+1}/{len(examples)}] {query[:60]}...")
            try:
                out = pipe_func(query, index, corpus_texts, embedder, top_k)
                answer = out["answer"]
                scores = out.get("scores", [])
                preds.append(answer)
                details.append({
                    "query": query,
                    "predicted": answer,
                    "gold_answer": ex["answer"],
                    "alt_ans": ex["alt_ans"],
                    "retrieval_scores": [round(s, 4) for s in scores[:3]],
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                preds.append("")
                details.append({
                    "query": query,
                    "predicted": "",
                    "gold_answer": ex["answer"],
                    "error": str(e),
                })

        acc, per_example = evaluation.evaluate_pipeline(preds, examples)
        results[name] = {
            "accuracy": round(acc * 100, 2),
            "correct": sum(per_example),
            "total": len(per_example),
        }
        all_details[name] = details
        print(f"\n  >> {name} Accuracy: {acc*100:.2f}% ({sum(per_example)}/{len(per_example)})")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Pipeline':15s} | {'Accuracy':>8s} | {'Correct':>7s} | {'Total':>5s}")
    print("-" * 45)
    for name, r in results.items():
        print(f"{name:15s} | {r['accuracy']:>7.2f}% | {r['correct']:>7d} | {r['total']:>5d}")

    # Save results to JSON
    output = {
        "summary": results,
        "details": all_details,
    }
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to eval_results.json")

    # Save plain-text summary to eval_results.txt
    lines = []
    lines.append("=" * 50)
    lines.append("EVALUATION RESULTS — RAG in the Wild")
    lines.append(f"Dev examples evaluated: {len(examples)}")
    lines.append("=" * 50)
    lines.append(f"{'Pipeline':15s} | {'Accuracy':>8s} | {'Correct':>7s} | {'Total':>5s}")
    lines.append("-" * 45)
    for name, r in results.items():
        lines.append(f"{name:15s} | {r['accuracy']:>7.2f}% | {r['correct']:>7d} | {r['total']:>5d}")
    lines.append("=" * 50)
    txt_content = "\n".join(lines) + "\n"
    with open("eval_results.txt", "w") as f:
        f.write(txt_content)
    print(f"Results saved to eval_results.txt")

if __name__ == "__main__":
    main()
