import os
import sys
import yaml
import traceback

# Add parent directory to path to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

from src import corpus, data_loader
from src.pipelines import rag_fusion, hyde, crag, graph_rag

app = Flask(__name__)
CORS(app)

# Global state
index = None
corpus_texts = None
embedder = None
config = {}
top_k = 5
sample_queries = []

pipelines = {
    "RAG Fusion": rag_fusion.run,
    "HyDE": hyde.run,
    "CRAG": crag.run,
    "Graph RAG": graph_rag.run
}

def init_system():
    global index, corpus_texts, embedder, config, top_k, sample_queries
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    top_k = config.get("top_k", 5)
    embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    
    index_path = config.get("index_path", "dataset/crag_index")
    index_path = os.path.join(os.path.dirname(__file__), "..", index_path)
    
    if os.path.exists(f"{index_path}.faiss"):
        print("Loading existing index for API...")
        index, corpus_texts = corpus.load_index(index_path)
    else:
        print("Warning: Index not found. Please run run_evaluation.py first to build the index.")
        index, corpus_texts = None, []
        
    embedder = SentenceTransformer(embedding_model_name)
    
    # Load sample queries from dataset
    dataset_path = config.get("dataset_path", "dataset/crag_task_1_and_2_dev_v4.jsonl")
    ds_path = os.path.join(os.path.dirname(__file__), "..", dataset_path)
    try:
        for ex in data_loader.load_examples(ds_path, limit=10):
            sample_queries.append(ex["query"])
        print(f"Loaded {len(sample_queries)} sample queries.")
    except Exception as e:
        print(f"Could not load sample queries: {e}")

@app.route("/api/pipelines", methods=["GET"])
def get_pipelines():
    return jsonify(list(pipelines.keys()))

@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(sample_queries)

@app.route("/api/query", methods=["POST"])
def query_pipeline():
    data = request.json
    query = data.get("query")
    pipeline_name = data.get("pipeline", "CRAG")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    if pipeline_name not in pipelines:
        return jsonify({"error": f"Unknown pipeline {pipeline_name}"}), 400
    if index is None:
        return jsonify({"error": "Index not built yet. Run run_evaluation.py first."}), 500
        
    try:
        pipe_func = pipelines[pipeline_name]
        result = pipe_func(query, index, corpus_texts, embedder, top_k)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_system()
    app.run(port=5000, debug=True, use_reloader=False)

