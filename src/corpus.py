import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any

def build_corpus(examples_generator) -> List[Dict[str, Any]]:
    """
    Extracts all 'page_snippet' chunks from the given examples generator.
    Returns a list of metadata dictionaries:
    [{'text': snippet, 'page_name': name, 'page_url': url}, ...]
    Deduplicates by exact text match to save index space.
    """
    corpus = []
    seen_texts = set()
    
    for ex in examples_generator:
        search_results = ex.get('search_results', [])
        for sr in search_results:
            snippet = sr.get('page_snippet') or ''
            snippet = snippet.strip()
            if not snippet:
                continue
            if snippet in seen_texts:
                continue
            
            seen_texts.add(snippet)
            corpus.append({
                'text': snippet,
                'page_name': sr.get('page_name', ''),
                'page_url': sr.get('page_url', '')
            })
            
    return corpus

def build_index(corpus: List[Dict[str, Any]], embedding_model_name: str) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]], np.ndarray]:
    """
    Embeds all chunk texts and builds a FAISS IndexFlatIP.
    Returns (faiss_index, corpus, embeddings).
    """
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    
    texts = [item['text'] for item in corpus]
    print(f"Embedding {len(texts)} chunks...")
    
    # Encode with L2 normalization so inner product equals cosine similarity
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    print(f"Built FAISS index with {index.ntotal} vectors.")
    return index, corpus, embeddings

def save_index(index: faiss.IndexFlatIP, corpus: List[Dict[str, Any]], path_prefix: str):
    """
    Saves the FAISS index and the corpus metadata to disk.
    """
    parent_dir = os.path.dirname(path_prefix)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    faiss_path = f"{path_prefix}.faiss"
    corpus_path = f"{path_prefix}_corpus.pkl"
    
    faiss.write_index(index, faiss_path)
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
        
    print(f"Saved index to {faiss_path} and corpus to {corpus_path}")

def load_index(path_prefix: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Loads the FAISS index and the corpus metadata from disk.
    """
    faiss_path = f"{path_prefix}.faiss"
    corpus_path = f"{path_prefix}_corpus.pkl"
    
    if not os.path.exists(faiss_path) or not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Index or corpus not found at prefix {path_prefix}")
        
    index = faiss.read_index(faiss_path)
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
        
    print(f"Loaded FAISS index with {index.ntotal} vectors from {faiss_path}.")
    return index, corpus
