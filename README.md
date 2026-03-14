# 🌍 RAG in the Wild — Advanced Retrieval Augmented Generation Case Study

![RAG Architecture](https://img.shields.io/badge/Architecture-RAG-blue) ![Python](https://img.shields.io/badge/Python-3.9+-green) ![React](https://img.shields.io/badge/React-18+-cyan) ![Status](https://img.shields.io/badge/Status-Complete-success)

Welcome to **RAG in the Wild**! This project is a comprehensive case study evaluating four cutting-edge Retrieval-Augmented Generation (RAG) strategies. 

Real-world datasets are rarely perfect. We are working with a **noisy, pre-crawled web corpus** from the CRAG dataset, where retrieved documents might contain the answer, or they might just be irrelevant ads or unrelated articles. This project demonstrates how different advanced retrieval algorithms handle diverse queries (comparisons, multi-hop reasoning, factual lookups) under these challenging conditions.

---

## 🚀 The 4 Pipelines Evaluated

This project implements and compares the following architectures, all pointing to a shared FAISS vector index of 3,546 document chunks:

1. **RAG Fusion (Multi-Query + Reciprocal Rank Fusion)**
   *Generates multiple variant phrasings of your question to improve recall, fetches documents for all variants, and mathematically fuses the rankings together using RRF.*
2. **HyDE (Hypothetical Document Embeddings)**
   *Asks the LLM to write a "fake" hypothetical answer first, then uses that fake document to search the vector database, bridging the vocabulary gap between questions and answers.*
3. **CRAG (Corrective RAG)** ✨ **Recommended for Production**
   *Our safest pipeline. Retrieves documents, then uses an LLM judge to evaluate "Confidence". If the documents don't contain the answer, it falls back gracefully instead of hallucinating. Also automatically generates **IEEE-style citations**!*
4. **Graph RAG (Vector + Network Expansion)**
   *Builds a massive `NetworkX` similarity graph of the entire corpus offline. At query time, it finds seed documents and expands to 1-hop graph neighbors to find hidden context.*

---

## 🛠️ Project Setup & Installation

Whether you are a recruiter looking at the code or a developer running the evaluation, follow these steps to get the project running locally.

### Prerequisites
- Python 3.9+ installed
- Node.js 18+ installed

### 1. Backend & Pipeline Setup (Python)
First, install the Python dependencies which power the embedding models (SentenceTransformers), the vector database (FAISS), and the routing framework (Flask).

```bash
# Install required Python packages
pip install -r requirements.txt
pip install tenacity # Ensures API rate-limiting is handled gracefully
```

### 2. Configure Your Environment
You will need a free API key from [Groq](https://console.groq.com/keys) to power the LLM (`llama-3.1-8b-instant`).
1. Copy the example config: `cp config/config.example.yaml config/config.yaml`
2. Open `config/config.yaml` and paste your Groq API key into the `groq_api_key` field.

### 3. Frontend Setup (React/Vite)
Navigate to the frontend directory and install the Node packages for the beautiful UI.

```bash
cd frontend
npm install
```

---

## 📊 Running the Evaluation (For the Report)

To test the pipelines against the 50-question dev set and see which performs best, run the evaluator. 
*(Note: Because of heavy API usage, this uses exponential backoff to respect Groq's rate limits and may take 15-20 minutes to complete).*

```bash
# Run from the root directory
python run_evaluation.py
```

Once finished, check the generated `eval_results.json` file for the accuracy breakdowns!

### Evaluation Results

| Pipeline    | Accuracy | Correct | Total |
|-------------|----------|---------|-------|
| HyDE        | 42.00%   | 21      | 50    |
| CRAG        | 40.00%   | 20      | 50    |
| Graph RAG   | 38.00%   | 19      | 50    |
| RAG Fusion  | 36.00%   | 18      | 50    |

**Best performer: HyDE (42%)**. See [`docs/report.md`](docs/report.md) for a detailed analysis and production recommendation.

---

## 🖥️ Running the Web Application (Interactive UI)

Want to test the pipelines yourself? We built a sleek React interface for you to query the database in real-time.

**Terminal 1 (Start the Backend API):**
```bash
python backend/app.py
```

**Terminal 2 (Start the Web Frontend):**
```bash
cd frontend
npm run dev
```
Open your browser to `http://localhost:5173` (or the port Vite provides) to start asking questions! You'll be able to see the retrieved chunks, the confidence scores, and the generated answers live.

---

## 📁 Repository Structure
```text
├── dataset/                  # Put your crag_task_1_and_2_dev_v4.jsonl here!
├── docs/                     # Schemas and the final Recommendation Report
├── src/                      
│   ├── corpus.py             # FAISS index builder and offline processing
│   ├── retrieval.py          # Vector search logic
│   ├── generation.py         # Groq LLM API wrapper (with retry logic)
│   ├── evaluation.py         # Accuracy scoring logic
│   └── pipelines/            # Implementations of CRAG, HyDE, Fusion, Graph
├── frontend/                 # React UI code
├── backend/app.py            # Flask API connecting pipelines to the frontend
└── run_evaluation.py         # Main script to evaluate all pipelines
```
