import React, { useState, useEffect } from 'react';
import './index.css';

export default function App() {
  const [query, setQuery] = useState('');
  const [pipeline, setPipeline] = useState('CRAG');
  const [pipelines, setPipelines] = useState(['CRAG', 'RAG Fusion', 'HyDE', 'Graph RAG']);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/pipelines')
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) setPipelines(data);
      })
      .catch(err => console.error("Could not load pipelines:", err));

    fetch('http://127.0.0.1:5000/api/samples')
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) setSamples(data);
      })
      .catch(err => console.error("Could not load samples:", err));
  }, []);

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch('http://127.0.0.1:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, pipeline })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Server error');
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>RAG in the Wild</h1>
        <p>A smart assistant powered by multi-strategy RAG pipelines.</p>
      </header>

      <main>
        <div className="input-section glass-panel">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a factual question (e.g. 'Who directed Inception?')"
            disabled={loading}
          />

          {samples.length > 0 && (
            <div className="samples-row">
              <label className="sample-label">Try a sample:</label>
              <select
                className="sample-select"
                value=""
                onChange={(e) => { if (e.target.value) setQuery(e.target.value); }}
                disabled={loading}
              >
                <option value="">— Choose a sample query —</option>
                {samples.map((s, i) => (
                  <option key={i} value={s}>{s.length > 70 ? s.slice(0, 70) + '...' : s}</option>
                ))}
              </select>
            </div>
          )}

          <div className="controls">
            <div className="pipeline-selector">
              {pipelines.map(p => (
                <label key={p} className={pipeline === p ? 'active' : ''}>
                  <input
                    type="radio"
                    name="pipeline"
                    value={p}
                    checked={pipeline === p}
                    onChange={() => setPipeline(p)}
                    disabled={loading}
                  />
                  {p}
                </label>
              ))}
            </div>

            <button className="run-btn" onClick={handleRun} disabled={loading || !query.trim()}>
              {loading ? (
                <span className="spinner-wrap"><span className="spinner"></span> Running...</span>
              ) : 'Run Pipeline'}
            </button>
          </div>
        </div>

        {error && <div className="error-panel">⚠ Error: {error}</div>}

        {result && (
          <div className="results-section">
            <div className="answer-panel glass-panel">
              <h2>Generated Answer</h2>
              <div className="answer-text">
                {result.answer.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>

              {result.confidence && (
                <div className={`metadata-badge ${result.confidence === 'high' ? 'badge-green' : 'badge-orange'}`}>
                  CRAG Confidence: {result.confidence}
                </div>
              )}
              {result.action && (
                <div className="metadata-badge">{result.action === 'used_retrieval' ? '✅ Used Retrieved Context' : '⚠ Fallback (Low Confidence)'}</div>
              )}
              {result.graph_node_count && (
                <div className="metadata-badge">Graph: {result.graph_node_count} nodes, {result.graph_edge_count} edges</div>
              )}
              {result.expanded_total && (
                <div className="metadata-badge">Expanded from {result.expanded_from_seeds} seeds → {result.expanded_total} chunks</div>
              )}
            </div>

            {result.hypothetical_doc && (
              <div className="meta-panel glass-panel">
                <h3>🧪 HyDE — Hypothetical Document</h3>
                <p className="hypothetical-text">{result.hypothetical_doc}</p>
              </div>
            )}

            {result.queries && (
              <div className="meta-panel glass-panel">
                <h3>🔀 RAG Fusion — Query Variants</h3>
                <ul className="fused-queries">
                  {result.queries.map((q, i) => <li key={i}>{q}</li>)}
                </ul>
              </div>
            )}

            <div className="chunks-panel glass-panel">
              <h3>📄 Retrieved Context ({result.passages?.length || 0} chunks)</h3>
              <div className="chunks-grid">
                {result.passages && result.passages.map((chunk, idx) => (
                  <div key={idx} className="chunk-card">
                    <div className="chunk-header">
                      <span className="domain">{chunk.page_name || 'Source'}</span>
                      {result.scores && <span className="score">Score: {typeof result.scores[idx] === 'number' ? result.scores[idx].toFixed(4) : result.scores[idx]}</span>}
                    </div>
                    <p className="chunk-text">{chunk.text}</p>
                    {chunk.page_url && (
                      <a href={chunk.page_url} target="_blank" rel="noopener noreferrer" className="chunk-url">
                        🔗 {chunk.page_url}
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer>
        <p>RAG Case Study — CRAG Dataset | Powered by Groq + FAISS + Sentence Transformers</p>
      </footer>
    </div>
  );
}
