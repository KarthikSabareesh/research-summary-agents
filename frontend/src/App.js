import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import QueryForm from './components/QueryForm';
import ResultsDisplay from './components/ResultsDisplay';
import SystemStatus from './components/SystemStatus';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Check system status on mount
  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status`);
      setSystemStatus(response.data);
    } catch (err) {
      console.error('Failed to check system status:', err);
      setSystemStatus({ status: 'error', error: 'Failed to connect to backend' });
    }
  };

  const handleQuery = async (query) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/query`,
        {
          query: query,
          thread_id: `session-${Date.now()}`
        },
        {
          timeout: 120000 // 2 minute timeout for long queries
        }
      );

      if (response.data.success) {
        setResults(response.data);
      } else {
        setError(response.data.error || 'Unknown error occurred');
      }
    } catch (err) {
      console.error('Query error:', err);

      if (err.response) {
        setError(err.response.data.error || 'Server error');
      } else if (err.request) {
        setError('No response from server. Please check if the backend is running.');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 Research Summary Agents</h1>
        <p className="subtitle">AI-Powered Research Assistant</p>
        <SystemStatus status={systemStatus} onRefresh={checkSystemStatus} />
      </header>

      <main className="App-main">
        <div className="container">
          <QueryForm
            onSubmit={handleQuery}
            loading={loading}
            disabled={systemStatus?.status !== 'online'}
          />

          {error && (
            <div className="error-message">
              <h3>❌ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {loading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>🔎 Researching your query...</p>
              <p className="loading-subtext">
                This may take 30-60 seconds as we search and generate a comprehensive summary.
              </p>
            </div>
          )}

          {results && !loading && (
            <ResultsDisplay results={results} />
          )}
        </div>
      </main>

      <footer className="App-footer">
        <p>
          Powered by LangGraph • Gemini • Pinecone • Tavily
          <br />
          <small>Hybrid Search (BM25 + Semantic) • Citation Management</small>
        </p>
      </footer>
    </div>
  );
}

export default App;
