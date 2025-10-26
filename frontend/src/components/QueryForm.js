import React, { useState } from 'react';
import './QueryForm.css';

function QueryForm({ onSubmit, loading, disabled }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit(query);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter key (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const exampleQueries = [
    "What are the latest developments in quantum computing?",
    "Who is the president of India?",
    "What are the benefits and risks of AI in healthcare?",
    "Explain the current state of electric vehicle adoption"
  ];

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery);
  };

  return (
    <div className="query-form-container">
      {/* Example queries moved above input */}
      <div className="example-queries">
        <p className="example-label">💡 Try these example queries:</p>
        <div className="example-buttons">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              disabled={disabled || loading}
              className="example-button"
              type="button"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="query-form">
        <div className="form-group">
          <label htmlFor="query">Research Query</label>
          <div className="input-with-button">
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your research question here... (Press Enter to submit)"
              rows="3"
              disabled={disabled || loading}
              className="query-input"
            />
            <button
              type="submit"
              disabled={disabled || loading || !query.trim()}
              className="inline-submit-button"
              title="Submit query"
            >
              {loading ? '⏳' : '🔍'}
            </button>
          </div>
          <p className="input-hint">Press <kbd>Enter</kbd> to submit or <kbd>Shift+Enter</kbd> for new line</p>
        </div>
      </form>
    </div>
  );
}

export default QueryForm;
