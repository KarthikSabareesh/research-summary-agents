import React, { useState } from 'react';
import './CitationsList.css';

function CitationsList({ citations }) {
  const [copiedIndex, setCopiedIndex] = useState(null);

  if (!citations || citations.length === 0) {
    return (
      <div className="citations-empty">
        <p>No citations available. This query may have used the knowledge base instead of web search.</p>
      </div>
    );
  }

  const handleCopy = (citation, index) => {
    navigator.clipboard.writeText(citation);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const extractUrl = (citation) => {
    const lines = citation.split('\n');
    const urlLine = lines.find(line => line.trim().startsWith('http'));
    return urlLine ? urlLine.trim() : null;
  };

  return (
    <div className="citations-list">
      <div className="citations-header">
        <h3>Citations</h3>
        <p className="citations-count">
          {citations.length} source{citations.length !== 1 ? 's' : ''} cited
        </p>
      </div>

      <div className="citations-items">
        {citations.map((citation, index) => {
          const url = extractUrl(citation);

          return (
            <div key={index} className="citation-item">
              <div className="citation-header">
                <div className="citation-number">#{index + 1}</div>
              </div>

              <div className="citation-content">
                <pre>{citation}</pre>
              </div>

              <div className="citation-actions">
                {url && (
                  <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="citation-link"
                  >
                    🔗 Visit Source
                  </a>
                )}
                <button
                  onClick={() => handleCopy(citation, index)}
                  className="copy-button"
                >
                  {copiedIndex === index ? '✅ Copied!' : '📋 Copy'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default CitationsList;
