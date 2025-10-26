import React, { useState } from 'react';
import './ResultsDisplay.css';
import CitationsList from './CitationsList';

function ResultsDisplay({ results }) {
  const [activeTab, setActiveTab] = useState('summary');

  if (!results || !results.summary) {
    return null;
  }

  // Check if this is a guardrail violation
  const isGuardrailViolation = (summaryText) => {
    return summaryText.includes('⛔') ||
           summaryText.includes('BLOCKED BY SAFETY CONTROLS') ||
           summaryText.includes('REQUEST BLOCKED') ||
           (metadata.blocked === true);
  };

  // Parse the summary text to extract sections
  const parseSummary = (summaryText) => {
    const sections = {
      title: '',
      keyFindings: [],
      summary: '',
      raw: summaryText,
      isBlocked: isGuardrailViolation(summaryText)
    };

    // If it's a guardrail violation, don't parse - just use the full message
    if (sections.isBlocked) {
      sections.summary = summaryText;
      return sections;
    }

    const lines = summaryText.split('\n');
    let currentSection = '';

    for (let line of lines) {
      if (line.includes('Title:')) {
        sections.title = line.replace('Title:', '').trim();
      } else if (line.includes('KEY FINDINGS:')) {
        currentSection = 'findings';
      } else if (line.includes('SUMMARY:')) {
        currentSection = 'summary';
      } else if (line.includes('SOURCE CREDIBILITY ANALYSIS:') || line.includes('CITATIONS:')) {
        currentSection = 'other';
      } else if (currentSection === 'findings' && line.match(/^\d+\./)) {
        sections.keyFindings.push(line.replace(/^\d+\./, '').trim());
      } else if (currentSection === 'summary' && line.trim() && !line.includes('─')) {
        sections.summary += line + '\n';
      }
    }

    return sections;
  };

  const metadata = results.metadata || {};
  const summaryData = parseSummary(results.summary);

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>Research Results</h2>
        <div className="query-info">
          <strong>Query:</strong> {results.query}
        </div>
        <div className="metadata-badges">
          {summaryData.isBlocked ? (
            <span className="badge badge-danger">⛔ Blocked by Safety Controls</span>
          ) : (
            <>
              {metadata.used_web_search && (
                <span className="badge badge-primary">🌐 Web Search</span>
              )}
              {metadata.used_kb && (
                <span className="badge badge-info">📚 Knowledge Base</span>
              )}
              {metadata.citations_count > 0 && (
                <span className="badge badge-success">
                  📝 {metadata.citations_count} Sources
                </span>
              )}
            </>
          )}
        </div>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          📄 Summary
        </button>
        <button
          className={`tab ${activeTab === 'citations' ? 'active' : ''}`}
          onClick={() => setActiveTab('citations')}
          disabled={!results.citations || results.citations.length === 0}
        >
          📚 Citations ({results.citations?.length || 0})
        </button>
        <button
          className={`tab ${activeTab === 'raw' ? 'active' : ''}`}
          onClick={() => setActiveTab('raw')}
        >
          🔍 Full Report
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'summary' && (
          <div className="summary-section">
            {summaryData.isBlocked ? (
              // Guardrail violation - display full message
              <div className="guardrail-blocked">
                <pre style={{ whiteSpace: 'pre-wrap' }}>{summaryData.summary}</pre>
              </div>
            ) : (
              // Normal summary - parse and display sections
              <>
                {summaryData.title && (
                  <h3 className="summary-title">{summaryData.title}</h3>
                )}

                {summaryData.keyFindings.length > 0 && (
                  <div className="key-findings">
                    <h4>Key Findings</h4>
                    <ul>
                      {summaryData.keyFindings.map((finding, idx) => (
                        <li key={idx}>{finding}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {summaryData.summary && (
                  <div className="executive-summary">
                    <h4>Executive Summary</h4>
                    <p style={{ whiteSpace: 'pre-wrap' }}>{summaryData.summary.trim()}</p>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'citations' && (
          <CitationsList citations={results.citations} />
        )}

        {activeTab === 'raw' && (
          <div className="raw-output">
            <pre>{results.summary}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultsDisplay;
