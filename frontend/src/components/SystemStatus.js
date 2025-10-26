import React from 'react';
import './SystemStatus.css';

function SystemStatus({ status, onRefresh }) {
  if (!status) {
    return (
      <div className="status-banner loading">
        <span>⏳ Checking system status...</span>
      </div>
    );
  }

  const isOnline = status.status === 'online' && status.initialized;

  return (
    <div className={`status-banner ${isOnline ? 'online' : 'offline'}`}>
      <div className="status-content">
        <span className="status-indicator">
          {isOnline ? '✅ System Online' : '❌ System Offline'}
        </span>
        {status.features && (
          <div className="features-list">
            {status.features.hybrid_search && <span className="feature">🔎 Hybrid Search</span>}
            {status.features.citation_management && <span className="feature">📚 Citations</span>}
          </div>
        )}
        <button onClick={onRefresh} className="refresh-button" title="Refresh status">
          🔄
        </button>
      </div>
      {status.error && (
        <div className="status-error">
          <small>Error: {status.error}</small>
        </div>
      )}
    </div>
  );
}

export default SystemStatus;
