import React from 'react';
import {
  AlertTriangle,
  Briefcase,
  Calendar,
  Clock,
  Eye,
  RefreshCw,
  Search,
  Trash2,
  TrendingUp,
} from 'lucide-react';

export default function DashboardHistory({
  accessToken,
  formatDate,
  handleClearHistory,
  handleViewResult,
  history,
  historyPage,
  historyTotalPages,
  isSyncing,
  loadMoreHistory,
  loadingResultId,
  navigate,
  riskLevelClass,
  riskLevelLabel,
  syncError,
  syncHistory,
}) {
  return (
    <>
      {/* History Table */}
      <section className="dash-card dash-history">
        <div className="dash-card-header">
          <Clock size={22} />
          <h2>Recent Scans</h2>
          <div className="dash-header-right">
            <button
              type="button"
              className="dash-refresh-btn"
              onClick={syncHistory}
              disabled={isSyncing || !accessToken}
              title="Refresh scan history"
              aria-label="Refresh scan history"
            >
              <RefreshCw size={16} className={isSyncing ? 'spin' : ''} />
              {isSyncing ? 'Syncing' : 'Refresh'}
            </button>
            {history.length > 0 && <span className="dash-badge">{history.length} records</span>}
            {history.some((entry) => entry.source !== 'server' || entry.analysisResult) && (
              <button
                className="dash-clear-btn"
                onClick={handleClearHistory}
                title="Clear full result details saved in this browser"
              >
                <Trash2 size={16} />
                Clear local cache
              </button>
            )}
          </div>
        </div>

        {syncError && (
          <div className="dash-sync-message" role="status">
            <AlertTriangle size={16} />
            {syncError}
          </div>
        )}

        {history.length > 0 ? (
          <>
            <div className="dash-history-table-wrap">
              <table className="dash-history-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                    <th>Verdict</th>
                    <th>Models</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((entry) => (
                    <tr key={entry.id} className="dash-history-row">
                      <td className="dash-date">
                        <Calendar size={14} />
                        {formatDate(entry.date)}
                      </td>
                      <td>
                        <span className={`dash-score-pill ${riskLevelClass(entry.riskLevel)}`}>
                          {entry.riskScore}/100
                        </span>
                      </td>
                      <td>
                        <span className={`dash-level-badge ${riskLevelClass(entry.riskLevel)}`}>
                          {riskLevelLabel(entry.riskLevel)}
                        </span>
                      </td>
                      <td className="dash-verdict">{entry.prediction || '—'}</td>
                      <td className="dash-model-count">
                        <TrendingUp size={14} />
                        {entry.modelCount ?? 2} models
                      </td>
                      <td>
                        <button
                          type="button"
                          className="dash-view-result"
                          onClick={() => handleViewResult(entry)}
                          disabled={
                            loadingResultId === entry.id ||
                            (!entry.analysisResult && !entry.hasResult)
                          }
                          title={
                            entry.analysisResult || entry.hasResult
                              ? 'Open full analysis result'
                              : 'Full result was not saved for this older scan'
                          }
                          aria-label={
                            entry.analysisResult || entry.hasResult
                              ? 'Open full analysis result'
                              : 'Full result unavailable for this older scan'
                          }
                        >
                          <Eye size={16} />
                          <span>
                            {loadingResultId === entry.id
                              ? 'Loading'
                              : entry.analysisResult || entry.hasResult
                                ? 'View'
                                : 'Unavailable'}
                          </span>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {historyPage < historyTotalPages && (
              <button
                type="button"
                className="dash-refresh-btn"
                onClick={loadMoreHistory}
                disabled={isSyncing}
              >
                {isSyncing ? 'Loading…' : 'Load more scans'}
              </button>
            )}
          </>
        ) : (
          <div className="dash-empty">
            <Briefcase size={36} />
            <p>No scans yet. Head to the Analyze page to get started.</p>
            <button className="btn-analyze" onClick={() => navigate('/analyze')}>
              <Search size={18} />
              Start Scanning
            </button>
          </div>
        )}
      </section>
    </>
  );
}
