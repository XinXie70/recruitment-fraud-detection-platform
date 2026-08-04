import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BookOpen,
  Briefcase,
  Calendar,
  CheckCircle2,
  Clock,
  Search,
  ShieldAlert,
  TrendingUp,
  Zap,
  Trash2,
  PieChart,
  Target,
  Eye,
  RefreshCw,
} from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import { apiUrl } from '../utils/api';

const HISTORY_KEY = 'fake_job_history';

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function verdictForRisk(level) {
  if (level === 'high') return 'Likely Deceptive';
  if (level === 'medium') return 'Suspicious';
  return 'Likely Legitimate';
}

function normalizeServerHistory(items) {
  return items.map((entry) => ({
    id: `server-${entry.id}`,
    serverId: entry.id,
    source: 'server',
    date: entry.created_at,
    inputText: entry.input_preview,
    riskLevel: entry.risk_level,
    riskScore: Math.round(Number(entry.risk_score || 0) * 100),
    prediction: verdictForRisk(entry.risk_level),
    modelCount: entry.ensemble_available,
    modelTotal: entry.ensemble_total,
    status: entry.status,
    hasResult: entry.has_result,
  }));
}

function mergeHistory(serverEntries, localEntries) {
  const unusedLocal = [...localEntries];
  const mergedServer = serverEntries.map((serverEntry) => {
    const serverTime = new Date(serverEntry.date).getTime();
    const matchIndex = unusedLocal.findIndex((localEntry) => {
      const localTime = new Date(localEntry.date).getTime();
      return (
        localEntry.riskLevel === serverEntry.riskLevel &&
        Number(localEntry.riskScore) === serverEntry.riskScore &&
        Number.isFinite(serverTime) &&
        Number.isFinite(localTime) &&
        Math.abs(serverTime - localTime) <= 120_000
      );
    });

    if (matchIndex < 0) return serverEntry;
    const [localMatch] = unusedLocal.splice(matchIndex, 1);
    return { ...localMatch, ...serverEntry, analysisResult: localMatch.analysisResult };
  });

  return [...mergedServer, ...unusedLocal]
    .map((entry) => ({ ...entry, source: entry.source || 'local' }))
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export default function DashboardPage({ auth, onLogout }) {
  const navigate = useNavigate();
  const accessToken = auth?.access_token;
  const [history, setHistory] = useState(loadHistory);
  const [syncError, setSyncError] = useState('');
  const [isSyncing, setIsSyncing] = useState(false);
  const [loadingResultId, setLoadingResultId] = useState(null);

  const syncHistory = useCallback(async () => {
    if (!accessToken) return;

    setIsSyncing(true);
    setSyncError('');
    const localHistory = loadHistory();

    try {
      const response = await fetch(apiUrl('/api/v1/history?page=1&page_size=100'), {
        headers: { Authorization: `Bearer ${accessToken}` },
      });

      if (response.status === 401) {
        onLogout?.();
        return;
      }
      if (!response.ok) throw new Error(`History request failed (${response.status})`);

      const payload = await response.json();
      setHistory(mergeHistory(normalizeServerHistory(payload.items || []), localHistory));
    } catch {
      setHistory(localHistory);
      setSyncError('Could not sync history. Showing results saved in this browser.');
    } finally {
      setIsSyncing(false);
    }
  }, [accessToken, onLogout]);

  const stats = useMemo(() => {
    const h = history;
    const high = h.filter((e) => e.riskLevel === 'high').length;
    const medium = h.filter((e) => e.riskLevel === 'medium').length;
    const low = h.filter((e) => e.riskLevel === 'low').length;
    const avgScore =
      h.length > 0 ? Math.round(h.reduce((s, e) => s + (e.riskScore || 0), 0) / h.length) : 0;
    const lastScan = h.length > 0 ? h[0].date : null;
    return { total: h.length, high, medium, low, avgScore, lastScan };
  }, [history]);

  useEffect(() => {
    syncHistory();
  }, [syncHistory]);
  const riskLevelLabel = (level) => {
    if (level === 'high') return 'High Risk';
    if (level === 'medium') return 'Medium Risk';
    return 'Low Risk';
  };

  const riskLevelClass = (level) => {
    if (level === 'high') return 'danger';
    if (level === 'medium') return 'warn';
    return 'safe';
  };

  const formatDate = (iso) => {
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return 'Unknown date';
    }
  };

  const safePercent = stats.total > 0 ? Math.round((stats.low / stats.total) * 100) : 0;
  const suspiciousPercent = stats.total > 0 ? Math.round((stats.medium / stats.total) * 100) : 0;
  const deceptivePercent = stats.total > 0 ? Math.round((stats.high / stats.total) * 100) : 0;

  const handleClearHistory = () => {
    localStorage.removeItem(HISTORY_KEY);
    setHistory((current) =>
      current
        .filter((entry) => entry.source === 'server')
        .map(({ analysisResult: _analysisResult, ...entry }) => entry),
    );
  };

  const openAnalysisResult = (analysisResult) => {
    window.sessionStorage.setItem('fake_job_last_analysis', JSON.stringify(analysisResult));
    navigate('/analyze', { state: { analysisResult } });
  };

  const handleViewResult = async (entry) => {
    if (entry.analysisResult) {
      openAnalysisResult(entry.analysisResult);
      return;
    }
    if (!entry.serverId || !entry.hasResult || !accessToken) return;

    setLoadingResultId(entry.id);
    try {
      const response = await fetch(apiUrl(`/api/v1/history/${entry.serverId}`), {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (response.status === 401) {
        onLogout?.();
        return;
      }
      if (!response.ok) throw new Error(`History detail request failed (${response.status})`);
      const payload = await response.json();
      openAnalysisResult(payload.analysis_result);
    } catch {
      setHistory((current) =>
        current.map((item) => (item.id === entry.id ? { ...item, hasResult: false } : item)),
      );
      setSyncError('This full analysis result is no longer available.');
    } finally {
      setLoadingResultId(null);
    }
  };

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />

      <main className="app-main dashboard-main">
        {/* Welcome Header */}
        <section className="dashboard-header">
          <div>
            <h1 className="dashboard-title">
              Welcome back, <span>{auth?.user?.username || 'User'}</span>
            </h1>
            <p>Your job scan dashboard — monitor risks and stay protected.</p>
          </div>
          <div className="dashboard-header-actions">
            <button className="btn-analyze" onClick={() => navigate('/analyze')}>
              <Search size={20} />
              New Scan
            </button>
          </div>
        </section>

        {/* Stats Cards */}
        <section className="dashboard-stats">
          <div className="dash-stat-card primary">
            <div className="dash-stat-icon-box">
              <Activity size={24} />
            </div>
            <div className="dash-stat-body">
              <strong>{stats.total}</strong>
              <span>Total Scans</span>
            </div>
            <div className="dash-stat-trend">
              <TrendingUp size={16} />
            </div>
          </div>

          <div className="dash-stat-card safe">
            <div className="dash-stat-icon-box">
              <CheckCircle2 size={24} />
            </div>
            <div className="dash-stat-body">
              <strong>{stats.low}</strong>
              <span>Legitimate</span>
            </div>
            <div className="dash-stat-sub">{safePercent}%</div>
          </div>

          <div className="dash-stat-card warn">
            <div className="dash-stat-icon-box">
              <AlertTriangle size={24} />
            </div>
            <div className="dash-stat-body">
              <strong>{stats.medium}</strong>
              <span>Suspicious</span>
            </div>
            <div className="dash-stat-sub">{suspiciousPercent}%</div>
          </div>

          <div className="dash-stat-card danger">
            <div className="dash-stat-icon-box">
              <ShieldAlert size={24} />
            </div>
            <div className="dash-stat-body">
              <strong>{stats.high}</strong>
              <span>Deceptive</span>
            </div>
            <div className="dash-stat-sub">{deceptivePercent}%</div>
          </div>
        </section>

        {/* Main Grid: Distribution + Donut + Quick Actions */}
        <section className="dashboard-grid">
          {/* Left: Risk Distribution */}
          <div className="dash-card">
            <div className="dash-card-header">
              <BarChart3 size={22} />
              <h2>Risk Distribution</h2>
              {stats.total > 0 && <span className="dash-badge">{stats.total} total</span>}
            </div>
            {stats.total > 0 ? (
              <div className="dash-distribution">
                <div className="dash-dist-bar-h">
                  <div className="dash-dist-bar-label">
                    <span>Safe</span>
                    <span>{safePercent}%</span>
                  </div>
                  <div className="dash-dist-track">
                    <div className="dash-dist-fill safe" style={{ width: `${safePercent}%` }} />
                  </div>
                </div>
                <div className="dash-dist-bar-h">
                  <div className="dash-dist-bar-label">
                    <span>Suspicious</span>
                    <span>{suspiciousPercent}%</span>
                  </div>
                  <div className="dash-dist-track">
                    <div
                      className="dash-dist-fill warn"
                      style={{ width: `${suspiciousPercent}%` }}
                    />
                  </div>
                </div>
                <div className="dash-dist-bar-h">
                  <div className="dash-dist-bar-label">
                    <span>Deceptive</span>
                    <span>{deceptivePercent}%</span>
                  </div>
                  <div className="dash-dist-track">
                    <div
                      className="dash-dist-fill danger"
                      style={{ width: `${deceptivePercent}%` }}
                    />
                  </div>
                </div>
                <div className="dash-avg-line">
                  <Target size={14} />
                  <span>
                    Average Risk Score: <strong>{stats.avgScore}/100</strong>
                  </span>
                </div>
              </div>
            ) : (
              <div className="dash-empty">
                <BarChart3 size={36} />
                <p>No scan data yet. Run your first analysis to see statistics.</p>
              </div>
            )}
          </div>

          {/* Middle: Donut + Quick Info */}
          <div className="dash-card">
            <div className="dash-card-header">
              <PieChart size={22} />
              <h2>Overview</h2>
            </div>
            {stats.total > 0 ? (
              <div className="dash-overview">
                <div className="dash-donut-wrap">
                  <svg viewBox="0 0 100 100" className="dash-donut">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#ece7e1" strokeWidth="12" />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="var(--safe)"
                      strokeWidth="12"
                      strokeDasharray={`${(stats.low / stats.total) * 251.3} 251.3`}
                      strokeDashoffset="0"
                      transform="rotate(-90 50 50)"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="var(--warn)"
                      strokeWidth="12"
                      strokeDasharray={`${(stats.medium / stats.total) * 251.3} 251.3`}
                      strokeDashoffset={`${-(stats.low / stats.total) * 251.3}`}
                      transform="rotate(-90 50 50)"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      fill="none"
                      stroke="var(--danger)"
                      strokeWidth="12"
                      strokeDasharray={`${(stats.high / stats.total) * 251.3} 251.3`}
                      strokeDashoffset={`${-((stats.low + stats.medium) / stats.total) * 251.3}`}
                      transform="rotate(-90 50 50)"
                    />
                    <text x="50" y="46" textAnchor="middle" className="dash-donut-num">
                      {stats.total}
                    </text>
                    <text x="50" y="62" textAnchor="middle" className="dash-donut-label">
                      scans
                    </text>
                  </svg>
                </div>
                <div className="dash-donut-legend">
                  <span>
                    <i className="dot safe" /> Legitimate <b>{stats.low}</b>
                  </span>
                  <span>
                    <i className="dot warn" /> Suspicious <b>{stats.medium}</b>
                  </span>
                  <span>
                    <i className="dot danger" /> Deceptive <b>{stats.high}</b>
                  </span>
                </div>
              </div>
            ) : (
              <div className="dash-empty">
                <PieChart size={36} />
                <p>Complete a scan to see your overview.</p>
              </div>
            )}
          </div>
        </section>

        {/* Quick Actions + Recent Scan Info */}
        <section className="dashboard-grid">
          <div className="dash-card">
            <div className="dash-card-header">
              <Zap size={22} />
              <h2>Quick Actions</h2>
            </div>
            <div className="dash-actions">
              <button className="dash-action-btn" onClick={() => navigate('/analyze')}>
                <Search size={20} />
                <div>
                  <strong>Analyze a Job</strong>
                  <span>Paste a job ad and get instant results</span>
                </div>
                <ArrowRight size={18} />
              </button>
              <button className="dash-action-btn" onClick={() => navigate('/education')}>
                <BookOpen size={20} />
                <div>
                  <strong>Education Centre</strong>
                  <span>Educational resources & red flags guide</span>
                </div>
                <ArrowRight size={18} />
              </button>
            </div>
          </div>

          <div className="dash-card dash-insight">
            <div className="dash-card-header">
              <Target size={22} />
              <h2>Scan Insight</h2>
            </div>
            {stats.total > 0 ? (
              <div className="dash-insight-body">
                <div className="dash-insight-row">
                  <Clock size={16} />
                  <span>
                    Last scan: <strong>{formatDate(stats.lastScan)}</strong>
                  </span>
                </div>
                <div className="dash-insight-row">
                  <Activity size={16} />
                  <span>
                    Average risk score: <strong>{stats.avgScore}/100</strong>
                  </span>
                </div>
                <div className="dash-insight-row">
                  {deceptivePercent > 40 ? (
                    <>
                      <ShieldAlert size={16} color="var(--danger)" />
                      <span className="text-danger">High scam detection rate — stay vigilant!</span>
                    </>
                  ) : deceptivePercent > 15 ? (
                    <>
                      <AlertTriangle size={16} color="var(--warn)" />
                      <span className="text-warn">Moderate risk detected in your scans.</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={16} color="var(--safe)" />
                      <span className="text-safe">Most of your scans appear legitimate.</span>
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div className="dash-empty" style={{ padding: '28px 24px' }}>
                <Target size={28} />
                <p>Insights will appear after your first scan.</p>
              </div>
            )}
          </div>
        </section>

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
                    {history.slice(0, 15).map((entry) => (
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
                          {entry.modelCount ?? 8} models
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
      </main>
    </div>
  );
}
