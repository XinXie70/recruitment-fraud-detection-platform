import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import { Search } from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import { apiUrl } from '../utils/api';
import DashboardHistory from './dashboard/DashboardHistory';
import DashboardOverview from './dashboard/DashboardOverview';
import {
  HISTORY_KEY,
  HISTORY_PAGE_SIZE,
  loadHistory,
  mergeHistory,
  normalizeServerHistory,
} from './dashboard/history';

export default function DashboardPage({ auth, onLogout }) {
  const navigate = useNavigate();
  const accessToken = auth?.access_token;
  const [history, setHistory] = useState(loadHistory);
  const [syncError, setSyncError] = useState('');
  const [isSyncing, setIsSyncing] = useState(false);
  const [loadingResultId, setLoadingResultId] = useState(null);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotalPages, setHistoryTotalPages] = useState(1);

  const syncHistory = useCallback(async () => {
    if (!accessToken) return;

    setIsSyncing(true);
    setSyncError('');
    const localHistory = loadHistory();

    try {
      const response = await fetch(
        apiUrl(`/api/v1/history?page=1&page_size=${HISTORY_PAGE_SIZE}`),
        {
          headers: { Authorization: `Bearer ${accessToken}` },
        },
      );

      if (response.status === 401) {
        onLogout?.();
        return;
      }
      if (!response.ok) throw new Error(`History request failed (${response.status})`);

      const payload = await response.json();
      setHistory(mergeHistory(normalizeServerHistory(payload.items || []), localHistory));
      setHistoryPage(1);
      setHistoryTotalPages(payload.total_pages || 1);
    } catch {
      setHistory(localHistory);
      setSyncError('Could not sync history. Showing results saved in this browser.');
    } finally {
      setIsSyncing(false);
    }
  }, [accessToken, onLogout]);

  const loadMoreHistory = async () => {
    if (!accessToken || isSyncing || historyPage >= historyTotalPages) return;
    const nextPage = historyPage + 1;
    setIsSyncing(true);
    setSyncError('');
    try {
      const response = await fetch(
        apiUrl(`/api/v1/history?page=${nextPage}&page_size=${HISTORY_PAGE_SIZE}`),
        { headers: { Authorization: `Bearer ${accessToken}` } },
      );
      if (response.status === 401) {
        onLogout?.();
        return;
      }
      if (!response.ok) throw new Error(`History request failed (${response.status})`);
      const payload = await response.json();
      const nextEntries = normalizeServerHistory(payload.items || []);
      setHistory((current) => {
        const existingIds = new Set(current.map((entry) => entry.id));
        return [...current, ...nextEntries.filter((entry) => !existingIds.has(entry.id))].sort(
          (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime(),
        );
      });
      setHistoryPage(nextPage);
      setHistoryTotalPages(payload.total_pages || nextPage);
    } catch {
      setSyncError('Could not load more history. Please try again.');
    } finally {
      setIsSyncing(false);
    }
  };

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
        .map(({ analysisResult: _analysisResult, inputText: _inputText, ...entry }) => entry),
    );
  };

  const openAnalysisResult = (analysisResult) => {
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

        <DashboardOverview
          deceptivePercent={deceptivePercent}
          formatDate={formatDate}
          navigate={navigate}
          safePercent={safePercent}
          stats={stats}
          suspiciousPercent={suspiciousPercent}
        />

        <DashboardHistory
          accessToken={accessToken}
          formatDate={formatDate}
          handleClearHistory={handleClearHistory}
          handleViewResult={handleViewResult}
          history={history}
          historyPage={historyPage}
          historyTotalPages={historyTotalPages}
          isSyncing={isSyncing}
          loadMoreHistory={loadMoreHistory}
          loadingResultId={loadingResultId}
          navigate={navigate}
          riskLevelClass={riskLevelClass}
          riskLevelLabel={riskLevelLabel}
          syncError={syncError}
          syncHistory={syncHistory}
        />
      </main>
    </div>
  );
}
