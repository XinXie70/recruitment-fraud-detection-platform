import React, { lazy, Suspense, useState, useMemo } from 'react';
import {
  Award,
  BarChart3,
  Brain,
  Cpu,
  Layers,
  Server,
  Zap,
  RefreshCw,
  Target,
  GitBranch,
  Gauge,
  Crosshair,
  Users,
  Activity,
} from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import AdminModelViews from './AdminModelViews';
import { METRIC_LABELS, formatPct } from './admin/constants';
import useAdminDashboardData from './admin/useAdminDashboardData';

const BarEChart = lazy(() => import('./charts/BarEChart'));
const PieEChart = lazy(() => import('./charts/PieEChart'));

function ChartFallback() {
  return (
    <div className="loading-state" role="status" style={{ height: 340 }}>
      Loading chart…
    </div>
  );
}

export default function AdminDashboard({ auth, onLogout }) {
  const [expandedModel, setExpandedModel] = useState(null);
  const [activeMetric, setActiveMetric] = useState('f1');
  const [viewMode, setViewMode] = useState('ranking');
  const { adminStats, healthStatus, metricsMeta, modelMetrics, refreshAll, statsError } =
    useAdminDashboardData({ accessToken: auth.access_token, onUnauthorized: onLogout });

  const bestModel = useMemo(
    () => [...modelMetrics].sort((a, b) => b.f1 - a.f1)[0] ?? null,
    [modelMetrics],
  );
  const avgF1 = useMemo(
    () =>
      modelMetrics.length
        ? modelMetrics.reduce((sum, model) => sum + model.f1, 0) / modelMetrics.length
        : 0,
    [modelMetrics],
  );
  const sorted = useMemo(
    () => [...modelMetrics].sort((a, b) => b[activeMetric] - a[activeMetric]),
    [activeMetric, modelMetrics],
  );
  const maxMetric = useMemo(
    () => Math.max(...modelMetrics.map((model) => model[activeMetric]), 1),
    [activeMetric, modelMetrics],
  );
  const healthPresentation = useMemo(() => {
    if (healthStatus?.status === 'healthy') return { className: 'safe', label: 'System Healthy' };
    if (healthStatus?.status === 'degraded') {
      return { className: 'warn', label: 'System Degraded' };
    }
    return { className: 'danger', label: 'System Offline' };
  }, [healthStatus]);

  const categories = useMemo(
    () => ({
      classic: modelMetrics.filter((model) => model.category === 'classic'),
      dl: modelMetrics.filter((model) => model.category === 'dl'),
      transformer: modelMetrics.filter((model) => model.category === 'transformer'),
    }),
    [modelMetrics],
  );

  const categoryAverages = useMemo(() => {
    const avg = {};
    Object.entries(categories).forEach(([key, models]) => {
      avg[key] = {
        f1: models.length ? models.reduce((s, m) => s + m.f1, 0) / models.length : 0,
        accuracy: models.length ? models.reduce((s, m) => s + m.accuracy, 0) / models.length : 0,
        precision: models.length ? models.reduce((s, m) => s + m.precision, 0) / models.length : 0,
        recall: models.length ? models.reduce((s, m) => s + m.recall, 0) / models.length : 0,
      };
    });
    return avg;
  }, [categories]);

  const riskChartOption = useMemo(
    () => ({
      color: ['#b85f4c', '#b98345', '#6f8067'],
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: {
        bottom: 0,
        textStyle: { color: '#67625d' },
      },
      series: [
        {
          name: 'Risk level',
          type: 'pie',
          radius: ['52%', '74%'],
          center: ['50%', '44%'],
          avoidLabelOverlap: true,
          itemStyle: { borderColor: '#fff', borderWidth: 3, borderRadius: 6 },
          label: { formatter: '{b}\n{c}', color: '#403c38', fontWeight: 600 },
          data: [
            { value: adminStats?.high_risk_count ?? 0, name: 'High' },
            { value: adminStats?.medium_risk_count ?? 0, name: 'Medium' },
            { value: adminStats?.low_risk_count ?? 0, name: 'Low' },
          ],
        },
      ],
    }),
    [adminStats],
  );
  const modelChartOption = useMemo(
    () => ({
      color: ['#5b7bb5', '#6f8067', '#b98345', '#b85f4c'],
      tooltip: { trigger: 'axis', valueFormatter: (value) => `${(value * 100).toFixed(1)}%` },
      legend: { top: 0, textStyle: { color: '#67625d' } },
      grid: { left: 48, right: 20, top: 48, bottom: 72 },
      xAxis: {
        type: 'category',
        data: modelMetrics.map((model) => model.model),
        axisLabel: { color: '#67625d', rotate: 28 },
        axisLine: { lineStyle: { color: '#d9d1c7' } },
      },
      yAxis: {
        type: 'value',
        min: 0.4,
        max: 1,
        axisLabel: { color: '#67625d', formatter: (value) => `${Math.round(value * 100)}%` },
        splitLine: { lineStyle: { color: '#ece6de' } },
      },
      series: ['accuracy', 'precision', 'recall', 'f1'].map((metric) => ({
        name: METRIC_LABELS[metric],
        type: 'bar',
        barMaxWidth: 18,
        data: modelMetrics.map((model) => model[metric]),
        emphasis: { focus: 'series' },
      })),
    }),
    [modelMetrics],
  );

  return (
    <div className="app">
      <MeteorBackground />
      <Navigation auth={auth} onLogout={onLogout} />

      <main className="app-main admin-main">
        {/* Header */}
        <section className="admin-header">
          <div>
            <h1 className="admin-title">
              <Cpu size={28} />
              Admin Analytics Dashboard
            </h1>
            <p>Live platform activity, risk distribution, and deployed model performance.</p>
          </div>
          <div className="admin-header-actions">
            {healthStatus && (
              <span className={`admin-health-pill ${healthPresentation.className}`}>
                <span className="health-dot" />
                {healthPresentation.label}
              </span>
            )}
            <button
              aria-label="Refresh system health"
              className="admin-refresh-btn"
              disabled={healthStatus === null}
              onClick={refreshAll}
              title="Refresh system health"
              type="button"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </section>

        {statsError && (
          <div className="admin-data-error" role="alert">
            {statsError}
          </div>
        )}

        <section className="admin-business-kpis" aria-label="Platform activity overview">
          <div className="admin-kpi-card">
            <div className="admin-kpi-icon" style={{ color: '#5b7bb5' }}>
              <Users size={22} />
            </div>
            <div>
              <strong>{adminStats?.total_users ?? '—'}</strong>
              <span>Total Users</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div className="admin-kpi-icon" style={{ color: '#6f8067' }}>
              <Activity size={22} />
            </div>
            <div>
              <strong>{adminStats?.total_analyses ?? '—'}</strong>
              <span>Total Analyses</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div className="admin-kpi-icon" style={{ color: '#b98345' }}>
              <Zap size={22} />
            </div>
            <div>
              <strong>{adminStats?.analyses_today ?? '—'}</strong>
              <span>Analyses Today</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div className="admin-kpi-icon" style={{ color: '#b85f4c' }}>
              <Gauge size={22} />
            </div>
            <div>
              <strong>{adminStats ? formatPct(adminStats.avg_risk_score) : '—'}</strong>
              <span>Average Risk</span>
            </div>
          </div>
        </section>

        <section className="admin-chart-grid">
          <article className="admin-card admin-chart-card">
            <div className="admin-card-header">
              <Activity size={22} />
              <h2>Live Risk Distribution</h2>
              <span className="admin-badge">{adminStats?.total_analyses ?? 0} analyses</span>
            </div>
            <Suspense fallback={<ChartFallback />}>
              <PieEChart option={riskChartOption} />
            </Suspense>
          </article>
          <article className="admin-card admin-chart-card admin-chart-card-wide">
            <div className="admin-card-header">
              <BarChart3 size={22} />
              <h2>Model Performance</h2>
              <span className="admin-badge">
                {metricsMeta
                  ? `${metricsMeta.dataset} · v${metricsMeta.version}`
                  : 'Loading metrics'}
              </span>
            </div>
            <Suspense fallback={<ChartFallback />}>
              <BarEChart option={modelChartOption} />
            </Suspense>
          </article>
        </section>

        {/* Top KPI Row */}
        <section className="admin-kpi-row">
          <div className="admin-kpi-card">
            <div
              className="admin-kpi-icon"
              style={{ background: 'rgba(91,123,181,0.12)', color: '#5b7bb5' }}
            >
              <Brain size={22} />
            </div>
            <div>
              <strong>{modelMetrics.length || '—'}</strong>
              <span>Models Deployed</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div
              className="admin-kpi-icon"
              style={{ background: 'rgba(111,128,103,0.12)', color: 'var(--safe)' }}
            >
              <Award size={22} />
            </div>
            <div>
              <strong>{bestModel?.model ?? '—'}</strong>
              <span>{bestModel ? `Best (F1: ${formatPct(bestModel.f1)})` : 'Loading metrics'}</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div
              className="admin-kpi-icon"
              style={{ background: 'rgba(232,168,111,0.12)', color: 'var(--blue-dark)' }}
            >
              <Target size={22} />
            </div>
            <div>
              <strong>{formatPct(avgF1)}</strong>
              <span>Average F1</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div
              className="admin-kpi-icon"
              style={{ background: 'rgba(184,95,76,0.10)', color: 'var(--danger)' }}
            >
              <Zap size={22} />
            </div>
            <div>
              <strong>{modelMetrics.filter((model) => model.f1 >= 0.85).length}</strong>
              <span>Models ≥ 85% F1</span>
            </div>
          </div>
          <div className="admin-kpi-card">
            <div
              className="admin-kpi-icon"
              style={{ background: 'rgba(185,131,69,0.12)', color: 'var(--warn)' }}
            >
              <GitBranch size={22} />
            </div>
            <div>
              <strong>3</strong>
              <span>Architecture Types</span>
            </div>
          </div>
        </section>

        {/* View Toggle */}
        <section className="admin-view-toggle">
          <button
            className={`admin-toggle-btn ${viewMode === 'ranking' ? 'active' : ''}`}
            onClick={() => setViewMode('ranking')}
          >
            <BarChart3 size={18} /> Performance Ranking
          </button>
          <button
            className={`admin-toggle-btn ${viewMode === 'radar' ? 'active' : ''}`}
            onClick={() => setViewMode('radar')}
          >
            <Crosshair size={18} /> Category Comparison
          </button>
          <button
            className={`admin-toggle-btn ${viewMode === 'detail' ? 'active' : ''}`}
            onClick={() => setViewMode('detail')}
          >
            <Layers size={18} /> Architecture Details
          </button>

          {/* Metric selector */}
          <div className="admin-metric-selector">
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <button
                key={key}
                className={`admin-metric-btn ${activeMetric === key ? 'active' : ''}`}
                onClick={() => setActiveMetric(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </section>

        <AdminModelViews
          activeMetric={activeMetric}
          categories={categories}
          categoryAverages={categoryAverages}
          expandedModel={expandedModel}
          maxMetric={maxMetric}
          onExpandedModelChange={setExpandedModel}
          sorted={sorted}
          viewMode={viewMode}
        />

        {/* System Info */}
        <section className="admin-card admin-system">
          <div className="admin-card-header">
            <Server size={22} />
            <h2>System Information</h2>
          </div>
          <div className="admin-system-grid">
            <div className="admin-system-item">
              <span>Backend Status</span>
              <strong className={healthStatus?.status === 'healthy' ? 'text-safe' : 'text-danger'}>
                <span
                  className="health-dot"
                  style={{ display: 'inline-block', marginRight: '6px' }}
                />
                {healthStatus?.status || 'Checking...'}
              </strong>
            </div>
            <div className="admin-system-item">
              <span>Models Ready</span>
              <strong>{healthStatus?.model_ready ? ' Models ready' : 'Loading...'}</strong>
            </div>
            <div className="admin-system-item">
              <span>Total Parameters</span>
              <strong>~110M BERT + LR gate</strong>
            </div>
            <div className="admin-system-item">
              <span>Framework</span>
              <strong>FastAPI + React + Vite</strong>
            </div>
            <div className="admin-system-item">
              <span>ML Libraries</span>
              <strong>scikit-learn · PyTorch · Transformers</strong>
            </div>
            <div className="admin-system-item">
              <span>Deployment</span>
              <strong>Docker Compose (3 services)</strong>
            </div>
            <div className="admin-system-item">
              <span>Inference Strategy</span>
              <strong>BERT primary + LR false-positive gate</strong>
            </div>
            <div className="admin-system-item">
              <span>Best Model</span>
              <strong>
                {bestModel ? `${bestModel.model} (F1: ${formatPct(bestModel.f1)})` : 'Loading...'}
              </strong>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
