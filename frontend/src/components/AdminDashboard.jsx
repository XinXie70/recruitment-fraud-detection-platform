import React, { useCallback, useState, useEffect, useMemo } from 'react';
import {
  Award,
  BarChart3,
  Brain,
  Cpu,
  Database,
  Layers,
  Server,
  Zap,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Target,
  GitBranch,
  Gauge,
  Crosshair,
  Percent,
} from 'lucide-react';
import Navigation from './Navigation';
import MeteorBackground from './MeteorBackground';
import { apiUrl } from '../utils/api';

const MODEL_METRICS = [
  {
    model: 'Bi-LSTM',
    accuracy: 0.9857,
    precision: 0.8389,
    recall: 0.8728,
    f1: 0.8555,
    threshold: 0.67,
    category: 'dl',
  },
  {
    model: 'SVM',
    accuracy: 0.979,
    precision: 0.7816,
    recall: 0.7861,
    f1: 0.7839,
    threshold: 0.31,
    category: 'classic',
  },
  {
    model: 'XGBoost',
    accuracy: 0.9746,
    precision: 0.6864,
    recall: 0.8728,
    f1: 0.7684,
    threshold: 0.17,
    category: 'classic',
  },
  {
    model: 'DNN',
    accuracy: 0.9704,
    precision: 0.645,
    recall: 0.8613,
    f1: 0.7376,
    threshold: 0.33,
    category: 'dl',
  },
  {
    model: 'Logistic Reg.',
    accuracy: 0.9175,
    precision: 0.4626,
    recall: 0.8266,
    f1: 0.5934,
    threshold: 0.23,
    category: 'classic',
  },
  {
    model: 'RNN',
    accuracy: 0.9763,
    precision: 0.7143,
    recall: 0.8324,
    f1: 0.7689,
    threshold: 0.24,
    category: 'dl',
  },
  {
    model: 'BERT',
    accuracy: 0.99,
    precision: 0.92,
    recall: 0.89,
    f1: 0.905,
    threshold: 0.5,
    category: 'transformer',
  },
  {
    model: 'RoBERTa',
    accuracy: 0.988,
    precision: 0.91,
    recall: 0.88,
    f1: 0.895,
    threshold: 0.5,
    category: 'transformer',
  },
];

const MODEL_ARCHITECTURES = {
  'Logistic Reg.': {
    type: 'Linear Classifier',
    params: '~10K',
    desc: 'Simple linear classifier with TF-IDF features. Fast, interpretable baseline for binary classification.',
  },
  SVM: {
    type: 'Kernel SVM (RBF)',
    params: '~50K',
    desc: 'Support Vector Machine with RBF kernel on TF-IDF vectors. Excels on high-dimensional sparse text data.',
  },
  XGBoost: {
    type: 'Gradient Boosting',
    params: '~200 trees',
    desc: 'Tree-based ensemble with gradient boosting. Handles mixed feature types and provides feature importance.',
  },
  DNN: {
    type: 'Deep Neural Network',
    params: '~500K',
    desc: '3-layer fully connected network with ReLU + dropout. Learns complex non-linear patterns in job descriptions.',
  },
  RNN: {
    type: 'LSTM Network',
    params: '~300K',
    desc: 'Long Short-Term Memory network capturing sequential dependencies in text. Good at understanding word order.',
  },
  'Bi-LSTM': {
    type: 'Bidirectional LSTM',
    params: '~600K',
    desc: 'Bidirectional LSTM with attention mechanism. Processes text both forward and backward for full context.',
  },
  BERT: {
    type: 'Transformer (Encoder)',
    params: '~110M',
    desc: 'Bidirectional Encoder from Transformers. Pre-trained on BooksCorpus + Wikipedia, fine-tuned for fake job detection.',
  },
  RoBERTa: {
    type: 'Transformer (Encoder)',
    params: '~125M',
    desc: 'Robustly Optimized BERT. Trained on more data with dynamic masking, better performance on nuanced classification.',
  },
};

const CATEGORY_COLORS = {
  classic: '#c97f3d',
  dl: '#6f8067',
  transformer: '#5b7bb5',
};

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1 Score',
};

export default function AdminDashboard({ auth, onLogout }) {
  const [expandedModel, setExpandedModel] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [activeMetric, setActiveMetric] = useState('f1');
  const [viewMode, setViewMode] = useState('ranking');

  const refreshHealth = useCallback(async () => {
    setHealthStatus(null);
    try {
      const response = await fetch(apiUrl('/api/health'));
      if (!response.ok) throw new Error(`Health request failed: ${response.status}`);
      setHealthStatus(await response.json());
    } catch {
      setHealthStatus({ status: 'unreachable' });
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const formatPct = (v) => `${(v * 100).toFixed(1)}%`;
  const bestModel = useMemo(() => [...MODEL_METRICS].sort((a, b) => b.f1 - a.f1)[0], []);
  const avgF1 = useMemo(
    () => MODEL_METRICS.reduce((s, m) => s + m.f1, 0) / MODEL_METRICS.length,
    [],
  );
  const sorted = useMemo(
    () => [...MODEL_METRICS].sort((a, b) => b[activeMetric] - a[activeMetric]),
    [activeMetric],
  );
  const maxMetric = useMemo(
    () => Math.max(...MODEL_METRICS.map((m) => m[activeMetric])),
    [activeMetric],
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
      classic: MODEL_METRICS.filter((m) => m.category === 'classic'),
      dl: MODEL_METRICS.filter((m) => m.category === 'dl'),
      transformer: MODEL_METRICS.filter((m) => m.category === 'transformer'),
    }),
    [],
  );

  const categoryAverages = useMemo(() => {
    const avg = {};
    Object.entries(categories).forEach(([key, models]) => {
      avg[key] = {
        f1: models.reduce((s, m) => s + m.f1, 0) / models.length,
        accuracy: models.reduce((s, m) => s + m.accuracy, 0) / models.length,
        precision: models.reduce((s, m) => s + m.precision, 0) / models.length,
        recall: models.reduce((s, m) => s + m.recall, 0) / models.length,
      };
    });
    return avg;
  }, [categories]);

  const radarMax = 1.0;

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
              Admin & Research Dashboard
            </h1>
            <p>Comprehensive evaluation metrics for all 8 deployed ML models.</p>
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
              onClick={refreshHealth}
              title="Refresh system health"
              type="button"
            >
              <RefreshCw size={16} />
            </button>
          </div>
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
              <strong>{MODEL_METRICS.length}</strong>
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
              <strong>{bestModel.model}</strong>
              <span>Best (F1: {formatPct(bestModel.f1)})</span>
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
              <strong>{MODEL_METRICS.filter((m) => m.f1 >= 0.85).length}</strong>
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

        {/* Ranking View */}
        {viewMode === 'ranking' && (
          <section className="admin-card admin-perf">
            <div className="admin-card-header">
              <BarChart3 size={22} />
              <h2>{METRIC_LABELS[activeMetric]} Ranking</h2>
              <span className="admin-badge">
                Sorted by {METRIC_LABELS[activeMetric].toLowerCase()}
              </span>
            </div>
            <div className="admin-perf-list">
              {sorted.map((m, i) => (
                <div
                  key={m.model}
                  className={`admin-perf-row ${expandedModel === m.model ? 'expanded' : ''}`}
                  onClick={() => setExpandedModel(expandedModel === m.model ? null : m.model)}
                >
                  <div className="admin-perf-main">
                    <span
                      className="admin-rank"
                      style={{
                        background:
                          i === 0
                            ? 'var(--blue)'
                            : i === 1
                              ? '#9e9e9e'
                              : i === 2
                                ? '#c97f3d'
                                : 'transparent',
                        color: i < 3 ? '#fff' : 'var(--muted)',
                      }}
                    >
                      #{i + 1}
                    </span>
                    <span
                      className="admin-cat-dot"
                      style={{ background: CATEGORY_COLORS[m.category] }}
                    />
                    <div className="admin-perf-bar-wrap">
                      <div className="admin-perf-label">
                        <strong>{m.model}</strong>
                        <span>{formatPct(m[activeMetric])}</span>
                      </div>
                      <div className="admin-perf-track">
                        <div
                          className="admin-perf-fill"
                          style={{
                            width: `${(m[activeMetric] / maxMetric) * 100}%`,
                            background: `linear-gradient(90deg, ${CATEGORY_COLORS[m.category]}, ${CATEGORY_COLORS[m.category]}88)`,
                          }}
                        />
                      </div>
                    </div>
                    <div className="admin-perf-mini-stats">
                      <span className="admin-mini-metric">F1 {formatPct(m.f1)}</span>
                      <span className="admin-mini-metric">Acc {formatPct(m.accuracy)}</span>
                    </div>
                    {expandedModel === m.model ? (
                      <ChevronUp size={18} />
                    ) : (
                      <ChevronDown size={18} />
                    )}
                  </div>
                  {expandedModel === m.model && (
                    <div className="admin-perf-detail">
                      <div className="admin-metric-grid">
                        <div className="admin-metric">
                          <span>Accuracy</span>
                          <strong>{formatPct(m.accuracy)}</strong>
                        </div>
                        <div className="admin-metric">
                          <span>Precision</span>
                          <strong>{formatPct(m.precision)}</strong>
                        </div>
                        <div className="admin-metric">
                          <span>Recall</span>
                          <strong>{formatPct(m.recall)}</strong>
                        </div>
                        <div className="admin-metric">
                          <span>F1 Score</span>
                          <strong>{formatPct(m.f1)}</strong>
                        </div>
                        <div className="admin-metric">
                          <span>Threshold</span>
                          <strong>{m.threshold.toFixed(2)}</strong>
                        </div>
                        <div className="admin-metric">
                          <span>Category</span>
                          <strong style={{ color: CATEGORY_COLORS[m.category] }}>
                            {m.category === 'transformer'
                              ? 'Transformer'
                              : m.category === 'dl'
                                ? 'Deep Learning'
                                : 'Classic ML'}
                          </strong>
                        </div>
                      </div>
                      {MODEL_ARCHITECTURES[m.model] && (
                        <div className="admin-arch-info">
                          <div className="admin-arch-header">
                            <span className="admin-arch-type">
                              {MODEL_ARCHITECTURES[m.model].type}
                            </span>
                            <span className="admin-arch-params">
                              {MODEL_ARCHITECTURES[m.model].params} params
                            </span>
                          </div>
                          <p>{MODEL_ARCHITECTURES[m.model].desc}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Radar / Category View */}
        {viewMode === 'radar' && (
          <section className="admin-card">
            <div className="admin-card-header">
              <Crosshair size={22} />
              <h2>Model Category Performance</h2>
              <span className="admin-badge">Classic ML vs DL vs Transformer</span>
            </div>

            {Object.entries(categories).map(([catKey, models]) => (
              <div key={catKey} className="admin-category-section">
                <div className="admin-category-header">
                  <span className="admin-cat-tag" style={{ background: CATEGORY_COLORS[catKey] }}>
                    {catKey === 'transformer'
                      ? 'Transformer Models'
                      : catKey === 'dl'
                        ? 'Deep Learning'
                        : 'Classic ML'}
                  </span>
                  <span className="admin-cat-avg">
                    Avg F1: <strong>{formatPct(categoryAverages[catKey].f1)}</strong>
                  </span>
                </div>
                <div className="admin-category-grid">
                  {/* Bar chart for category metrics */}
                  <div className="admin-cat-bars">
                    {['f1', 'accuracy', 'precision', 'recall'].map((metric) => (
                      <div key={metric} className="admin-cat-bar-row">
                        <span className="admin-cat-bar-label">{METRIC_LABELS[metric]}</span>
                        <div className="admin-cat-bar-track">
                          <div
                            className="admin-cat-bar-fill"
                            style={{
                              width: `${(categoryAverages[catKey][metric] / radarMax) * 100}%`,
                              background: CATEGORY_COLORS[catKey],
                            }}
                          />
                        </div>
                        <span className="admin-cat-bar-val">
                          {formatPct(categoryAverages[catKey][metric])}
                        </span>
                      </div>
                    ))}
                  </div>
                  {/* Model list in category */}
                  <div className="admin-cat-models">
                    {models.map((m) => (
                      <div key={m.model} className="admin-cat-model-chip">
                        <strong>{m.model}</strong>
                        <span>F1: {formatPct(m.f1)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </section>
        )}

        {/* Architecture Detail View */}
        {viewMode === 'detail' && (
          <section className="admin-card">
            <div className="admin-card-header">
              <Layers size={22} />
              <h2>Architecture Overview</h2>
            </div>

            {Object.entries(categories).map(([catKey, models]) => (
              <div key={catKey} className="admin-category-section">
                <div className="admin-category-header">
                  <span className="admin-cat-tag" style={{ background: CATEGORY_COLORS[catKey] }}>
                    {catKey === 'transformer'
                      ? 'Transformer Models'
                      : catKey === 'dl'
                        ? 'Deep Learning'
                        : 'Classic ML'}
                  </span>
                </div>
                <div className="admin-arch-grid">
                  {models.map((m) => {
                    const arch = MODEL_ARCHITECTURES[m.model];
                    return (
                      <div key={m.model} className={`admin-arch-card ${catKey}`}>
                        <div className="admin-arch-card-header">
                          <strong>{m.model}</strong>
                          <span
                            className="admin-type-badge"
                            style={{
                              background: CATEGORY_COLORS[catKey] + '22',
                              color: CATEGORY_COLORS[catKey],
                              borderColor: CATEGORY_COLORS[catKey] + '44',
                            }}
                          >
                            {arch?.type || catKey}
                          </span>
                        </div>
                        <div className="admin-arch-metrics-row">
                          <div className="admin-arch-mini">
                            <Percent size={13} />
                            <span>F1</span>
                            <strong>{formatPct(m.f1)}</strong>
                          </div>
                          <div className="admin-arch-mini">
                            <Target size={13} />
                            <span>Acc</span>
                            <strong>{formatPct(m.accuracy)}</strong>
                          </div>
                          <div className="admin-arch-mini">
                            <Crosshair size={13} />
                            <span>Prec</span>
                            <strong>{formatPct(m.precision)}</strong>
                          </div>
                        </div>
                        <p className="admin-arch-desc">
                          {arch?.desc || 'No description available.'}
                        </p>
                        <div className="admin-arch-meta">
                          <span>
                            <Database size={14} /> {arch?.params || 'N/A'}
                          </span>
                          <span>
                            <Gauge size={14} /> threshold: {m.threshold.toFixed(2)}
                          </span>
                        </div>
                        <div className="admin-arch-f1-bar">
                          <div className="admin-arch-f1-track">
                            <div
                              className="admin-arch-f1-fill"
                              style={{
                                width: `${m.f1 * 100}%`,
                                background: CATEGORY_COLORS[catKey],
                              }}
                            />
                          </div>
                          <span>F1</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </section>
        )}

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
              <strong>{healthStatus?.model_ready ? '✓ All 8 models loaded' : 'Loading...'}</strong>
            </div>
            <div className="admin-system-item">
              <span>Total Parameters</span>
              <strong>~236M (across all models)</strong>
            </div>
            <div className="admin-system-item">
              <span>Framework</span>
              <strong>FastAPI + React + Vite</strong>
            </div>
            <div className="admin-system-item">
              <span>ML Libraries</span>
              <strong>scikit-learn · XGBoost · TensorFlow · PyTorch · Transformers</strong>
            </div>
            <div className="admin-system-item">
              <span>Deployment</span>
              <strong>Docker Compose (3 services)</strong>
            </div>
            <div className="admin-system-item">
              <span>Inference Strategy</span>
              <strong>Ensemble (8 models → weighted average)</strong>
            </div>
            <div className="admin-system-item">
              <span>Best Model</span>
              <strong>
                {bestModel.model} (F1: {formatPct(bestModel.f1)})
              </strong>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
