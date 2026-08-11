import React from 'react';
import {
  BarChart3,
  ChevronDown,
  ChevronUp,
  Crosshair,
  Database,
  Gauge,
  Layers,
  Percent,
  Target,
} from 'lucide-react';
import { CATEGORY_COLORS, METRIC_LABELS, MODEL_ARCHITECTURES, formatPct } from './admin/constants';

export default function AdminModelViews({
  activeMetric,
  categories,
  categoryAverages,
  expandedModel,
  maxMetric,
  onExpandedModelChange,
  sorted,
  viewMode,
}) {
  const radarMax = 1;
  return (
    <>
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
                onClick={() => onExpandedModelChange(expandedModel === m.model ? null : m.model)}
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
                  {expandedModel === m.model ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
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
                      <p className="admin-arch-desc">{arch?.desc || 'No description available.'}</p>
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
    </>
  );
}
