import React from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Clock,
  PieChart,
  Search,
  ShieldAlert,
  Target,
  TrendingUp,
  Zap,
} from 'lucide-react';

export default function DashboardOverview({
  deceptivePercent,
  formatDate,
  navigate,
  safePercent,
  stats,
  suspiciousPercent,
}) {
  return (
    <>
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
                  <div className="dash-dist-fill warn" style={{ width: `${suspiciousPercent}%` }} />
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
    </>
  );
}
