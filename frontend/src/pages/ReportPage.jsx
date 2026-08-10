import React from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CalendarClock,
  CheckCircle2,
  Info,
  LayoutDashboard,
  Loader2,
  ShieldAlert,
} from 'lucide-react';
import { Link } from 'react-router';
import MeteorBackground from '../components/MeteorBackground';
import AttributionTable from '../features/xai_gentle/AttributionTable';
import ExplanationText from '../features/xai_gentle/ExplanationText';
import GentleGuidance from '../features/xai_gentle/GentleGuidance';
import ModelContributions from '../features/xai_gentle/ModelContributions';

function riskLabel(level) {
  if (level === 'high') return 'High Risk';
  if (level === 'medium') return 'Medium Risk';
  return 'Low Risk';
}

export default function ReportPage({
  result,
  onBack,
  explanationLoading = false,
  explanationError = '',
}) {
  const riskLevel = result.ensemble.risk_level;
  const score = Math.round(result.ensemble.risk_score * 100);
  const verdict = result.ensemble.classification_label;
  const scanType = 'Text / Email Scan';
  const caseId = `TXT-${String(score).padStart(3, '0')}`;
  const evidence = result.xai?.items || [];
  const usesFpGate = result.ensemble.method === 'bert_lr_fp_gate';

  return (
    <div className={`report-page ${riskLevel}`}>
      <MeteorBackground />
      <header className="report-topbar report-module-enter" style={{ '--module-order': 0 }}>
        <button type="button" className="report-back" onClick={onBack}>
          <ArrowLeft size={22} />
          <span>New Scan</span>
        </button>
        <div className={`report-status ${riskLevel}`}>
          <ShieldAlert size={18} />
          <span>{riskLabel(riskLevel)}</span>
        </div>
        <div className="report-topbar-end">
          <nav className="report-nav" aria-label="Report navigation">
            <Link to="/dashboard">
              <LayoutDashboard size={17} />
              Dashboard
            </Link>
          </nav>
          <div className="report-case">
            <span>Case ID</span>
            <strong>{caseId}</strong>
          </div>
        </div>
      </header>

      <main className="report-layout">
        <aside className="report-sidebar report-module-enter" style={{ '--module-order': 1 }}>
          <section className={`report-risk-panel ${riskLevel}`}>
            <div className="report-risk-kicker">
              <ShieldAlert size={18} />
              <span>{verdict}</span>
            </div>
            <div className="risk-dial" style={{ '--score': score }}>
              <div>
                <strong>{score}</strong>
                <span>/100</span>
              </div>
            </div>
            <div className={`report-risk-pill ${riskLevel}`}>
              <span />
              {riskLabel(riskLevel)}
            </div>
            <div className="report-confidence">
              <div>
                <span>Ensemble Risk Score</span>
                <strong>{score}%</strong>
              </div>
              <div className="report-score-track">
                <div className={`report-score-fill ${riskLevel}`} style={{ width: `${score}%` }} />
              </div>
            </div>
          </section>
        </aside>

        <section className="report-main">
          <div className="report-meta report-module-enter" style={{ '--module-order': 2 }}>
            <span>
              <CalendarClock size={18} />
              {new Date().toLocaleString()}
            </span>
            <span># {scanType}</span>
            <span>{caseId}</span>
          </div>

          <section
            className={`report-action-panel report-module-enter ${riskLevel}`}
            style={{ '--module-order': 3 }}
          >
            <div className="report-action-heading">
              <div className="report-action-icon">
                {riskLevel === 'low' ? <CheckCircle2 size={24} /> : <AlertTriangle size={24} />}
              </div>
              <div>
                <h2>{result.ensemble.recommended_action}</h2>
                <p>Guidance is based only on the ensemble result and structured XAI evidence.</p>
              </div>
            </div>
            <GentleGuidance guidance={result.gentle_ai} />
          </section>

          <details
            className="report-panel technical-details report-module-enter"
            style={{ '--module-order': 4 }}
          >
            <summary className="technical-details-summary">
              <div className="section-title compact">
                <Activity size={22} />
                <div>
                  <h2>{usesFpGate ? 'LR + BERT Technical Details' : 'Model Technical Details'}</h2>
                  <span className="classification-note">
                    {usesFpGate ? 'Scores and decision roles' : 'Scores and contributions'}
                  </span>
                </div>
              </div>

              <span className="technical-details-action">Expand details</span>
            </summary>

            <div className="technical-details-content">
              <ModelContributions members={result.member_outputs} ensemble={result.ensemble} />
            </div>
          </details>

          <section
            className="report-panel report-module-enter xai-result-module"
            style={{ '--module-order': 5 }}
          >
            <div className="report-panel-header">
              <div className="section-title compact">
                <Info size={22} />
                <h2>Why the Ensemble Produced This Score</h2>
              </div>
              <span>{result.xai.method.replaceAll('_', ' ')}</span>
            </div>
            {explanationLoading ? (
              <div className="partial-result-notice" role="status">
                <Loader2 size={20} className="spin-icon" />
                <div>
                  <strong>Risk score ready</strong>
                  <p>Preparing the detailed model-derived explanation…</p>
                  <div
                    className="xai-progress"
                    role="progressbar"
                    aria-label="Preparing XAI explanation"
                  >
                    <span className="xai-progress-indicator" />
                  </div>
                </div>
              </div>
            ) : result.xai.status === 'success' ? (
              <>
                <div className="evidence-legend" aria-label="XAI highlight legend">
                  <span className="raises_risk">Raises risk</span>
                  <span className="lowers_risk">Lowers risk</span>
                </div>
                <ExplanationText text={result.inputText} items={evidence} />
                <AttributionTable
                  items={evidence}
                  explanations={result.gentle_ai?.evidence_explanations || []}
                />
              </>
            ) : (
              <div className="partial-result-notice" role="status">
                <AlertTriangle size={20} />
                <div>
                  <strong>Explanation temporarily unavailable</strong>
                  <p>
                    {explanationError ||
                      'The ensemble risk result is still available, but the detailed explanation could not be generated. You can continue using the model scores above.'}
                  </p>
                </div>
              </div>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
