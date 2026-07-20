import React from 'react';
import { Activity, AlertTriangle, ArrowLeft, BookOpen, CalendarClock, CheckCircle2, Info, Share2, ShieldAlert } from 'lucide-react';
import MeteorBackground from './MeteorBackground';
import ScoreBar from './ScoreBar';

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  svm: 'SVM',
  xgboost: 'XGBoost',
  dnn: 'Deep Neural Network',
  rnn: 'RNN',
  bilstm: 'Bi-LSTM',
  bert: 'BERT',
  roberta: 'RoBERTa',
};

const MODEL_KEYS = ['logistic_regression', 'svm', 'xgboost', 'dnn', 'rnn', 'bilstm', 'bert', 'roberta'];

function levelFromClassification(label) {
  if (label === 'Likely Deceptive') return 'high';
  if (label === 'Suspicious') return 'medium';
  return 'low';
}

function riskLabel(level) {
  if (level === 'high') return 'High Risk';
  if (level === 'medium') return 'Medium Risk';
  return 'Low Risk';
}

function combinedClassification(level) {
  if (level === 'high') return 'Likely Deceptive';
  if (level === 'medium') return 'Suspicious';
  return 'Likely Legitimate';
}

function modelScoreBars(result) {
  return MODEL_KEYS
    .filter((key) => result.models[key])
    .map((key) => ({
      key,
      title: MODEL_LABELS[key],
      score: Math.round(result.models[key].risk_score * 100),
      classification: result.models[key].classification_label,
      action: result.models[key].recommended_action,
      level: levelFromClassification(result.models[key].classification_label),
    }));
}

export default function ReportPage({ result, onBack }) {
  const riskLevel = result.riskLevel;
  const score = result.riskScore;
  const verdict = combinedClassification(riskLevel);
  const scanType = 'Text / Email Scan';
  const confidence = Math.max(score, Math.round(result.models.combined.combinedProb * 100));
  const caseId = `TXT-${String(score).padStart(3, '0')}`;
  const scoreItems = modelScoreBars(result);
  const signalCount = result.reasons.length;
  const isHigh = riskLevel === 'high';
  const isMedium = riskLevel === 'medium';

  return (
    <div className={`report-page ${riskLevel}`}>
      <MeteorBackground />
      <header className="report-topbar">
        <button type="button" className="report-back" onClick={onBack}>
          <ArrowLeft size={22} />
          <span>New Scan</span>
        </button>
        <div className={`report-status ${riskLevel}`}>
          <ShieldAlert size={18} />
          <span>{riskLabel(riskLevel)}</span>
        </div>
        <div className="report-case">
          <span>Case ID</span>
          <strong>{caseId}</strong>
          <button type="button" className="report-share" aria-label="Share report">
            <Share2 size={18} />
          </button>
        </div>
      </header>

      <main className="report-layout">
        <aside className="report-sidebar">
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
            <p>
              {isHigh
                ? 'High-confidence fraud indicators were detected.'
                : isMedium
                  ? 'Elevated signals require careful review.'
                  : 'No major danger signals were detected.'}
            </p>
            <div className="report-confidence">
              <div>
                <span>Analysis Confidence</span>
                <strong>{confidence}%</strong>
              </div>
              <div className="report-score-track">
                <div className={`report-score-fill ${riskLevel}`} style={{ width: `${confidence}%` }} />
              </div>
              <small>Based on {scoreItems.length} detection layer(s) and {signalCount} signal(s).</small>
            </div>
          </section>
        </aside>

        <section className="report-main">
          <div className="report-meta">
            <span>
              <CalendarClock size={18} />
              {new Date().toLocaleString()}
            </span>
            <span># {scanType}</span>
            <span>{caseId}</span>
          </div>

          <section className={`report-action-panel ${riskLevel}`}>
            <div className="report-action-heading">
              <div className="report-action-icon">
                {riskLevel === 'low' ? <CheckCircle2 size={24} /> : <AlertTriangle size={24} />}
              </div>
              <div>
                <h2>
                  {riskLevel === 'high'
                    ? 'Immediate Action Required'
                    : riskLevel === 'medium'
                      ? 'Review Before Proceeding'
                      : 'Proceed With Standard Caution'}
                </h2>
                <p>
                  {riskLevel === 'high'
                    ? 'Stop communication, do not send documents, money, or banking details.'
                    : riskLevel === 'medium'
                      ? 'Verify the company, domain, and contact channel before applying.'
                      : 'Continue to verify the employer through official channels.'}
                </p>
              </div>
            </div>

            <ol className="report-action-list">
              {result.reasons.slice(0, 4).map((reason, index) => (
                <li key={reason}>
                  <strong>{String(index + 1).padStart(2, '0')}</strong>
                  <span>{reason}</span>
                </li>
              ))}
            </ol>
          </section>

          <section className="report-panel">
            <div className="report-panel-header">
              <div className="section-title compact">
                <Activity size={22} />
                <h2>Model Score Breakdown</h2>
                <span className="classification-note">
                  3 classes: Likely Legitimate / Suspicious / Likely Deceptive
                </span>
              </div>
            </div>
            <div className="report-score-bars">
              {scoreItems.map((item) => (
                <ScoreBar key={item.key} item={item} />
              ))}
            </div>
          </section>

          <section className="report-panel">
            <div className="report-panel-header">
              <div className="section-title compact">
                <Info size={22} />
                <h2>Detected Risk Signals</h2>
              </div>
              <span>{result.reasons.length} findings</span>
            </div>
            <ul className="report-signal-list">
              {result.reasons.map((reason, index) => (
                <li key={reason} className={riskLevel}>
                  <div>
                    {index === 0 && <strong>Model consensus</strong>}
                    <span>{reason}</span>
                  </div>
                  <b>{index === 0 ? 'Primary' : riskLabel(riskLevel)}</b>
                </li>
              ))}
            </ul>
          </section>

          <section className="report-panel">
            <div className="section-title compact">
              <BookOpen size={22} />
              <h2>How to Stay Safe</h2>
            </div>
            <ul className="report-tips">
              {result.tips.map((tip, index) => (
                <li key={tip}>
                  <strong>{index + 1}</strong>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </section>
        </section>
      </main>
    </div>
  );
}
