import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  BookOpen,
  Briefcase,
  CheckCircle2,
  Gift,
  Info,
  Loader2,
  Search,
  ShieldAlert,
  X,
} from 'lucide-react';
import {
  SAMPLES,
  SAFETY_TIPS,
  combineModelScores,
  buildReasons,
} from './utils/analysisUtils';
import './App.css';

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  dnn: 'Deep Neural Network',
};

const CLASSIFICATION_OPTIONS = [
  'Likely Legitimate',
  'Suspicious',
  'Likely Deceptive',
];

function getRiskLevel(score) {
  if (score >= 60) return 'high';
  if (score >= 30) return 'medium';
  return 'low';
}

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

function classificationDescription(label) {
  if (label === 'Likely Deceptive') {
    return 'This listing has a high fraud risk and should be treated as a likely deceptive job advertisement.';
  }
  if (label === 'Suspicious') {
    return 'This listing has mixed or elevated risk signals and should be reviewed carefully before applying.';
  }
  return 'This listing has a low fraud risk and is broadly consistent with legitimate job advertisements.';
}

function modelScoreCards(models) {
  return [
    {
      key: 'logistic_regression',
      title: MODEL_LABELS.logistic_regression,
      score: Math.round(models.lr.risk_score * 100),
      classification: models.lr.classification_label,
      action: models.lr.recommended_action,
    },
    {
      key: 'dnn',
      title: MODEL_LABELS.dnn,
      score: Math.round(models.dnn.risk_score * 100),
      classification: models.dnn.classification_label,
      action: models.dnn.recommended_action,
    },
  ].map((model) => ({
    ...model,
    level: levelFromClassification(model.classification),
  }));
}

function MeteorBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrame;
    let meteors = [];

    const createMeteor = (width, height) => ({
      x: Math.random() * width,
      y: Math.random() * height - height,
      length: 90 + Math.random() * 160,
      speed: 0.55 + Math.random() * 1.25,
      drift: -0.12 + Math.random() * 0.32,
      alpha: 0.06 + Math.random() * 0.16,
      width: Math.random() > 0.78 ? 2 : 1,
    });

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      meteors = Array.from({ length: Math.max(18, Math.floor(width / 70)) }, () =>
        createMeteor(width, height),
      );
    };

    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      ctx.clearRect(0, 0, width, height);

      meteors.forEach((meteor, index) => {
        const endX = meteor.x + meteor.drift * meteor.length;
        const endY = meteor.y + meteor.length;
        const gradient = ctx.createLinearGradient(meteor.x, meteor.y, endX, endY);
        gradient.addColorStop(0, `rgba(201, 127, 61, ${meteor.alpha * 0.28})`);
        gradient.addColorStop(0.42, `rgba(201, 127, 61, ${meteor.alpha * 0.62})`);
        gradient.addColorStop(0.82, `rgba(201, 127, 61, ${meteor.alpha})`);
        gradient.addColorStop(1, 'rgba(201, 127, 61, 0)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = meteor.width;
        ctx.beginPath();
        ctx.moveTo(meteor.x, meteor.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        meteor.x += meteor.drift;
        meteor.y += meteor.speed;

        if (meteor.y > height + meteor.length) {
          meteors[index] = createMeteor(width, height);
          meteors[index].y = -meteor.length;
        }
      });

      animationFrame = window.requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return <canvas className="meteor-canvas" ref={canvasRef} aria-hidden="true" />;
}

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      const combined = combineModelScores(data.logistic_regression, data.dnn);

      setResult({
        prediction: combined.prediction,
        riskScore: combined.riskScore,
        riskLevel: combined.riskLevel,
        reasons: buildReasons(text, data, combined),
        tips: SAFETY_TIPS,
        models: {
          lr: data.logistic_regression,
          dnn: data.dnn,
          combined,
        },
      });
    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred while contacting the server.');
    } finally {
      setLoading(false);
    }
  };

  const handleSample = (sampleText) => {
    setText(sampleText);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  const isFake = result?.prediction === 'fake';
  const resultClass = result ? combinedClassification(result.riskLevel) : null;
  const scoreCards = result ? modelScoreCards(result.models) : [];

  return (
    <div className="app">
      <MeteorBackground />
      <nav className="app-nav">
        <div className="app-nav-inner">
          <div className="nav-brand">
            <div className="nav-logo">
              <ShieldAlert size={22} />
            </div>
            <span className="nav-brand-text">FakeJobDetect</span>
          </div>
          <div className="nav-badge">
            <Briefcase size={18} />
            <span>Job Safety Analyzer</span>
          </div>
        </div>
      </nav>

      <main className="app-main">
        <section className="hero">
          <h1>Detect Fake Job Advertisements</h1>
          <p>
            Paste any job listing below. Our analyzer scores it with two machine
            learning models and highlights the risk signals.
          </p>
        </section>

        <section className="input-card">
          <label className="input-label" htmlFor="jobText">
            Job Advertisement Text
          </label>
          <textarea
            id="jobText"
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={8}
            placeholder="Paste the full job advertisement here..."
            disabled={loading}
          />
          <div className="input-actions">
            <button
              id="btn-analyze"
              type="button"
              className="btn-analyze"
              onClick={handleAnalyze}
              disabled={!text.trim() || loading}
            >
              {loading ? <Loader2 size={22} className="spin-icon" /> : <Search size={22} />}
              {loading ? 'Analyzing' : 'Analyze'}
            </button>

            <button
              type="button"
              className="btn-sample"
              onClick={() => handleSample(SAMPLES[1].text)}
              disabled={loading}
            >
              Load fake sample
            </button>
            <button
              type="button"
              className="btn-sample"
              onClick={() => handleSample(SAMPLES[0].text)}
              disabled={loading}
            >
              Load legit sample
            </button>

            {text && (
              <button type="button" className="btn-clear" onClick={handleClear} disabled={loading}>
                <X size={16} />
                Clear
              </button>
            )}
          </div>
        </section>

        {error && (
          <section className="error-banner">
            <AlertTriangle size={22} />
            <div>
              <strong>Analysis Failed</strong>
              <p>{error}</p>
            </div>
          </section>
        )}

        {loading && (
          <section className="loading-state">
            <Loader2 size={34} className="spin-icon" />
            <p>Running Logistic Regression and Deep Neural Network models...</p>
          </section>
        )}

        {result && !loading && (
          <section className="results">
            <div className="summary-grid">
              <article className={`prediction-card ${result?.riskLevel || ''}`}>
                <div className="section-kicker">
                  {isFake ? <AlertTriangle size={22} /> : <CheckCircle2 size={22} />}
                  <span>Three-Class Prediction Result</span>
                </div>
                <h2>{resultClass}</h2>
                <p>{classificationDescription(resultClass)}</p>
                <div className="class-legend">
                  <span>Likely Legitimate</span>
                  <span>Suspicious</span>
                  <span>Likely Deceptive</span>
                </div>
              </article>

              {scoreCards.map((model) => (
                <article key={model.key} className={`risk-card ${model.level}`}>
                  <div className="risk-header">
                    <div className="section-kicker">
                      <Info size={22} />
                      <span>{model.title}</span>
                    </div>
                    <span className={`risk-pill ${model.level}`}>{riskLabel(model.level)}</span>
                  </div>
                  <div className="score-row">
                    <span className="score-number">{model.score}</span>
                    <span className="score-total">/ 100</span>
                  </div>
                  <div className={`classification-result ${model.level}`}>
                    {model.classification}
                  </div>
                  <div className="score-track">
                    <div className="score-fill" style={{ width: `${model.score}%` }} />
                  </div>
                  <div className="score-labels">
                    <span>0 - Safe</span>
                    <span>100 - Danger</span>
                  </div>
                  <div className="classification-tabs" aria-label={`${model.title} classification`}>
                    {CLASSIFICATION_OPTIONS.map((label) => (
                      <span
                        key={label}
                        className={label === model.classification ? `active ${model.level}` : ''}
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>

            <article className="explanation-card">
              <div className="section-title">
                <AlertTriangle size={24} />
                <h2>Explanation</h2>
              </div>
              <ul className="reason-list">
                {result.reasons.map((reason, index) => (
                  <li key={index} className={`reason-item ${isFake ? 'fake' : 'legit'}`}>
                    <span className="reason-arrow">›</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </article>

            <article className="tips-card">
              <div className="section-title">
                <BookOpen size={24} />
                <h2>How to Stay Safe</h2>
              </div>
              <ul className="tips-grid">
                {result.tips.map((tip, index) => (
                  <li key={tip} className="tip-item">
                    <span className="tip-num">{index + 1}</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </article>
          </section>
        )}

        {!result && !loading && !error && (
          <section className="empty-state">
            <Gift size={30} />
            <p>Paste a job advertisement and run both models to see separate risk scores.</p>
          </section>
        )}
      </main>
    </div>
  );
}
